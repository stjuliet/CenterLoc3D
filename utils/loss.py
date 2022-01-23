# multi loss
import torch
import torch.nn.functional as F
from utils.utils import cal_pred_2dvertex, basic_iou, basic_giou, basic_diou, basic_ciou, basic_cdiou
import numpy as np


def focal_loss(pred, target):
    """
    pred: [bt, xx, 128, 128]
    target: gt [bt, 128, 128, xx]
    """
    # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    pred = pred.permute(0,2,3,1)

    # find positive samples
    pos_inds = target.eq(1).float()  # =
    neg_inds = target.lt(1).float()  # <
    # lower weights close to positive samples
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    # high weight for hard sample, low weight for easy sample
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    # loss normalization
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, target, mask, index):
    """
    pred: [bt, xx, 128, 128]
    target: gt [bt, 128, 128, xx]
    mask: [bt, 128, 128] (object: 1, no object: 0)
    index: output channels
    """
    # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    pred = pred.permute(0, 2, 3, 1)
    # [bt, 128, 128] -> [bt, 128, 128, index]
    expand_mask = torch.unsqueeze(mask,-1).repeat(1, 1, 1, index)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def reproject_l1_loss(pred_vertex, calib_matrix, pred_box_size, mask, base_point, index, perspective, fp_size, input_size, raw_img_hs, raw_img_ws):
    """
    pred_vertex: prediction vertex of 3d box in img (relative to 128*128 feature map)
    calib_matrix: 3*4 array
    pred_box_size: vehicle size
    mask: (object: 1, no object: 0)
    base_point: (relative to raw img h,w)
    index: output channels: number of vertex of 3d box
    perspective: vehicle view
    fp_size: output feature map size (128*128)
    input_size: input img size (512*512)
    raw_img_hs: raw img h
    raw_img_ws: raw img w
    """
    
    # 128, 128
    featmap_h, featmap_w = fp_size
    # 512, 512
    input_h, input_w = input_size[0], input_size[1]

    # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    pred_vertex = pred_vertex.permute(0, 2, 3, 1)

    # use predict vehicle size and base_point
    # to calculate projection loss
    batch_size = pred_vertex.shape[0]
    batch_calc_vertex = []
    calc_vertex = np.zeros((featmap_h, featmap_w, 16), dtype=np.float32)
    pred_vertex_new = torch.zeros_like(pred_vertex)

    for b in range(batch_size):
        image_h, image_w = raw_img_hs[b], raw_img_ws[b]
        
        # relative to : 128*128 -> raw img h,w
        pred_vertex_new[b, :, :, 0:16:2] = pred_vertex[b, :, :, 0:16:2] * max(image_w, image_h) / featmap_h
        pred_vertex_new[b, :, :, 1:16:2] = pred_vertex[b, :, :, 1:16:2] * max(image_w, image_h) / featmap_h - abs(image_w-image_h)/2.
        
        calib_m = calib_matrix[b]
        t_list = np.where(mask[b].cpu() == 1.0)  # all object
        for y, x in zip(t_list[0], t_list[1]):
            pers = perspective[b, y, x]
            bp = base_point[b, y, x]
            l = pred_box_size[b, 0, y, x]
            w = pred_box_size[b, 1, y, x]
            h = pred_box_size[b, 2, y, x]
            calc_vertex[y, x] = cal_pred_2dvertex(pers, bp, l, w, h, calib_m)
        batch_calc_vertex.append(calc_vertex)

    batch_calc_vertex = torch.from_numpy(np.array(batch_calc_vertex)).type(torch.FloatTensor).cuda()
    # [bt, 128, 128] -> [bt, 128, 128, index]
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, index)

    loss = F.l1_loss(pred_vertex_new * expand_mask, batch_calc_vertex * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def reg_iou_loss(iou_type, pred_vertex, target, mask, perspective, fp_size, input_size, raw_img_hs, raw_img_ws):
    """
    ref: https://arxiv.org/pdf/1902.09630.pdf
    common l1 loss: measure distance between gt and pred coordinates, without considering correlation between coordinates
    iou loss: consider correlation between coordinates, treat coordinates as complete structure

    iou_type: iou/giou/diou/ciou/cdiou
    pred_vertex: prediction vertex of 3d box in img (relative to 128*128 feature map)
    target: gt vertex of 3d box in img (relative to 128*128 feature map)
    mask: (object: 1, no object: 0)
    fp_size: output feature map size (128*128)
    input_size: input img size (512*512)
    raw_img_hs: raw img h
    raw_img_ws: raw img w
    """
    # use predict vertex of 3d box in img to generate minimum enclosing rectangle (relative to raw img)
    # and gt 2d box
    # to calculate iou loss

    # 128, 128
    featmap_h, featmap_w = fp_size
    # 512, 512
    input_h, input_w = input_size[0], input_size[1]

    # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    pred_vertex = pred_vertex.permute(0, 2, 3, 1)

    batch_size = pred_vertex.shape[0]
    batch_calc_ious_loss = []
    batch_gt_ious_loss = []
    calc_ious_loss = np.zeros((featmap_h, featmap_w), dtype=np.float32)
    gt_ious_loss = np.zeros((featmap_h, featmap_w), dtype=np.float32)
    pred_vertex_new = torch.zeros_like(pred_vertex)
    target_new = torch.zeros_like(target)

    for b in range(batch_size):
        image_h, image_w = raw_img_hs[b], raw_img_ws[b]
        # relative to : 128*128 -> raw img h,w
        pred_vertex_new[b, :, :, 0:16:2] = pred_vertex[b, :, :, 0:16:2] * max(image_w, image_h) / featmap_h
        pred_vertex_new[b, :, :, 1:16:2] = pred_vertex[b, :, :, 1:16:2] * max(image_w, image_h) / featmap_h - abs(image_w-image_h)/2.

        target_new[b, :, :, 0:16:2] = target[b, :, :, 0:16:2] * max(image_w, image_h) / featmap_h
        target_new[b, :, :, 1:16:2] = target[b, :, :, 1:16:2] * max(image_w, image_h) / featmap_h - abs(image_w-image_h)/2.

        t_list = np.where(mask[b].cpu() == 1.0)  # 遍历图像域中所有目标点
        for y, x in zip(t_list[0], t_list[1]):
            pers = perspective[b, y, x]
            if pers == 1:   # right view
                bbox = np.array([pred_vertex_new[b,y,x,14], pred_vertex_new[b,y,x,15], pred_vertex_new[b,y,x,2], pred_vertex_new[b,y,x,3]], dtype=np.float32)
                bbox_gt = np.array([target_new[b,y,x,14], target_new[b,y,x,15], target_new[b,y,x,2], target_new[b,y,x,3]], dtype=np.float32)
            else:   # left view
                bbox = np.array([pred_vertex_new[b,y,x,2], pred_vertex_new[b,y,x,3], pred_vertex_new[b,y,x,14], pred_vertex_new[b,y,x,15]], dtype=np.float32)
                bbox_gt = np.array([target_new[b,y,x,2], target_new[b,y,x,3], target_new[b,y,x,14], target_new[b,y,x,15]], dtype=np.float32)

            if iou_type == "iou":
                iou = basic_iou(bbox, bbox_gt)
                calc_ious_loss[y, x] = 1.0 - iou
            elif iou_type == "giou":
                giou = basic_giou(bbox, bbox_gt)
                calc_ious_loss[y, x] = 1.0 - giou
            elif iou_type == "diou":
                diou = basic_diou(bbox, bbox_gt)
                calc_ious_loss[y, x] = 1.0 - diou
            elif iou_type == "ciou":
                ciou = basic_ciou(bbox, bbox_gt)
                calc_ious_loss[y, x] = 1.0 - ciou
            elif iou_type == "cdiou":
                cdiou = basic_cdiou(pred_vertex_new[b,y,x], target_new[b,y,x], bbox, bbox_gt)
                calc_ious_loss[y, x] = 1.0 - cdiou
            else:
                pass

        batch_calc_ious_loss.append(calc_ious_loss)
        batch_gt_ious_loss.append(gt_ious_loss)

    batch_calc_ious_loss = torch.from_numpy(np.array(batch_calc_ious_loss)).type(torch.FloatTensor).cuda()
    batch_gt_ious_loss = torch.from_numpy(np.array(batch_gt_ious_loss)).type(torch.FloatTensor).cuda()

    batch_calc_ious_loss = torch.unsqueeze(batch_calc_ious_loss, -1).repeat(1, 1, 1, 1)
    batch_gt_ious_loss = torch.unsqueeze(batch_gt_ious_loss, -1).repeat(1, 1, 1, 1)

    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 1)

    loss = F.l1_loss(batch_calc_ious_loss * expand_mask, batch_gt_ious_loss * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
