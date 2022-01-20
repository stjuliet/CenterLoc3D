# multi loss
import torch
import torch.nn.functional as F
from utils import cal_pred_2dvertex, basic_iou, basic_giou, basic_diou, basic_ciou, basic_cdiou, get_vanish_point, get_distance_from_point_to_line
import numpy as np
import math


def focal_loss(pred, target):
    """
    pred: 预测值 [bt, xx, 128, 128]
    target: 真实值 [bt, 128, 128, xx]
    """
    # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    pred = pred.permute(0,2,3,1)

    # 找到每张图片的正样本和负样本
    # 一个真实框对应一个正样本
    # 除去正样本的特征点，其余为负样本
    pos_inds = target.eq(1).float()  # =
    neg_inds = target.lt(1).float()  # <
    # 正样本特征点附近的负样本的权值更小一些
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    # 计算focal loss
    # 难分类样本权重大，易分类样本权重小
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    # 进行损失的归一化
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
    pred: 预测值 [bt, xx, 128, 128]
    target: 真实值 [bt, 128, 128, xx]
    mask: 有目标的位置 [bt, 128, 128]
    index: 通道数维度扩张数
    """
    # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    pred = pred.permute(0,2,3,1)
    # [bt, 128, 128] -> [bt, 128, 128, index]
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,index)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def reproject_l1_loss(pred_vertex, calib_matrix, pred_box_size, mask, base_point, index, perspective, fp_size, input_size, raw_img_hs, raw_img_ws):
    """
    pred_vertex: 预测车辆三维框顶点坐标，相对于128*128特征图
    calib_matrix: 相机标定矩阵
    pred_box_size: 预测车辆物理尺寸
    mask: 是否有车辆目标
    base_point: 车辆基准点坐标，相对于原始宽高图像
    index: 通道数维度扩张数, 此处为三维框顶点坐标数量
    perspective: 车辆视角
    fp_size: 输出特征图尺寸 128*128
    input_size: resize后图像尺寸 512*512
    raw_img_hs: 原始图像高度
    raw_img_ws: 原始图像宽度
    """
    
    # 输出feature_map尺寸 128, 128
    featmap_h, featmap_w = fp_size
    # 输入feature_map尺寸 512, 512
    input_h, input_w = input_size[0], input_size[1]

    # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    pred_vertex = pred_vertex.permute(0,2,3,1)

    # 通过预测尺寸和原始base_point，计算顶点与预测顶点之差作为loss
    batch_size = pred_vertex.shape[0]
    batch_calc_vertex = []
    calc_vertex = np.zeros((featmap_h, featmap_w, 16), dtype=np.float32)
    pred_vertex_new = torch.zeros_like(pred_vertex)

    for b in range(batch_size):
        image_h, image_w = raw_img_hs[b], raw_img_ws[b]
        
        # pred是计算到特征图上的尺寸, 恢复到原始图像
        pred_vertex_new[b, :, :, 0:16:2] = pred_vertex[b, :, :, 0:16:2] * max(image_w, image_h) / featmap_h
        pred_vertex_new[b, :, :, 1:16:2] = pred_vertex[b, :, :, 1:16:2] * max(image_w, image_h) / featmap_h - abs(image_w-image_h)/2.
        
        calib_m = calib_matrix[b]
        t_list = np.where(mask[b].cpu() == 1.0)  # 遍历图像域中所有目标点
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
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,index)

    loss = F.l1_loss(pred_vertex_new * expand_mask, batch_calc_vertex * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def reg_iou_loss(iou_type, pred_vertex, target, mask, perspective, fp_size, input_size, raw_img_hs, raw_img_ws):
    """
    iou损失:
    常规L1 loss用于衡量预测坐标与真实坐标之间的距离，并没有考虑坐标之间的相关性，但是实际评价检测结果时用到的iou指标利用了坐标之间的相关性。
    因此，考虑设计iou loss用于更好地回归坐标。

    iou_type: iou类型, iou/giou/diou/ciou/cdiou
    pred_vertex: 预测车辆三维框顶点坐标，相对于128*128特征图
    target: 真实车辆三维框顶点坐标，相对于128*128特征图
    mask: 是否有车辆目标的位置
    perspective: 车辆视角
    fp_size: 输出特征图尺寸 128*128
    input_size: resize后图像尺寸 512*512
    raw_img_hs: 原始图像高度
    raw_img_ws: 原始图像宽度
    """
    # 将预测所得顶点坐标映射为2D box(相对于原图), 与真实标注框(相对于原图), 计算iou损失
    # details: https://arxiv.org/pdf/1902.09630.pdf

    featmap_h, featmap_w = fp_size  # feature_map尺寸 128, 128
    input_h, input_w = input_size[0], input_size[1]  # 512,512

    # pred是计算到特征图上的尺寸, 恢复到原始图像
    pred_vertex = pred_vertex.permute(0,2,3,1)

    batch_size = pred_vertex.shape[0]
    batch_calc_ious_loss = []
    batch_gt_ious_loss = []
    calc_ious_loss = np.zeros((featmap_h, featmap_w), dtype=np.float32)
    gt_ious_loss = np.zeros((featmap_h, featmap_w), dtype=np.float32)
    pred_vertex_new = torch.zeros_like(pred_vertex)
    target_new = torch.zeros_like(target)

    for b in range(batch_size):
        image_h, image_w = raw_img_hs[b], raw_img_ws[b]
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

    batch_calc_ious_loss = torch.unsqueeze(batch_calc_ious_loss,-1).repeat(1,1,1,1)
    batch_gt_ious_loss = torch.unsqueeze(batch_gt_ious_loss,-1).repeat(1,1,1,1)

    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,1)

    loss = F.l1_loss(batch_calc_ious_loss * expand_mask, batch_gt_ious_loss * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def reg_vp_loss(pred_vertex, target, mask, index, perspective, fp_size, input_size, raw_img_hs, raw_img_ws):
    """
    pred_vertex: 预测车辆三维框顶点坐标，相对于128*128特征图
    target: 真实车辆三维框顶点坐标，相对于128*128特征图
    mask: 是否有车辆目标
    index: 通道数维度扩张数, 此处为vp坐标数量
    perspective: 车辆视角
    fp_size: 输出特征图尺寸 128*128
    input_size: resize后图像尺寸 512*512
    raw_img_hs: 原始图像高度
    raw_img_ws: 原始图像宽度
    """
    
    # 输出feature_map尺寸 128, 128
    featmap_h, featmap_w = fp_size
    # 输入feature_map尺寸 512, 512
    input_h, input_w = input_size[0], input_size[1]

    # [bt, xx, 128, 128] -> [bt, 128, 128, xx]
    pred_vertex = pred_vertex.permute(0,2,3,1)

    batch_size = pred_vertex.shape[0]

    batch_gt_vp = []
    calc_gt_vp = np.zeros((featmap_h, featmap_w, 1), dtype=np.float32)
    batch_pred_vp = []
    calc_pred_vp = np.zeros((featmap_h, featmap_w, 1), dtype=np.float32)

    pred_vertex_new = torch.zeros_like(pred_vertex)
    target_new = torch.zeros_like(target)

    for b in range(batch_size):
        image_h, image_w = raw_img_hs[b], raw_img_ws[b]
        # calib_m = calib_matrix[b]
        # print(calib_m[0][0])
        
        # pred是计算到特征图上的尺寸, 恢复到原始图像
        pred_vertex_new[b, :, :, 0:16:2] = pred_vertex[b, :, :, 0:16:2] * max(image_w, image_h) / featmap_h
        pred_vertex_new[b, :, :, 1:16:2] = pred_vertex[b, :, :, 1:16:2] * max(image_w, image_h) / featmap_h - abs(image_w-image_h)/2.
        
        target_new[b, :, :, 0:16:2] = target[b, :, :, 0:16:2] * max(image_w, image_h) / featmap_h
        target_new[b, :, :, 1:16:2] = target[b, :, :, 1:16:2] * max(image_w, image_h) / featmap_h - abs(image_w-image_h)/2.
        
        t_list = np.where(mask[b].cpu() == 1.0)  # 遍历图像域中所有目标点
        for y, x in zip(t_list[0], t_list[1]):
            pers = perspective[b, y, x]
            # 0--3(0,1,6,7)
            k1_gt = (target_new[b, y, x, 7] - target_new[b, y, x, 1]) / (target_new[b, y, x, 6] - target_new[b, y, x, 0])
            b1_gt = target_new[b, y, x, 7] - k1_gt * target_new[b, y, x, 6]
            # 1--2(2,3,4,5)
            k2_gt = (target_new[b, y, x, 5] - target_new[b, y, x, 3]) / (target_new[b, y, x, 4] - target_new[b, y, x, 2])
            b2_gt = target_new[b, y, x, 5] - k2_gt * target_new[b, y, x, 4]
            # 4--7(8,9,14,15)
            k3_gt = (target_new[b, y, x, 15] - target_new[b, y, x, 9]) / (target_new[b, y, x, 14] - target_new[b, y, x, 8])
            b3_gt = target_new[b, y, x, 15] - k3_gt * target_new[b, y, x, 14]
            # 5--6(10,11,12,13)
            k4_gt = (target_new[b, y, x, 13] - target_new[b, y, x, 11]) / (target_new[b, y, x, 12] - target_new[b, y, x, 10])
            b4_gt = target_new[b, y, x, 13] - k4_gt * target_new[b, y, x, 12]

            # gt_vplines = [[k1_gt.cpu().detach().numpy(), b1_gt.cpu().detach().numpy()], [k2_gt.cpu().detach().numpy(), b2_gt.cpu().detach().numpy()], [k3_gt.cpu().detach().numpy(), b3_gt.cpu().detach().numpy()], [k4_gt.cpu().detach().numpy(), b4_gt.cpu().detach().numpy()]]

            k1_pred = (pred_vertex_new[b, y, x, 7] - pred_vertex_new[b, y, x, 1]) / (pred_vertex_new[b, y, x, 6] - pred_vertex_new[b, y, x, 0])
            b1_pred = pred_vertex_new[b, y, x, 7] - k1_pred * pred_vertex_new[b, y, x, 6]
            k2_pred = (pred_vertex_new[b, y, x, 5] - pred_vertex_new[b, y, x, 3]) / (pred_vertex_new[b, y, x, 4] - pred_vertex_new[b, y, x, 2])
            b2_pred = pred_vertex_new[b, y, x, 5] - k2_pred * pred_vertex_new[b, y, x, 4]
            k3_pred = (pred_vertex_new[b, y, x, 15] - pred_vertex_new[b, y, x, 9]) / (pred_vertex_new[b, y, x, 14] - pred_vertex_new[b, y, x, 8])
            b3_pred = pred_vertex_new[b, y, x, 15] - k3_pred * pred_vertex_new[b, y, x, 14]
            k4_pred = (pred_vertex_new[b, y, x, 13] - pred_vertex_new[b, y, x, 11]) / (pred_vertex_new[b, y, x, 12] - pred_vertex_new[b, y, x, 10])
            b4_pred = pred_vertex_new[b, y, x, 13] - k4_pred * pred_vertex_new[b, y, x, 12]

            # pred_lines = [[k1_pred.cpu().detach().numpy(), b1_pred.cpu().detach().numpy()], [k2_pred.cpu().detach().numpy(), b2_pred.cpu().detach().numpy()]]

            vp_gt = (k1_gt+k2_gt+k3_gt+k4_gt)/4
            vp_calc = (k1_pred+k2_pred+k3_pred+k4_pred)/4
            calc_gt_vp[y, x] = math.atan(vp_gt.cpu().detach().numpy())
            calc_pred_vp[y, x] = math.atan(vp_calc.cpu().detach().numpy())
            # if calib_m[0][0].item() == 3025.2532:
            #     calc_pred_vp[y, x] = get_distance_from_point_to_line(vp_calc, [144.737, 34.7794], [24380.6, 254.759])
            # if calib_m[0][0].item() == 4025.1802:
            #     calc_pred_vp[y, x] = get_distance_from_point_to_line(vp_calc, [812.616, -109.12], [57113.7, -63.7785])
            # if calib_m[0][0] == 3051.9802:
            #     calc_pred_vp[y, x] = get_distance_from_point_to_line(vp_calc, [1855.68, -373.44], [-20163.2, -74.5503])

        batch_gt_vp.append(calc_gt_vp)
        batch_pred_vp.append(calc_pred_vp)

    batch_gt_vp = torch.from_numpy(np.array(batch_gt_vp)).type(torch.FloatTensor).cuda()
    batch_pred_vp = torch.from_numpy(np.array(batch_pred_vp)).type(torch.FloatTensor).cuda()
    # [bt, 128, 128] -> [bt, 128, 128, index]
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,index)

    loss = F.l1_loss(batch_pred_vp * expand_mask, batch_gt_vp * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
