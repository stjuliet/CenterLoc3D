# multi loss
import torch
import torch.nn.functional as F
from utils import cal_pred_2dvertex, basic_iou, basic_giou, basic_ciou
import numpy as np


def focal_loss(pred, target):
    # pred: [bt, xx, 128, 128]
    pred = pred.permute(0,2,3,1)  # [bt, xx, 128, 128] -> [bt, 128, 128, xx]

    # 找到每张图片的正样本和负样本
    # 一个真实框对应一个正样本
    # 除去正样本的特征点，其余为负样本
    pos_inds = target.eq(1).float()  # 等于
    neg_inds = target.lt(1).float()  # 小于
    # 正样本特征点附近的负样本的权值更小一些
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    # 计算focal loss。难分类样本权重大，易分类样本权重小
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
    # 计算l1_loss
    pred = pred.permute(0,2,3,1)  # [bt, h, w, channel]
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,index)  # 最后一个2对应维度数，需要变化

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def reproject_l1_loss(pred_vertex, calib_matrix, pred_box_size, mask, base_point, index, perspective, fp_size, input_size, raw_img_hs, raw_img_ws):
    # 反投影损失
    # pred_vertex: 预测车辆三维框顶点坐标，相对于128*128特征图
    # calib_matrix: 相机标定矩阵
    # pred_box_size: 预测车辆物理尺寸
    # mask: 是否有车辆目标
    # base_point: 车辆基准点坐标，相对于原始宽高图像
    # index: 三维框顶点坐标数量
    # perspective: 车辆视角
    # fp_size: 输出特征图尺寸 128*128
    # input_size: resize后图像尺寸 512*512
    # raw_img_hs: 原始图像高度
    # raw_img_ws: 原始图像宽度
    
    featmap_h, featmap_w = fp_size  # feature_map尺寸 128, 128
    input_h, input_w = input_size[0], input_size[1]  # 512,512
    
    # pred是计算到特征图上的尺寸, 恢复到原始图像
    pred_vertex = pred_vertex.permute(0,2,3,1)

    # 通过预测尺寸和原始base_point，计算顶点与预测顶点之差作为loss
    batch_size = pred_vertex.shape[0]
    batch_calc_vertex = []
    calc_vertex = np.zeros((featmap_h, featmap_w, 16), dtype=np.float32)
    pred_vertex_new = torch.zeros_like(pred_vertex)

    for b in range(batch_size):
        image_h, image_w = raw_img_hs[b], raw_img_ws[b]
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

    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,index)  # 最后一个2对应维度数，需要变化

    loss = F.l1_loss(pred_vertex_new * expand_mask, batch_calc_vertex * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def reg_iou_loss(pred_vertex, target, mask, perspective, fp_size, input_size, raw_img_hs, raw_img_ws):
    """
    iou损失:
    常规L1 loss用于衡量预测坐标与真实坐标之间的距离，并没有考虑坐标之间的相关性，但是实际评价检测结果时用到的iou指标利用了坐标之间的相关性。
    因此，考虑设计iou loss用于更好地回归坐标。

    pred_vertex: 预测车辆三维框顶点坐标，相对于128*128特征图
    target: 真实车辆三维框顶点坐标，相对于128*128特征图
    mask: 是否有车辆目标的位置
    perspective: 车辆视角
    fp_size: 输出特征图尺寸 128*128
    input_size: resize后图像尺寸 512*512
    raw_img_hs: 原始图像高度
    raw_img_ws: 原始图像宽度
    """
    # 将预测所得顶点坐标映射为2D box(相对于原图), 与真实标注框(相对于原图), 计算giou损失
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
            # # 判断bbox_p中是否为负数
            # # 且是否左上角点坐标小于右下角点
            # if (bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0) and \
            #     (bbox[0] < bbox[2]) and (bbox[1] < bbox[3]):
            ciou = basic_ciou(bbox, bbox_gt)
            calc_ious_loss[y, x] = 1.0 - ciou
            # else:
            #     calc_ious_loss[y, x] = 1e11

        batch_calc_ious_loss.append(calc_ious_loss)
        batch_gt_ious_loss.append(gt_ious_loss)
    batch_calc_ious_loss = torch.from_numpy(np.array(batch_calc_ious_loss)).type(torch.FloatTensor).cuda()
    batch_gt_ious_loss = torch.from_numpy(np.array(batch_gt_ious_loss)).type(torch.FloatTensor).cuda()

    batch_calc_ious_loss = torch.unsqueeze(batch_calc_ious_loss,-1).repeat(1,1,1,1)
    batch_gt_ious_loss = torch.unsqueeze(batch_gt_ious_loss,-1).repeat(1,1,1,1)

    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,1)  # 最后一个2对应维度数，需要变化

    loss = F.l1_loss(batch_calc_ious_loss * expand_mask, batch_gt_ious_loss * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
