import torch
import torch.nn.functional as F
from utils import cal_pred_2dvertex
import numpy as np

def focal_loss(pred, target):
    pred = pred.permute(0,2,3,1)

    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()  # 等于
    neg_inds = target.lt(1).float()  # 小于
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, target, mask, index):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,1)  # [bt, h, w, channel]
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,index)  # 最后一个2对应维度数，需要变化

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

def reproject_l1_loss(pred_vertex, target_hm, calib_matrix, pred_box_size, mask, base_point, index, perspective, fp_size, input_size, raw_img_hs, raw_img_ws):

    featmap_h, featmap_w = fp_size  # feature_map尺寸 128, 128
    input_h, input_w = input_size[0], input_size[1]
    
    # pred是计算到特征图上的尺寸, 恢复到原始图像    
    pred_vertex = pred_vertex.permute(0,2,3,1)

    # 通过预测尺寸和原始base_point，计算顶点与预测顶点之差作为loss
    batch_size = pred_vertex.shape[0]
    batch_calc_vertex = []
    calc_vertex = np.zeros((featmap_h, featmap_w, 16), dtype=np.float32)    
    for b in range(batch_size):
        image_h, image_w = raw_img_hs[b], raw_img_ws[b]
        pred_vertex[b, :, :, 0:16:2] = pred_vertex[b, :, :, 0:16:2] * max(image_w, image_h) / featmap_h
        pred_vertex[b, :, :, 1:16:2] = pred_vertex[b, :, :, 1:16:2] * max(image_w, image_h) / featmap_h - abs(image_w-image_h)/2.
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

    loss = F.l1_loss(pred_vertex * expand_mask, batch_calc_vertex * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
