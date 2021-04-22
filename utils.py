
'''
包含工具函数

'''
import cv2 as cv
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from xml.etree import ElementTree as ET
from xml.dom.minidom import Document

# '''
# func: 计算地平线
# '''


# def CalVPLine(rd_vpx, rd_vpy, prd_vpx, prd_vpy):
#     k = (prd_vpy - rd_vpy) / (prd_vpx - rd_vpx)
#     return (k, rd_vpx, rd_vpy)


# '''
# func: 读入标定参数, 消失点
# '''


# def ReadCalibParam(calib_xml_path):
#     xml_dir = ET.parse(calib_xml_path)
#     focal = float(xml_dir.find('f').text)
#     fi = float(xml_dir.find('fi').text)
#     theta = float(xml_dir.find('theta').text)
#     cam_height = float(xml_dir.find('h').text)
#     list_vps = xml_dir.find('vanishPoints').text.split()
#     np_list_vps = np.array(list_vps).astype(np.float32)
#     rd_vpx = int(np_list_vps[0])
#     rd_vpy = int(np_list_vps[1])
#     prd_vpx = int(np_list_vps[2])
#     prd_vpy = int(np_list_vps[3])
#     vpline = CalVPLine(rd_vpx, rd_vpy, prd_vpx, prd_vpy)
#     return focal, fi, theta, cam_height, (rd_vpx, rd_vpy), vpline


# '''
# func: 将标定参数转换为变换矩阵(世界坐标y轴沿道路方向)
# '''


# def ParamToMatrix(focal, fi, theta, h, pcx, pcy):
#     K = np.array([focal, 0, pcx, 0, focal, pcy, 0, 0, 1]).reshape(3, 3).astype(np.float)
#     Rx = np.array([1, 0, 0, 0, -math.sin(fi), -math.cos(fi), 0, math.cos(fi), -math.sin(fi)]).reshape(3, 3).astype(np.float)
#     Rz = np.array([math.cos(theta), -math.sin(theta), 0, math.sin(theta), math.cos(theta), 0, 0, 0, 1]).reshape(3,3).astype(np.float)
#     R = np.dot(Rx, Rz)
#     T = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -h]).reshape(3, 4).astype(np.float)
#     trans = np.dot(R, T)
#     H = np.dot(K, trans)
#     return H

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def nms(results, nms):
    outputs = []
    # 对每一个图片进行处理
    for i in range(len(results)):
        #------------------------------------------------#
        #   具体过程可参考
        #   https://www.bilibili.com/video/BV1Lz411B7nQ
        #------------------------------------------------#
        detections = results[i]
        unique_class = np.unique(detections[:,-1])

        best_box = []
        if len(unique_class) == 0:
            results.append(best_box)
            continue
        #-------------------------------------------------------------------#
        #   对种类进行循环，
        #   非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
        #   对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
        #-------------------------------------------------------------------#
        for c in unique_class:
            cls_mask = detections[:,-1] == c

            detection = detections[cls_mask]
            scores = detection[:,4]
            # 根据得分对该种类进行从大到小排序。
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]
            while np.shape(detection)[0]>0:
                # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1],detection[1:])
                detection = detection[1:][ious<nms]
        outputs.append(best_box)
    return outputs

def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou


def calib_param_to_matrix(focal, fi, theta, h, pcx, pcy):
    """
    将标定参数转换为变换矩阵(世界坐标y轴沿道路方向)
    :param focal: 焦距
    :param fi: 俯仰角
    :param theta: 旋转角
    :param h: 相机高度
    :param pcx: 主点u
    :param pcy: 主点v
    :return: world -> image 变换矩阵
    """
    K = np.array([focal, 0, pcx, 0, focal, pcy, 0, 0, 1]).reshape(3, 3).astype(np.float)
    Rx = np.array([1, 0, 0, 0, -math.sin(fi), -math.cos(fi), 0, math.cos(fi), -math.sin(fi)]).reshape(3, 3).astype(np.float)
    Rz = np.array([math.cos(theta), -math.sin(theta), 0, math.sin(theta), math.cos(theta), 0, 0, 0, 1]).reshape(3,3).astype(np.float)
    R = np.dot(Rx, Rz)
    T = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -h]).reshape(3, 4).astype(np.float)
    trans = np.dot(R, T)
    H = np.dot(K, trans)
    return H


def read_calib_params(xml_path, width, height):
    """
    读入标定参数矩阵
    :param xml_path: 标定xml文件路径
    :param width: 图像宽度
    :param height: 图像高度
    :return: world -> image 变换矩阵
    """
    xml_dir = ET.parse(xml_path)  # 读入所有xml文件
    root = xml_dir.getroot()  # 根结点
    node_f = xml_dir.find('f')
    node_fi = xml_dir.find('fi')
    node_theta = xml_dir.find('theta')
    node_h = xml_dir.find('h')
    calib_matrix = calib_param_to_matrix(float(node_f.text), float(node_fi.text), float(node_theta.text),
                                float(node_h.text), width / 2, height / 2)
    return calib_matrix




'''
func: 图像坐标--->世界坐标, 世界坐标z需要指定
'''


def RDUVtoXYZ(CalibTMatrix, u, v, z):
    h11 = CalibTMatrix[0][0]
    h12 = CalibTMatrix[0][1]
    h13 = CalibTMatrix[0][2]
    h14 = CalibTMatrix[0][3]
    h21 = CalibTMatrix[1][0]
    h22 = CalibTMatrix[1][1]
    h23 = CalibTMatrix[1][2]
    h24 = CalibTMatrix[1][3]
    h31 = CalibTMatrix[2][0]
    h32 = CalibTMatrix[2][1]
    h33 = CalibTMatrix[2][2]
    h34 = CalibTMatrix[2][3]

    a11 = h11 - u * h31
    a12 = h12 - u * h32
    a21 = h21 - v * h31
    a22 = h22 - v * h32
    b1 = u * (h33 * z + h34) - (h13 * z + h14)  # 与之前版本有修改
    b2 = v * (h33 * z + h34) - (h23 * z + h24)
    x = (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21)
    y = (a11 * b2 - b1 * a21) / (a11 * a22 - a12 * a21)
    return (x, y, z)


'''
func: 世界坐标--->图像坐标
'''


def RDXYZToUV(CalibTMatrix, x, y, z):
    h11 = CalibTMatrix[0][0]
    h12 = CalibTMatrix[0][1]
    h13 = CalibTMatrix[0][2]
    h14 = CalibTMatrix[0][3]
    h21 = CalibTMatrix[1][0]
    h22 = CalibTMatrix[1][1]
    h23 = CalibTMatrix[1][2]
    h24 = CalibTMatrix[1][3]
    h31 = CalibTMatrix[2][0]
    h32 = CalibTMatrix[2][1]
    h33 = CalibTMatrix[2][2]
    h34 = CalibTMatrix[2][3]

    u = (h11 * x + h12 * y + h13 * z + h14) / (h31 * x + h32 * y + h33 * z + h34)  # 与之前版本有修改
    v = (h21 * x + h22 * y + h23 * z + h24) / (h31 * x + h32 * y + h33 * z + h34)
    return (int(u), int(v))

def dashLine(img, p1, p2, color, thickness, interval):
    '''绘制虚线'''
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    if p1[0] == p2[0]:
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
    len = math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
    k = (float)(p2[1] - p1[1]) / (float)(p2[0] - p1[0] + 0.000000000001)
    seg = (int)(len / (float)(2 * interval))
    dev_x = 2 * interval / math.sqrt(1 + k * k)
    dev_y = k * dev_x   # 短直线向量
    pend1 = (p1[0] + dev_x / 2, p1[1] + dev_y / 2)
    # 绘制虚线点
    for i in range(seg):
        pbeg = (round(p1[0] + dev_x * i), round(p1[1] + dev_y * i))
        pend = (round(pend1[0] + dev_x * i), round(pend1[1] + dev_y * i))
        cv.line(img, pbeg, pend, color, thickness)
    # 补齐最后一段
    plastBeg = (round(p1[0] + dev_x * seg), round(p1[1] + dev_y * seg))
    if plastBeg[0] < p2[0]:
        cv.line(img, plastBeg, p2, color, thickness)


def cal_3dbbox(perspective, m_trans, veh_base_point, veh_turple_vp, l, w, h):
    # 重绘3D bbox
    veh_world_base_point = RDUVtoXYZ(m_trans, veh_base_point[0], veh_base_point[1], 0)
    veh_world_vp = RDUVtoXYZ(m_trans, veh_turple_vp[0], veh_turple_vp[1], 0)
    k0 = (veh_world_vp[1] - veh_world_base_point[1]) / (veh_world_vp[0] - veh_world_base_point[0])
    k1 = -1.0 / k0
    dev_x0 = l / math.sqrt(1 + k0 * k0)  # 车长向量
    dev_y0 = k0 * dev_x0
    dev_x1 = w / math.sqrt(1 + k1 * k1)  # 车宽向量
    dev_y1 = k1 * dev_x1

    p1_3d = veh_world_base_point
    p5_3d = (p1_3d[0], p1_3d[1], h)
    if perspective == 'left':
        p0_3d = (p1_3d[0] + dev_x1, p1_3d[1] + dev_y1, 0)
        p2_3d = (p1_3d[0] - dev_x0, p1_3d[1] - dev_y0, 0)
        p3_3d = (p0_3d[0] - dev_x0, p0_3d[1] - dev_y0, 0)
        p4_3d = (p0_3d[0], p0_3d[1], h)
        p6_3d = (p2_3d[0], p2_3d[1], h)
        p7_3d = (p3_3d[0], p3_3d[1], h)
    else:
        p0_3d = (p1_3d[0] - dev_x1, p1_3d[1] - dev_y1, 0)
        p2_3d = (p1_3d[0] - dev_x0, p1_3d[1] - dev_y0, 0)
        p3_3d = (p0_3d[0] - dev_x0, p0_3d[1] - dev_y0, 0)
        p4_3d = (p0_3d[0], p0_3d[1], h)
        p6_3d = (p2_3d[0], p2_3d[1], h)
        p7_3d = (p3_3d[0], p3_3d[1], h)
    list_3dbbox_3dvertex = [p0_3d, p1_3d, p2_3d, p3_3d, p4_3d, p5_3d, p6_3d, p7_3d]
    list_3dbbox_2dvertex = []
    for i in range(len(list_3dbbox_3dvertex)):
        if i == 1:
            list_3dbbox_2dvertex.append(veh_base_point)
        else:
            list_3dbbox_2dvertex.append(RDXYZToUV(m_trans, list_3dbbox_3dvertex[i][0], list_3dbbox_3dvertex[i][1], list_3dbbox_3dvertex[i][2]))
    return list_3dbbox_2dvertex, list_3dbbox_3dvertex

def draw_bbox3d(draw, image, vertex, color, width):
    # 宽度方向
    # 0-1  2-3  4-5  6-7
    draw.line([vertex[0], vertex[1], vertex[2], vertex[3]], fill=128, width=2)
    draw.line([vertex[4], vertex[5], vertex[6], vertex[7]], fill=128, width=2)
    draw.line([vertex[8], vertex[9], vertex[10], vertex[11]], fill=128, width=2)
    draw.line([vertex[12], vertex[13], vertex[14], vertex[15]], fill=128, width=2)

    # 长度方向
    # 0-3 1-2 4-7 5-6
    draw.line([vertex[0], vertex[1], vertex[6], vertex[7]], fill=128, width=2)
    draw.line([vertex[2], vertex[3], vertex[4], vertex[5]], fill=128, width=2)
    draw.line([vertex[8], vertex[9], vertex[14], vertex[15]], fill=128, width=2)
    draw.line([vertex[10], vertex[11], vertex[12], vertex[13]], fill=128, width=2)

    # 高度方向
    # 0-4 1-5 2-6 3-7
    draw.line([vertex[0], vertex[1], vertex[8], vertex[9]], fill=128, width=2)
    draw.line([vertex[2], vertex[3], vertex[10], vertex[11]], fill=128, width=2)
    draw.line([vertex[4], vertex[5], vertex[12], vertex[13]], fill=128, width=2)
    draw.line([vertex[6], vertex[7], vertex[14], vertex[15]], fill=128, width=2)

def cal_pred_2dvertex(perspective, base_point, l, w, h, m_trans):
    w_p1 = RDUVtoXYZ(m_trans, base_point[0], base_point[1], 0)
    if perspective == 1:  # right view
        p0 = RDXYZToUV(m_trans, w_p1[0] - w * 1000, w_p1[1], w_p1[2])
        p2 = RDXYZToUV(m_trans, w_p1[0], w_p1[1] + l * 1000, w_p1[2])
        p3 = RDXYZToUV(m_trans, w_p1[0] - w * 1000, w_p1[1] + l * 1000, w_p1[2])
        p4 = RDXYZToUV(m_trans, w_p1[0] - w * 1000, w_p1[1], w_p1[2] + h * 1000)
        p5 = RDXYZToUV(m_trans, w_p1[0], w_p1[1], w_p1[2] + h * 1000)
        p6 = RDXYZToUV(m_trans, w_p1[0], w_p1[1] + l * 1000, w_p1[2] + h * 1000)
        p7 = RDXYZToUV(m_trans, w_p1[0] - w * 1000, w_p1[1] + l * 1000, w_p1[2] + h * 1000)
    else:
        p0 = RDXYZToUV(m_trans, w_p1[0] + w * 1000, w_p1[1], w_p1[2])
        p2 = RDXYZToUV(m_trans, w_p1[0], w_p1[1] + l * 1000, w_p1[2])
        p3 = RDXYZToUV(m_trans, w_p1[0] + w * 1000, w_p1[1] + l * 1000, w_p1[2])
        p4 = RDXYZToUV(m_trans, w_p1[0] + w * 1000, w_p1[1], w_p1[2] + h * 1000)
        p5 = RDXYZToUV(m_trans, w_p1[0], w_p1[1], w_p1[2] + h * 1000)
        p6 = RDXYZToUV(m_trans, w_p1[0], w_p1[1] + l * 1000, w_p1[2] + h * 1000)
        p7 = RDXYZToUV(m_trans, w_p1[0] + w * 1000, w_p1[1] + l * 1000, w_p1[2] + h * 1000)
    return np.array([p0[0], p0[1], base_point[0], base_point[1], p2[0], p2[1], p3[0], p3[1],
            p4[0], p4[1], p5[0], p5[1], p6[0], p6[1], p7[0], p7[1]], dtype=np.float32)

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_bbox(pred_hms, pred_center, pred_vertex, pred_size, image_size, threshold, cuda, topk=100):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128, 
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    for batch in range(b):
        #-------------------------------------------------------------------------#
        #   pred_hms        128*128, num_classes    热力图
        #   pred_center     128*128, 2              特征点的xy轴偏移情况
        #   pred_vertex     128*128, 16             特征点对应3D框顶点
        #   pred_size       128*128, 3              特征点对应3D尺寸
        #-------------------------------------------------------------------------#
        heat_map = pred_hms[batch].permute(1,2,0).view([-1,c])
        pred_center = pred_center[batch].permute(1,2,0).view([-1,2])
        pred_vertex = pred_vertex[batch].permute(1,2,0).view([-1,16])
        pred_size = pred_size[batch].permute(1,2,0).view([-1,3])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        #-------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        #-------------------------------------------------------------------------#
        xv, yv = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        #-------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        #   mask  大于置信度的位置
        #-------------------------------------------------------------------------#
        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > threshold

        #-----------------------------------------#
        #   取出得分筛选后对应的结果
        #-----------------------------------------#
        pred_center_mask = pred_center[mask]
        pred_vertex_mask = pred_vertex[mask]
        pred_size_mask = pred_size[mask]
        if len(pred_center_mask) == 0:
            detects.append([])
            continue     

        #----------------------------------------#
        #   计算调整后预测框的中心
        #----------------------------------------#
        xv_mask = torch.unsqueeze(xv[mask] + pred_center_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_center_mask[..., 1], -1)

        # 计算归一化中心点坐标
        norm_center = torch.cat([xv_mask, yv_mask], dim=1)
        norm_center[:, 0] /= output_w
        norm_center[:, 1] /= output_h

        # 计算归一化box3D坐标
        pred_vertex_mask[:, 0:16:2] = pred_vertex_mask[:, 0:16:2] / output_w
        pred_vertex_mask[:, 1:16:2] = pred_vertex_mask[:, 1:16:2] / output_h

        detect = torch.cat([norm_center, pred_vertex_mask, pred_size_mask, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)

        arg_sort = torch.argsort(detect[:,-2], descending=True)  # cls_conf倒序排列，便于筛选
        detect = detect[arg_sort]

        detects.append(detect.cpu().numpy()[:topk])
    return detects
