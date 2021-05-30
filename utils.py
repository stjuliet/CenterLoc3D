# 包含工具函数

import math
import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from xml.etree import ElementTree as ET


def letterbox_image(image, size):
    """给图片加灰条，不失真缩放"""
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def pool_nms(heat, kernel=3):
    """最大池化的nms"""
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def nms(results, nms_threshold):
    """普通nms"""
    outputs = []
    # 对每一个图片进行处理
    for i in range(len(results)):

        detections = results[i]
        unique_class = np.unique(detections[:,-1])

        best_box = []
        if len(unique_class) == 0:
            results.append(best_box)
            continue

        #   对种类进行循环，
        #   非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框
        for c in unique_class:
            cls_mask = detections[:,-1] == c

            detection = detections[cls_mask]
            scores = detection[:,4]
            # 根据得分对该种类进行从大到小排序
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]
            while np.shape(detection)[0]>0:
                # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1],detection[1:])
                detection = detection[1:][ious<nms_threshold]
        outputs.append(best_box)
    return outputs


def iou(b1, b2):
    """二维框iou"""
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


def basic_iou(bbox_p, bbox_g):
    # 计算预测框面积
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # 计算真实框面积
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    # 计算预测框和真实框的交集面积
    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)
    
    # 计算预测框和真实框的并集面积
    union = area_p + area_g - intersection

    # 计算iou
    iou = intersection / np.maximum(union, 1e-6)

    return iou


def basic_giou(bbox_p, bbox_g):
    # 计算预测框面积
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # 计算真实框面积
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    # 计算预测框和真实框的交集面积
    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)
    
    # 计算预测框和真实框的并集面积
    union = area_p + area_g - intersection

    # 计算预测框和真实框并集的外接矩形
    x_min_union = np.minimum(bbox_p[0], bbox_g[0])
    y_min_union = np.minimum(bbox_p[1], bbox_g[1])
    x_max_union = np.maximum(bbox_p[2], bbox_g[2])
    y_max_union = np.maximum(bbox_p[3], bbox_g[3])
    external_rectangle = np.maximum(abs(x_max_union - x_min_union), 0) * np.maximum(abs(y_max_union - y_min_union), 0)

    # 计算iou
    iou = intersection / np.maximum(union, 1e-6)

    # 计算giou
    giou = iou - ((external_rectangle - union) / np.maximun(external_rectangle, 1e-6))

    return giou


def basic_diou(bbox_p, bbox_g):
    # 计算预测框面积
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # 计算真实框面积
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    center_p = np.array([(bbox_p[0]+bbox_p[2])/2., (bbox_p[1]+bbox_p[3])/2.], dtype=np.float32)
    center_g = np.array([(bbox_g[0]+bbox_g[2])/2., (bbox_g[1]+bbox_g[3])/2.], dtype=np.float32)

    # 计算中心点距离
    box_center_dis = (center_p[0] - center_g[0])**2 + (center_p[1] - center_g[1])**2

    # 计算预测框和真实框的交集面积
    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)
    
    # 计算预测框和真实框的并集面积
    union = area_p + area_g - intersection

    # 计算预测框和真实框并集的外接矩形的对角线距离
    x_min_union = np.minimum(bbox_p[0], bbox_g[0])
    y_min_union = np.minimum(bbox_p[1], bbox_g[1])
    x_max_union = np.maximum(bbox_p[2], bbox_g[2])
    y_max_union = np.maximum(bbox_p[3], bbox_g[3])

    external_dis = (x_max_union - x_min_union)**2 + (y_max_union - y_min_union)**2

    # 计算iou
    iou = intersection / np.maximum(union, 1e-6)
    
    # 计算diou
    diou = iou - box_center_dis / np.maximum(external_dis, 1e-6)

    return diou


def basic_ciou(bbox_p, bbox_g):
    # 计算预测框面积
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # 计算真实框面积
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    bbox_h_p = bbox_p[3] - bbox_p[1]
    bbox_w_p = bbox_p[2] - bbox_p[0]
    bbox_h_g = bbox_g[3] - bbox_g[1]
    bbox_w_g = bbox_g[2] - bbox_g[0]

    center_p = np.array([(bbox_p[0]+bbox_p[2])/2., (bbox_p[1]+bbox_p[3])/2.], dtype=np.float32)
    center_g = np.array([(bbox_g[0]+bbox_g[2])/2., (bbox_g[1]+bbox_g[3])/2.], dtype=np.float32)

    # 计算中心点距离
    box_center_dis = (center_p[0] - center_g[0])**2 + (center_p[1] - center_g[1])**2

    # 计算预测框和真实框的交集面积
    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)
    
    # 计算预测框和真实框的并集面积
    union = area_p + area_g - intersection

    # 计算预测框和真实框并集的外接矩形的对角线距离
    x_min_union = np.minimum(bbox_p[0], bbox_g[0])
    y_min_union = np.minimum(bbox_p[1], bbox_g[1])
    x_max_union = np.maximum(bbox_p[2], bbox_g[2])
    y_max_union = np.maximum(bbox_p[3], bbox_g[3])

    external_dis = (x_max_union - x_min_union)**2 + (y_max_union - y_min_union)**2

    # 计算iou
    iou = intersection / np.maximum(union, 1e-6)

    # 计算box长宽比惩罚项
    v = (4 / math.pi**2) * (math.atan(bbox_w_g/bbox_h_g) - math.atan(bbox_w_p/bbox_h_p))**2
    alpha = v / np.maximum((1 - iou + v), 1e-6)
    
    # 计算ciou
    ciou = iou - (box_center_dis / np.maximum(external_dis, 1e-6)) - v*alpha

    return ciou


def basic_3diou(b1, b2):
    """
    两个box在特定perspective下的3d iou计算
    left: x3_1 > x1_2
    right: x3_1 < x1_2
    b1, b2: [x0, y0, z0, x1, y1, z1, ... , x7, y7, z7]
    0: 0, 1, 2,
    1: 3, 4, 5,
    2: 6, 7, 8,
    3: 9, 10, 11,
    4: 12, 13, 14,
    5: 15, 16, 17,
    6: 18, 19, 20,
    7: 21, 22, 23,
    """
    if b1[9] < b2[3]:  # right
        min_x = np.maximum(b1[0], b2[0])
        max_x = np.minimum(b1[3], b2[3])
    else:  # left
        min_x = np.maximum(b1[3], b2[3])
        max_x = np.minimum(b1[0], b2[0])

    min_y = np.maximum(b1[1], b2[1])
    max_y = np.minimum(b1[10], b2[10])

    min_z = np.maximum(b1[2], b2[2])
    max_z = np.minimum(b1[14], b2[14])

    x_overlap = max_x - min_x
    y_overlap = max_y - min_y
    z_overlap = max_z - min_z

    if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
        overlap_volumn = x_overlap*y_overlap*z_overlap
        b1_volumn = abs(b1[3]-b1[0])*(b1[10]-b1[1])*(b1[14]-b1[2])
        b2_volumn = abs(b2[3]-b2[0])*(b2[10]-b2[1])*(b2[14]-b2[2])

        union_volumn = b1_volumn + b2_volumn - overlap_volumn
        iou = overlap_volumn / union_volumn
    else:
        iou = 0

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


def RDUVtoXYZ(CalibTMatrix, u, v, z):
    '''
    func: 图像坐标--->世界坐标, 世界坐标z需要指定
    世界坐标单位：mm
    '''
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


def RDXYZToUV(CalibTMatrix, x, y, z):
    '''
    func: 世界坐标--->图像坐标
    世界坐标单位：mm
    '''
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


def draw_bbox3d(draw, image, vertex, color, width):
    # 在图像中绘制3D box

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
    # 根据标定结果、基准点、车辆尺寸、视角计算出3D box的顶点在图像中坐标
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


def cal_pred_3dvertex(vertex_2d, h, m_trans):
    # 根据标定结果、2d顶点坐标、车辆高度，计算出3D box的顶点在世界中坐标
    w_p0 = RDUVtoXYZ(m_trans, vertex_2d[0], vertex_2d[1], 0)
    w_p1 = RDUVtoXYZ(m_trans, vertex_2d[2], vertex_2d[3], 0)
    w_p2 = RDUVtoXYZ(m_trans, vertex_2d[4], vertex_2d[5], 0)
    w_p3 = RDUVtoXYZ(m_trans, vertex_2d[6], vertex_2d[7], 0)
    w_p4 = RDUVtoXYZ(m_trans, vertex_2d[8], vertex_2d[9], h * 1000)
    w_p5 = RDUVtoXYZ(m_trans, vertex_2d[10], vertex_2d[11], h * 1000)
    w_p6 = RDUVtoXYZ(m_trans, vertex_2d[12], vertex_2d[13], h * 1000)
    w_p7 = RDUVtoXYZ(m_trans, vertex_2d[14], vertex_2d[15], h * 1000)
    return np.array([w_p0[0], w_p0[1], w_p0[2], w_p1[0], w_p1[1], w_p1[2],w_p2[0], w_p2[1], w_p2[2],
    w_p3[0], w_p3[1], w_p3[2],w_p4[0], w_p4[1], w_p4[2],w_p5[0], w_p5[1], w_p5[2],
    w_p6[0], w_p6[1], w_p6[2],w_p7[0], w_p7[1], w_p7[2]], dtype=np.float32)


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

        # 计算归一化中心点坐标 --- [0, 1]
        norm_center = torch.cat([xv_mask, yv_mask], dim=1)
        norm_center[:, 0] /= output_w
        norm_center[:, 1] /= output_h

        # 计算归一化box3D坐标 --- [0, 1]
        pred_vertex_mask[:, 0:16:2] = pred_vertex_mask[:, 0:16:2] / output_w
        pred_vertex_mask[:, 1:16:2] = pred_vertex_mask[:, 1:16:2] / output_h

        detect = torch.cat([norm_center, pred_vertex_mask, pred_size_mask, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)

        arg_sort = torch.argsort(detect[:,-2], descending=True)  # cls_conf倒序排列，便于筛选
        detect = detect[arg_sort]

        detects.append(detect.cpu().numpy()[:topk])
    return detects


def correct_vertex_norm2raw(norm_vertex, raw_image_shape):
    raw_img_h, raw_img_w = raw_image_shape
    if raw_img_h < raw_img_w:  # 扁图
        norm_vertex[:, 0:norm_vertex.shape[1]:2] = norm_vertex[:, 0:norm_vertex.shape[1]:2] * max(raw_img_h, raw_img_w)
        norm_vertex[:, 1:norm_vertex.shape[1]:2] = norm_vertex[:, 1:norm_vertex.shape[1]:2] * max(raw_img_h, raw_img_w) - abs(raw_img_h-raw_img_w)//2.
    else:  # 竖图
        norm_vertex[:, 0:norm_vertex.shape[1]:2] = norm_vertex[:, 0:norm_vertex.shape[1]:2] * max(raw_img_h, raw_img_w) - abs(raw_img_h-raw_img_w)//2.
        norm_vertex[:, 1:norm_vertex.shape[1]:2] = norm_vertex[:, 1:norm_vertex.shape[1]:2] * max(raw_img_h, raw_img_w)
    return norm_vertex


# -----------------------------高斯核函数-----------------------------------------------------#
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
# -----------------------------高斯核函数-----------------------------------------------------#
