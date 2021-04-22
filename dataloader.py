# load datasets
import math
import cv2 as cv
import numpy as np
from random import shuffle

from PIL import Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils import draw_gaussian, gaussian_radius, read_calib_params


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class Bbox3dDatasets(Dataset):
    def __init__(self, train_lines, input_size, num_classes, is_train):
        super(Bbox3dDatasets, self).__init__()

        self.train_lines = train_lines
        self.input_size = input_size  # 512, 512
        self.output_size = (int(input_size[0]/4) , int(input_size[1]/4))  # 128, 128
        self.num_classes = num_classes
        self.is_train = is_train

    def __len__(self):
        return len(self.train_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    
    def letterbox_image(self, image, image_h, image_w, input_shape):
        featmap_w, featmap_h = input_shape
        scale = min(featmap_w/image_w, featmap_h/image_h)
        new_image_w = int(image_w*scale)
        new_image_h = int(image_h*scale)
        image = image.resize((new_image_w,new_image_h), Image.BICUBIC)
        new_image = Image.new('RGB', input_shape, (128,128,128))
        new_image.paste(image, ((featmap_w-new_image_w)//2, (featmap_h-new_image_h)//2))
        return new_image

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, augment=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        # line[0]:image, line[1]:calib_file_path, line[2:]:box_info
        image = Image.open(line[0])  # 原始图像
        image_w, image_h = image.size  # 原始图像尺寸
        featmap_h, featmap_w = input_shape  # feature_map尺寸
        scale = min(featmap_w/image_w, featmap_h/image_h)
        new_image_w = int(image_w*scale)
        new_image_h = int(image_h*scale)
        dx = (featmap_w-new_image_w)//2
        dy = (featmap_h-new_image_h)//2

        calib_matrix = read_calib_params(line[1], image_w, image_h)

        box_info = np.array([np.array(list(map(float,box_info.split(',')))) for box_info in line[2:]])  # [len(box_info), 29]

        # 从box_info中拆解出
        # left,top,width,height,cls_id,cx1,cy1,u0,v0,...,u7,v7,v_l,v_w,v_h,pers,bpx1,bpx2  (29 items)
        box2d_len = 4
        cls_len = box2d_len + 1
        box_center_len = cls_len + 2
        vertex_len = box_center_len + 16
        size_len = vertex_len + 3
        pers_len = size_len + 1
        bsp_len = pers_len + 2

        box2d = box_info[:, :box2d_len].astype(np.float32)
        box_cls = box_info[:, box2d_len:cls_len].astype(np.float32)
        box_center = box_info[:, cls_len:box_center_len].astype(np.float32)
        box_vertex = box_info[:, box_center_len:vertex_len].astype(np.float32)
        box_size = box_info[:, vertex_len:size_len].astype(np.float32)
        box_perspective = box_info[:, size_len:pers_len].astype(np.float32)
        box_base_point = box_info[:, pers_len:bsp_len].astype(np.float32)

        # correct boxes (box2d, box_center, box_vertex, box_base_point)
        correct_box2d = np.zeros((len(box_info), 4))
        correct_box_center = np.zeros((len(box_info), 2))
        correct_box_vertex = np.zeros((len(box_info), 16))
        # correct_box_base_point = np.zeros((len(box_info), 2))
        # raw_box_base_point = np.zeros((len(box_info), 2))

        # raw_box_base_point[:len(box_info)] = box_base_point

        # resize image (加灰条)
        image = self.letterbox_image(image, image_h, image_w, input_shape)

        if augment: # 如果进行数据增强(色域变换/随机水平翻转)
            # 随机水平翻转
            # flip = rand()<.5
            flip = False
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

                if len(box_info) > 0:
                    box2d[:, 0] = featmap_w - ((box2d[:, 0] + box2d[:, 2]) * new_image_w/image_w + dx)
                    box2d[:, 1] = box2d[:, 1] * new_image_h/image_h + dy
                    box2d[:, 2] = box2d[:, 2] * featmap_w/max(image_w, image_h)
                    box2d[:, 3] = box2d[:, 3] * featmap_h/max(image_w, image_h)
                    correct_box2d[:len(box_info)] = box2d

                    box_center[:, 0] = featmap_w - (box_center[:, 0] * new_image_w/image_w + dx)
                    box_center[:, 1] = box_center[:, 1] * new_image_h/image_h + dy
                    correct_box_center[:len(box_info)] = box_center

                    box_vertex[:, 0:16:2] = featmap_w - (box_vertex[:, 0:16:2] * new_image_w/image_w + dx)  # x
                    box_vertex[:, 1:16:2] = box_vertex[:, 1:16:2] * new_image_h/image_h + dy  # y
                    correct_box_vertex[:len(box_info)] = box_vertex

                    # box_base_point[:, 0] = featmap_w - (box_base_point[:, 0] * new_image_w/image_w + dx)
                    # box_base_point[:, 1] = box_base_point[:, 1] * new_image_h/image_h + dy
                    # correct_box_base_point[:len(box_info)] = box_base_point
            else:
                if len(box_info) > 0:
                    box2d[:, 0] = box2d[:, 0] * new_image_w/image_w + dx
                    box2d[:, 1] = box2d[:, 1] * new_image_h/image_h + dy
                    box2d[:, 2] = box2d[:, 2] * featmap_w/max(image_w, image_h)
                    box2d[:, 3] = box2d[:, 3] * featmap_h/max(image_w, image_h)
                    correct_box2d[:len(box_info)] = box2d

                    box_center[:, 0] = box_center[:, 0] * new_image_w/image_w + dx
                    box_center[:, 1] = box_center[:, 1] * new_image_h/image_h + dy
                    correct_box_center[:len(box_info)] = box_center

                    box_vertex[:, 0:16:2] = box_vertex[:, 0:16:2] * new_image_w/image_w + dx  # x
                    box_vertex[:, 1:16:2] = box_vertex[:, 1:16:2] * new_image_h/image_h + dy  # y
                    correct_box_vertex[:len(box_info)] = box_vertex

                    # box_base_point[:, 0] = box_base_point[:, 0] * new_image_w/image_w + dx
                    # box_base_point[:, 1] = box_base_point[:, 1] * new_image_h/image_h + dy
                    # correct_box_base_point[:len(box_info)] = box_base_point

            # 色域变换
            hue = rand(-hue, hue)
            sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
            val = rand(1, val) if rand()<.5 else 1/rand(1, val)
            x = cv.cvtColor(np.array(image,np.float32)/255, cv.COLOR_RGB2HSV)
            x[..., 0] += hue*360
            x[..., 0][x[..., 0]>1] -= 1
            x[..., 0][x[..., 0]<0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:,:, 0]>360, 0] = 360
            x[:, :, 1:][x[:, :, 1:]>1] = 1
            x[x<0] = 0
            image = cv.cvtColor(x, cv.COLOR_HSV2RGB)*255

            return image, calib_matrix, correct_box2d, box_cls, correct_box_center, correct_box_vertex, box_size, box_perspective, box_base_point, image_w, image_h
        else:
            if len(box_info) > 0:
                box2d[:, 0] = box2d[:, 0] * new_image_w/image_w + dx
                box2d[:, 1] = box2d[:, 1] * new_image_h/image_h + dy
                box2d[:, 2] = box2d[:, 2] * featmap_w/max(image_w, image_h)
                box2d[:, 3] = box2d[:, 3] * featmap_h/max(image_w, image_h)
                correct_box2d[:len(box_info)] = box2d

                box_center[:, 0] = box_center[:, 0] * new_image_w/image_w + dx
                box_center[:, 1] = box_center[:, 1] * new_image_h/image_h + dy
                correct_box_center[:len(box_info)] = box_center

                box_vertex[:, 0:16:2] = box_vertex[:, 0:16:2] * new_image_w/image_w + dx  # x
                box_vertex[:, 1:16:2] = box_vertex[:, 1:16:2] * new_image_h/image_h + dy  # y
                correct_box_vertex[:len(box_info)] = box_vertex

                # box_base_point[:, 0] = box_base_point[:, 0] * new_image_w/image_w + dx
                # box_base_point[:, 1] = box_base_point[:, 1] * new_image_h/image_h + dy
                # correct_box_base_point[:len(box_info)] = box_base_point
            
            return image, calib_matrix, correct_box2d, box_cls, correct_box_center, correct_box_vertex, box_size, box_perspective, box_base_point, image_w, image_h

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines

        # 进行数据增强
        image_data, calib_matrix, correct_box2d, box_cls, correct_box_center, correct_box_vertex, box_size, box_perspective, box_base_point, raw_img_w, raw_img_h = self.get_random_data(lines[index], [self.input_size[0],self.input_size[1]], augment=self.is_train)
        
        batch_hm = np.zeros((self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
        batch_center_reg = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_vertex_reg = np.zeros((self.output_size[0], self.output_size[1], 16), dtype=np.float32)
        batch_size_reg = np.zeros((self.output_size[0], self.output_size[1], 3), dtype=np.float32)
        batch_box_perspective = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
        batch_base_point = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_center_mask = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
        
        if len(box_cls) != 0: # 如果有目标
            # 转换成相对于特征层的大小 !!!
            # 取出宽高(相对于特征层), 用于绘制热力图
            fp_boxes = np.array(correct_box2d, dtype=np.float32)
            fp_boxes[:,0] = fp_boxes[:,0] / self.input_size[1] * self.output_size[1]
            fp_boxes[:,1] = fp_boxes[:,1] / self.input_size[0] * self.output_size[0]
            fp_boxes[:,2] = fp_boxes[:,2] / self.input_size[1] * self.output_size[1]
            fp_boxes[:,3] = fp_boxes[:,3] / self.input_size[0] * self.output_size[0]

            # 取出中心点
            fp_box_center = np.array(correct_box_center, dtype=np.float32)
            fp_box_center[:,0] = fp_box_center[:,0] / self.input_size[1] * self.output_size[1]
            fp_box_center[:,1] = fp_box_center[:,1] / self.input_size[0] * self.output_size[0]

            # 取出顶点
            fp_box_vertex = np.array(correct_box_vertex, dtype=np.float32)
            fp_box_vertex[:,0:16:2] = fp_box_vertex[:,0:16:2] / self.input_size[1] * self.output_size[1]
            fp_box_vertex[:,1:16:2] = fp_box_vertex[:,1:16:2] / self.input_size[0] * self.output_size[0]

        for i in range(len(box_cls)):
            fp_box_center_copy = fp_box_center[i].copy()
            fp_box_center_copy = np.array(fp_box_center_copy)
            # 防止中心点超出特征层的范围
            fp_box_center_copy[0] = np.clip(fp_box_center_copy[0], 0, self.output_size[1] - 1)
            fp_box_center_copy[1] = np.clip(fp_box_center_copy[1], 0, self.output_size[0] - 1)

            box_cls_id = int(box_cls[i])

            # 计算每个目标对应3D box的最小外接矩形宽高, 作为热力图半径 !!!
            fp_box_w, fp_box_h = abs(fp_box_vertex[i, 2] - fp_box_vertex[i, 6]), abs(fp_box_vertex[i, 1] - fp_box_vertex[i, 13])
            if fp_box_w > 0 and fp_box_h > 0:
                radius = gaussian_radius((math.ceil(fp_box_h), math.ceil(fp_box_w)))
                radius = max(0, int(radius))
                # 计算真实框所属的特征点
                ct = np.array([fp_box_center_copy[0], fp_box_center_copy[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # 绘制高斯热力图
                batch_hm[:, :, box_cls_id] = draw_gaussian(batch_hm[:, :, box_cls_id], ct_int, radius)
                batch_center_reg[ct_int[1], ct_int[0]] = ct - ct_int
                batch_vertex_reg[ct_int[1], ct_int[0]] = fp_box_vertex[i]
                batch_size_reg[ct_int[1], ct_int[0]] = box_size[i]
                batch_box_perspective[ct_int[1], ct_int[0]] = box_perspective[i]
                batch_base_point[ct_int[1], ct_int[0]] = box_base_point[i]
                batch_center_mask[ct_int[1], ct_int[0]] = 1

        img = np.array(image_data, dtype=np.float32)[:,:,::-1]  # RGB -> BGR
        img = np.transpose(preprocess_image(img), (2, 0, 1))  # BGR -> RGB

        return img, calib_matrix, batch_hm, batch_center_reg, batch_vertex_reg, batch_size_reg, batch_center_mask, batch_box_perspective, batch_base_point, raw_img_w, raw_img_h


# DataLoader中collate_fn使用
def bbox3d_dataset_collate(batch):
    imgs, calib_matrixs, batch_hms, batch_center_regs, batch_vertex_regs, batch_size_regs, batch_center_masks, batch_box_perspectives, batch_base_points, raw_img_ws, raw_img_hs = [], [], [], [], [], [], [], [], [], [], []

    for img, calib_matrix, batch_hm, batch_center_reg, batch_vertex_reg, batch_size_reg, batch_center_mask, batch_box_perspective, batch_base_point, raw_img_w, raw_img_h in batch:
        imgs.append(img)
        calib_matrixs.append(calib_matrix)
        batch_hms.append(batch_hm)
        batch_center_regs.append(batch_center_reg)
        batch_vertex_regs.append(batch_vertex_reg)
        batch_size_regs.append(batch_size_reg)
        batch_center_masks.append(batch_center_mask)
        batch_box_perspectives.append(batch_box_perspective)
        batch_base_points.append(batch_base_point)
        raw_img_hs.append(raw_img_h)
        raw_img_ws.append(raw_img_w)

    imgs = np.array(imgs)
    calib_matrixs = np.array(calib_matrixs)
    batch_hms = np.array(batch_hms)
    batch_center_regs = np.array(batch_center_regs)
    batch_vertex_regs = np.array(batch_vertex_regs)
    batch_size_regs = np.array(batch_size_regs)
    batch_center_masks = np.array(batch_center_masks)
    batch_box_perspectives = np.array(batch_box_perspectives)
    batch_base_points = np.array(batch_base_points)
    raw_img_hs = np.array(raw_img_hs)
    raw_img_ws = np.array(raw_img_ws)
    return imgs, calib_matrixs, batch_hms, batch_center_regs, batch_vertex_regs, batch_size_regs, batch_center_masks, batch_box_perspectives, batch_base_points, raw_img_ws, raw_img_hs
