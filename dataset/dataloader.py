# load datasets
import math
import cv2 as cv
import numpy as np
from random import shuffle

import platform
from PIL import Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils.utils import draw_gaussian, gaussian_radius, read_calib_params, coord_to_homog, coord_to_normal


class Bbox3dDatasets(Dataset):
    def __init__(self, train_lines, input_size, num_classes, is_train):
        super(Bbox3dDatasets, self).__init__()
        self.train_lines = train_lines
        self.input_size = input_size  # 512, 512
        self.output_size = (int(input_size[0]/4), int(input_size[1]/4))  # 128, 128
        self.num_classes = num_classes
        self.is_train = is_train

    def __len__(self):
        return len(self.train_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    
    def preprocess_image(self, image):
        mean = [0.40789655, 0.44719303, 0.47026116]
        std = [0.2886383, 0.27408165, 0.27809834]
        return ((np.float32(image) / 255.) - mean) / std
    
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
        """实时数据增强的随机预处理"""
        line = annotation_line.split()
        # line[0]:image, line[1]:calib_file_path, line[2:]:box_info
        image = Image.open(line[0])  # raw img
        image_w, image_h = image.size  # raw img h,w
        featmap_h, featmap_w = input_shape  # feature_map h,w
        scale = min(featmap_w/image_w, featmap_h/image_h)
        new_image_w = int(image_w*scale)
        new_image_h = int(image_h*scale)
        dx = (featmap_w-new_image_w)//2
        dy = (featmap_h-new_image_h)//2

        calib_matrix = read_calib_params(line[1], image_w, image_h)

        box_info = np.array([np.array(list(map(float,box_info.split(',')))) for box_info in line[2:]])  # [len(box_info), 29]

        if platform.system() == "Linux":
            if line[1].split("/")[-1].split("_")[0] + line[1].split("/")[-1].split("_")[1] == "session0centre":
                pers_control_points = np.array([475, 830, 275, 330, 590, 370, 1345, 830]).astype(np.float32).reshape(-1, 2)
            elif line[1].split("/")[-1].split("_")[0] + line[1].split("/")[-1].split("_")[1] == "session0right":
                pers_control_points = np.array([285, 775, 630, 215, 1025, 215, 1340, 775]).astype(np.float32).reshape(-1, 2)
            elif line[1].split("/")[-1].split("_")[0] + line[1].split("/")[-1].split("_")[1] == "session6right":
                pers_control_points = np.array([200, 760, 1140, 115, 1765, 115, 1660, 760]).astype(np.float32).reshape(-1, 2)
            elif line[1].split("/")[-1].split("_")[0] == "0cam":
                pers_control_points = np.array([227, 663, 670, 197, 920, 197, 915, 667]).astype(np.float32).reshape(-1, 2)
            elif line[1].split("/")[-1].split("_")[0] == "1cam":
                pers_control_points = np.array([45, 684, 323, 206, 550, 206, 716, 684]).astype(np.float32).reshape(-1, 2)
        elif platform.system() == "Windows":
            if line[1].split("\\")[-1].split("_")[0] + line[1].split("\\")[-1].split("_")[1] == "session0centre":
                pers_control_points = np.array([475, 830, 275, 330, 590, 370, 1345, 830]).astype(np.float32).reshape(-1, 2)
            elif line[1].split("\\")[-1].split("_")[0] + line[1].split("\\")[-1].split("_")[1] == "session0right":
                pers_control_points = np.array([285, 775, 630, 215, 1025, 215, 1340, 775]).astype(np.float32).reshape(-1, 2)
            elif line[1].split("\\")[-1].split("_")[0] + line[1].split("\\")[-1].split("_")[1] == "session6right":
                pers_control_points = np.array([200, 760, 1140, 115, 1765, 115, 1660, 760]).astype(np.float32).reshape(-1, 2)
            elif line[1].split("\\")[-1].split("_")[0] == "0cam":
                pers_control_points = np.array([227, 663, 670, 197, 920, 197, 915, 667]).astype(np.float32).reshape(-1, 2)
            elif line[1].split("\\")[-1].split("_")[0] == "1cam":
                pers_control_points = np.array([45, 684, 323, 206, 550, 206, 716, 684]).astype(np.float32).reshape(-1, 2)
        else:
            raise ValueError("unsupported system!")

        # from box_info
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

        # correct boxes (box2d, box_center, box_vertex)
        correct_box2d = np.zeros((len(box_info), 4))
        correct_box_center = np.zeros((len(box_info), 2))
        correct_box_vertex = np.zeros((len(box_info), 16))

        # resize image (without deform)
        image = self.letterbox_image(image, image_h, image_w, input_shape)  # RGB, HWC

        if len(box_info) > 0:
            box2d[:, 0] = box2d[:, 0] * new_image_w / image_w + dx
            box2d[:, 1] = box2d[:, 1] * new_image_h / image_h + dy
            box2d[:, 2] = box2d[:, 2] * featmap_w / max(image_w, image_h)
            box2d[:, 3] = box2d[:, 3] * featmap_h / max(image_w, image_h)
            correct_box2d[:len(box_info)] = box2d

            box_center[:, 0] = box_center[:, 0] * new_image_w / image_w + dx
            box_center[:, 1] = box_center[:, 1] * new_image_h / image_h + dy
            correct_box_center[:len(box_info)] = box_center

            box_vertex[:, 0:16:2] = box_vertex[:, 0:16:2] * new_image_w / image_w + dx  # x
            box_vertex[:, 1:16:2] = box_vertex[:, 1:16:2] * new_image_h / image_h + dy  # y
            correct_box_vertex[:len(box_info)] = box_vertex

            pers_control_points[:, 0] = pers_control_points[:, 0] * new_image_w / image_w + dx  # x
            pers_control_points[:, 1] = pers_control_points[:, 1] * new_image_h / image_h + dy  # y

        if augment:  # data augmentation
            # random flip
            flip = False
            # random rotation
            rotation = False
            # random color jitter
            color_change = self.rand() < .5
            # random transformation
            persp_transform = False

            if persp_transform:
                cv_img = np.array(image, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR, HWC
                shift = np.random.randint(-5, 5, [4, 2])
                pers_boxes, pers_center_points, pers_vertex = [], [], []
                if len(box_info) > 0:
                    pers_control_points_aft = pers_control_points + np.array(shift).astype(np.float32)

                    pers_trans_m = cv.getPerspectiveTransform(pers_control_points, pers_control_points_aft)
                    pers_trans_cv_img = cv.warpPerspective(cv_img, pers_trans_m, (cv_img.shape[1], cv_img.shape[0]))
                    image = Image.fromarray(np.uint8(pers_trans_cv_img[:, :, ::-1]))  # BGR -> RGB

                    for i in range(box_info.shape[0]):
                        # 1、2d box
                        box2d_four_points = np.array([box2d[i, 0], box2d[i, 1], box2d[i, 0] + box2d[i, 2], box2d[i, 1], box2d[i, 0], box2d[i, 1] + box2d[i, 3], box2d[i, 0] + box2d[i, 2], box2d[i, 1] + box2d[i, 3]])
                        raw_box2d_four_points = box2d_four_points.reshape(4, 2)
                        homo_box2d_four_points = coord_to_homog(raw_box2d_four_points)
                        pers_res_box = np.matmul(pers_trans_m, homo_box2d_four_points.T)
                        pers_res_box = pers_res_box.T
                        pers_res_box = coord_to_normal(pers_res_box)
                        pers_res_box = pers_res_box.reshape(1, -1)
                        box_xs = pers_res_box[:, 0:8:2]
                        box_ys = pers_res_box[:, 1:8:2]
                        rot_new_box_xmin, rot_new_box_xmax = np.min(box_xs), np.max(box_xs)
                        rot_new_box_ymin, rot_new_box_ymax = np.min(box_ys), np.max(box_ys)
                        pers_boxes.append([rot_new_box_xmin, rot_new_box_ymin, abs(rot_new_box_xmax-rot_new_box_xmin), abs(rot_new_box_ymax-rot_new_box_ymin)])
                        # 2、center
                        raw_ct_points = box_center[i].reshape(1, 2)
                        homo_ct_points = coord_to_homog(raw_ct_points)
                        pers_res_ct = np.matmul(pers_trans_m, homo_ct_points.T)
                        pers_res_ct = pers_res_ct.T
                        pers_res_ct = coord_to_normal(pers_res_ct)
                        pers_res_ct = pers_res_ct.reshape(1, -1)
                        pers_center_points.append(pers_res_ct)
                        # 3、vertex
                        raw_vertex = box_vertex[i].reshape(8, 2)
                        homo_raw_vertex = coord_to_homog(raw_vertex)
                        pers_res_vertex = np.matmul(pers_trans_m, homo_raw_vertex.T)
                        pers_res_vertex = pers_res_vertex.T
                        pers_res_vertex = coord_to_normal(pers_res_vertex)
                        pers_res_vertex = pers_res_vertex.reshape(1, -1)
                        pers_vertex.append(pers_res_vertex)

                    box2d = np.array(pers_boxes).astype(np.float32)
                    box_center = np.array(pers_center_points).astype(np.float32).squeeze(axis=1)
                    box_vertex = np.array(pers_vertex).astype(np.float32).squeeze(axis=1)
                    correct_box2d[:len(box_info)] = box2d
                    correct_box_center[:len(box_info)] = box_center
                    correct_box_vertex[:len(box_info)] = box_vertex
            if rotation:
                rot_boxes, rot_center_points, rot_vertex = [], [], []
                if len(box_info) > 0:
                    rot_angle = np.random.uniform(-5, 5)
                    rot_theta = np.deg2rad(rot_angle)
                    rot_center = np.array([featmap_h/2, featmap_w/2]).reshape(2, 1)
                    rot_matrix = np.array([np.cos(rot_theta), np.sin(rot_theta), -np.sin(rot_theta), np.cos(rot_theta)]).reshape(2,2)
                    for i in range(box_info.shape[0]):
                        # 1、2d box
                        box2d_four_points = np.array([box2d[i, 0], box2d[i, 1], box2d[i, 0] + box2d[i, 2], box2d[i, 1], box2d[i, 0], box2d[i, 1] + box2d[i, 3], box2d[i, 0] + box2d[i, 2], box2d[i, 1] + box2d[i, 3]])
                        rot_res_box = np.matmul(rot_matrix, box2d_four_points.reshape(4, 2).T - rot_center) + rot_center
                        rot_res_box = rot_res_box.T.reshape(1, -1)
                        box_xs = rot_res_box[:, 0:8:2]
                        box_ys = rot_res_box[:, 1:8:2]
                        rot_new_box_xmin, rot_new_box_xmax = np.min(box_xs), np.max(box_xs)
                        rot_new_box_ymin, rot_new_box_ymax = np.min(box_ys), np.max(box_ys)
                        rot_boxes.append([rot_new_box_xmin, rot_new_box_ymin, abs(rot_new_box_xmax-rot_new_box_xmin), abs(rot_new_box_ymax-rot_new_box_ymin)])
                        # 2、center
                        rot_res_ct = np.matmul(rot_matrix, box_center[i].reshape(1, 2).T - rot_center) + rot_center
                        rot_res_ct = rot_res_ct.T.reshape(1, -1)
                        rot_center_points.append(rot_res_ct)
                        # 3、vertex
                        rot_res_vertex = np.matmul(rot_matrix, box_vertex[i].reshape(8, 2).T - rot_center) + rot_center
                        rot_res_vertex = rot_res_vertex.T.reshape(1, -1)
                        rot_vertex.append(rot_res_vertex)

                    box2d = np.array(rot_boxes).astype(np.float32)
                    box_center = np.array(rot_center_points).astype(np.float32).squeeze(axis=1)
                    box_vertex = np.array(rot_vertex).astype(np.float32).squeeze(axis=1)
                    correct_box2d[:len(box_info)] = box2d
                    correct_box_center[:len(box_info)] = box_center
                    correct_box_vertex[:len(box_info)] = box_vertex

                    image = image.rotate(rot_angle)
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if len(box_info) > 0:
                    box2d[:, 0] = featmap_w - (box2d[:, 0] + box2d[:, 2])
                    correct_box2d[:len(box_info)] = box2d

                    box_center[:, 0] = featmap_w - box_center[:, 0]
                    correct_box_center[:len(box_info)] = box_center

                    box_vertex[:, 0:18:2] = featmap_w - box_vertex[:, 0:16:2]
                    correct_box_vertex[:len(box_info)] = box_vertex
            if color_change:
                hue = self.rand(-hue, hue)
                sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
                val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
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
            return image, calib_matrix, correct_box2d, box_cls, correct_box_center, correct_box_vertex, box_size, box_perspective, box_base_point, image_w, image_h

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines

        # load data
        image_data, calib_matrix, correct_box2d, box_cls, correct_box_center, correct_box_vertex, box_size, box_perspective, box_base_point, raw_img_w, raw_img_h = self.get_random_data(lines[index], [self.input_size[0],self.input_size[1]], augment=self.is_train)
        
        batch_hm = np.zeros((self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
        batch_center_reg = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_vertex_reg = np.zeros((self.output_size[0], self.output_size[1], 16), dtype=np.float32)
        batch_size_reg = np.zeros((self.output_size[0], self.output_size[1], 3), dtype=np.float32)
        batch_box_perspective = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
        batch_base_point = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_center_mask = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
        
        if len(box_cls) != 0:  # if any object
            # change to size relative to feature h,w !!!
            # for heatmap drawing
            fp_boxes = np.array(correct_box2d, dtype=np.float32)
            fp_boxes[:,0] = fp_boxes[:,0] / self.input_size[1] * self.output_size[1]
            fp_boxes[:,1] = fp_boxes[:,1] / self.input_size[0] * self.output_size[0]
            fp_boxes[:,2] = fp_boxes[:,2] / self.input_size[1] * self.output_size[1]
            fp_boxes[:,3] = fp_boxes[:,3] / self.input_size[0] * self.output_size[0]

            # centroid
            fp_box_center = np.array(correct_box_center, dtype=np.float32)
            fp_box_center[:,0] = fp_box_center[:,0] / self.input_size[1] * self.output_size[1]
            fp_box_center[:,1] = fp_box_center[:,1] / self.input_size[0] * self.output_size[0]

            # vertex
            fp_box_vertex = np.array(correct_box_vertex, dtype=np.float32)
            fp_box_vertex[:,0:16:2] = fp_box_vertex[:,0:16:2] / self.input_size[1] * self.output_size[1]
            fp_box_vertex[:,1:16:2] = fp_box_vertex[:,1:16:2] / self.input_size[0] * self.output_size[0]

        for i in range(len(box_cls)):
            fp_box_center_copy = fp_box_center[i].copy()
            fp_box_center_copy = np.array(fp_box_center_copy)
            # prevent centroid outer feature map
            fp_box_center_copy[0] = np.clip(fp_box_center_copy[0], 0, self.output_size[1] - 1)
            fp_box_center_copy[1] = np.clip(fp_box_center_copy[1], 0, self.output_size[0] - 1)

            box_cls_id = int(box_cls[i])

            # calculate minimum enclosing rectangle of each object for heatmap radius !!!
            fp_box_w, fp_box_h = abs(fp_box_vertex[i, 2] - fp_box_vertex[i, 6]), abs(fp_box_vertex[i, 1] - fp_box_vertex[i, 13])
            if fp_box_w > 0 and fp_box_h > 0:
                radius = gaussian_radius((math.ceil(fp_box_h), math.ceil(fp_box_w)))
                radius = max(0, int(radius))
                # gt centroid
                ct = np.array([fp_box_center_copy[0], fp_box_center_copy[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # draw gaussian heatmap
                batch_hm[:, :, box_cls_id] = draw_gaussian(batch_hm[:, :, box_cls_id], ct_int, radius)
                batch_center_reg[ct_int[1], ct_int[0]] = ct - ct_int
                batch_vertex_reg[ct_int[1], ct_int[0]] = fp_box_vertex[i]
                batch_size_reg[ct_int[1], ct_int[0]] = box_size[i]
                batch_box_perspective[ct_int[1], ct_int[0]] = box_perspective[i]
                batch_base_point[ct_int[1], ct_int[0]] = box_base_point[i]
                batch_center_mask[ct_int[1], ct_int[0]] = 1

        img = np.array(image_data, dtype=np.float32)[:,:,::-1]  # RGB -> BGR
        img = np.transpose(self.preprocess_image(img), (2, 0, 1))  # normalization，HWC->CHW

        return img, calib_matrix, batch_hm, batch_center_reg, batch_vertex_reg, batch_size_reg, batch_center_mask, batch_box_perspective, batch_base_point, raw_img_w, raw_img_h


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
