# predict and decode results
import os
import time
import cv2 as cv
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn
from torch.autograd import Variable

from nets.fpn import KeyPointDetection
from nets.hourglass_official import HourglassNet, Bottleneck
from utils.utils import *


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


# use model to predict
# model_path、classes_path和backbone
class Bbox3dPred(object):
    _defaults = {
        "model_path"        : '../logs/resnet50-Epoch99-ciou-Total_train_Loss1.5854-Val_Loss2.2824.pth',
        "classes_path"      : '../model_data/classes.txt',
        "backbone"          : "resnet50",
        "image_size"        : [512,512,3],
        "confidence"        : 0.3,
        # backbone: resnet50 - True
        # backbone: hourglass - False
        "nms"               : True,
        "nms_threhold"      : 0.5,
        "cuda"              : True,
        "letterbox_image"   : True   # suggested True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):
        self.__dict__.update(self._defaults)  # dict key -> class attr, use self.** to use
        self.class_names = self._get_class()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # load model
    def generate(self):
        self.backbone_resnet_index = {"resnet18": 0, "resnet34": 1, "resnet50": 2, "resnet101": 3, "resnet152": 4}
        self.backbone_efficientnet_index = {"efficientnetb0": 0, "efficientnetb1": 1, "efficientnetb2": 2,
                     "efficientnetb3": 3, "efficientnetb4": 4, "efficientnetb5": 5, "efficientnetb6": 6, "efficientnetb7": 7}

        self.num_classes = len(self.class_names)

        # build model
        if self.backbone[:-2] == "resnet":
            self.model = KeyPointDetection(model_name=self.backbone[:-2], model_index=self.backbone_resnet_index[self.backbone], num_classes=self.num_classes)
        if self.backbone[:-2] == "efficientnet":
            self.model = KeyPointDetection(model_name=self.backbone[:-2], model_index=self.backbone_efficientnet_index[self.backbone], num_classes=self.num_classes)
        if self.backbone == "hourglass":
            self.model = HourglassNet(Bottleneck, num_stacks=8, num_blocks=1, num_classes=self.num_classes)

        # load weights
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict["state_dict"], strict=True)
        # eval
        self.model = self.model.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.model.cuda()

        print('{} model, classes loaded.'.format(self.model_path))

        # cls colors
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image, image_id, is_draw_gt, is_record_result, calib_path=None, mode="test"):
        """
        image: PIL img
        """
        image_shape = np.array(np.shape(image)[0:2])

        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.image_size[1], self.image_size[0])))
            crop_img = np.array(crop_img, dtype=np.float32)[:,:,::-1]  # rgb -> bgr
        else:
            crop_img = np.array(image, dtype=np.float32)[:,:,::-1]  # rgb -> bgr
            crop_img = cv.cvtColor(crop_img, cv.COLOR_RGB2BGR)
            crop_img = cv.resize(crop_img, (self.image_size[1], self.image_size[0]), cv.INTER_CUBIC)

        photo = np.array(crop_img, dtype=np.float32)
        letter_img = np.array(crop_img, dtype=np.float32)
        
        # preprocess, normalization, [1, 3, 512, 512]
        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)), [1, self.image_size[2], self.image_size[0], self.image_size[1]])
        
        with torch.no_grad():
            images = Variable(torch.from_numpy(np.asarray(photo)).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            # [bt, num_classes, 128, 128]
            # [bt, 2, 128, 128]
            # [bt, 16, 128, 128]
            # [bt, 3, 128, 128]

            t1 = time.time()
            output_hm, output_center, output_vertex, output_size = self.model(images)

            # ----------------------------------save specific class heatmap (fusion)-----------------------------------#
            # # save heatmap
            # # heatmap of cls-0
            # hotmaps = output_hm[0].cpu().numpy().transpose(1, 2, 0)[..., 0]
            # print(hotmaps.shape)

            # import matplotlib.pyplot as plt

            # heatmap = np.maximum(hotmaps, 0)
            # heatmap /= np.max(heatmap)
            # # plt.matshow(heatmap)
            # # plt.show()

            # heatmap = cv.resize(heatmap, (self.image_size[0], self.image_size[1]))
            # heatmap = np.uint8(255 * heatmap)
            # heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
            # superimposed_img = heatmap * 0.4 + letter_img * 0.8

            # cv.imwrite('img/raw_heat.png', heatmap)
            # cv.waitKey()
            # ----------------------------------save specific class heatmap (fusion)-----------------------------------#

            # ----------------------------------save all class heatmap (fusion)-----------------------------------#
            # final_heatmap = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            # for i in range(self.num_classes):
            #     hotmap = output_hm[0].cpu().numpy().transpose(1, 2, 0)[..., i]  # each class

            #     import matplotlib.pyplot as plt

            #     heatmap = np.maximum(hotmap, 0)
            #     heatmap /= np.max(heatmap)

            #     heatmap = cv.resize(heatmap, (self.image_size[0], self.image_size[1]))
            #     heatmap = np.uint8(255 * heatmap)
            #     heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)

            #     heatmap = heatmap / self.num_classes
            #     heatmap = heatmap.astype(np.uint8)

            #     final_heatmap += heatmap
            # superimposed_img = final_heatmap * 0.4 + letter_img * 0.8
            # r_heatmap = Image.fromarray(superimposed_img)
            # cv.imwrite('img/hotmap.jpg', superimposed_img)
            # cv.waitKey()
            # ----------------------------------save all class heatmap (fusion)-----------------------------------#

            outputs = decode_bbox(output_hm, output_center, output_vertex, output_size, self.confidence, self.cuda, 50)

            # nms
            empty_box = []
            try:
                if self.nms:
                    for i in range(len(outputs)):
                        x1, y1, x7, y7, cls_id = outputs[i][4], outputs[i][5], outputs[i][16], outputs[i][17], outputs[i][22]
                        # vehicle view
                        if x7 < x1:   # 1、right  x7<x1
                            empty_box.append([x7, y7, x1, y1, cls_id])
                        else:  # 2、left   x7>=x1
                            empty_box.append([x1, y7, x7, y1, cls_id])
                    np_det_results = np.array(empty_box, dtype=np.float32)
                    outputs = np.array(nms(np_det_results, self.nms_threhold))
            except:
                pass
            
            output = outputs[0]
            if len(output) <= 0:
                return image
            
            # normalization coordinate [0, 1]
            norm_center, norm_vertex, box_size, det_conf, det_cls = output[:,:2], output[:,2:18], output[:,18:21], output[:,21], output[:,22]
            
            # filter box conf > threshold
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_cls[top_indices].tolist()
            top_norm_center = norm_center[top_indices]
            top_norm_vertex = norm_vertex[top_indices]
            top_box_size = box_size[top_indices]
            
            # coordinate, raw img
            if self.letterbox_image:
                top_norm_center = correct_vertex_norm2raw(top_norm_center, image_shape)
                top_norm_vertex = correct_vertex_norm2raw(top_norm_vertex, image_shape)
            else:
                top_norm_center[:, 0] = top_norm_center[:, 0] * image_shape[1]
                top_norm_center[:, 1] = top_norm_center[:, 1] * image_shape[0]
                top_norm_vertex[:, 0:16:2] = top_norm_vertex[:, 0:16:2] * image_shape[1]
                top_norm_vertex[:, 1:16:2] = top_norm_vertex[:, 1:16:2] * image_shape[0]

            t2 = time.time()

            process_time = t2 - t1

            # ----------------------------------save specific class heatmap-----------------------------------#
            # # save heatmap
            # # heatmap of cls-0
            hotmaps = output_hm[0].cpu().numpy().transpose(1, 2, 0)[..., 0]
            # print(hotmaps.shape)

            import matplotlib.pyplot as plt

            heatmap = np.maximum(hotmaps, 0)
            heatmap /= np.max(heatmap)
            # plt.matshow(heatmap)
            # plt.show()

            heatmap = cv.resize(heatmap, (self.image_size[0], self.image_size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)

        # font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32')//2)
        font = ImageFont.truetype(font="../model_data/Times New Roman.ttf", size=28)

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.image_size[0], 1)

        if is_record_result:
            # 2D
            if not os.path.exists("../%s/input-2D"%mode):
                os.makedirs("../%s/input-2D"%mode)
            if not os.path.exists("../%s/input-2D/detection-results"%mode):
                os.makedirs("../%s/input-2D/detection-results"%mode)
            if not os.path.exists("../%s/input-2D/images-optional"%mode):
                os.makedirs("../%s/input-2D/images-optional"%mode)
            if not os.path.exists("../%s/input-2D/images-heatmap"%mode):
                os.makedirs("../%s/input-2D/images-heatmap"%mode)
            # 3D
            if not os.path.exists("../%s/input-3D"%mode):
                os.makedirs("../%s/input-3D"%mode)
            if not os.path.exists("../%s/input-3D/detection-results"%mode):
                os.makedirs("../%s/input-3D/detection-results"%mode)
            if not os.path.exists("../%s/input-3D/images-optional"%mode):
                os.makedirs("../%s/input-3D/images-optional"%mode)

            # open txt files
            f_2d = open("../%s/input-2D/detection-results/"%mode+image_id+".txt","w")
            f_3d = open("../%s/input-3D/detection-results/"%mode+image_id+".txt","w")

            calib_matrix = read_calib_params(calib_path, image_shape[1], image_shape[0])

        if is_draw_gt:
            draw_gt = ImageDraw.Draw(image)
            if os.path.exists("../%s/gt-for-draw/"%mode+image_id+"_vt2d.txt"):
                with open("../%s/gt-for-draw/"%mode+image_id+"_vt2d.txt") as f_draw_gt:
                    f_draw_lines = f_draw_gt.readlines()
                    for draw_line in f_draw_lines:
                        gt_vertex_2d = list(map(int, draw_line.split(" ")[:16]))
                        gt_cx, gt_cy = list(map(int, draw_line.split(" ")[16:18]))

                        draw_gt.ellipse((gt_cx - 3, gt_cy - 3, gt_cx + 3, gt_cy + 3), outline=(255, 0, 255), width=2)

                        # 3D box
                        # width
                        # 0-1  2-3  4-5  6-7
                        # if (vertex[14] <= cx < = vertex[2]) and (vertex[13] <= cy <= vertex[1]):
                        draw_gt.line([gt_vertex_2d[0], gt_vertex_2d[1], gt_vertex_2d[2], gt_vertex_2d[3]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[4], gt_vertex_2d[5], gt_vertex_2d[6], gt_vertex_2d[7]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[8], gt_vertex_2d[9], gt_vertex_2d[10], gt_vertex_2d[11]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[12], gt_vertex_2d[13], gt_vertex_2d[14], gt_vertex_2d[15]], fill=(255, 0, 255), width=2)

                        # length
                        # 0-3 1-2 4-7 5-6
                        draw_gt.line([gt_vertex_2d[0], gt_vertex_2d[1], gt_vertex_2d[6], gt_vertex_2d[7]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[2], gt_vertex_2d[3], gt_vertex_2d[4], gt_vertex_2d[5]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[8], gt_vertex_2d[9], gt_vertex_2d[14], gt_vertex_2d[15]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[10], gt_vertex_2d[11], gt_vertex_2d[12], gt_vertex_2d[13]], fill=(255, 0, 255), width=2)

                        # height
                        # 0-4 1-5 2-6 3-7
                        draw_gt.line([gt_vertex_2d[0], gt_vertex_2d[1], gt_vertex_2d[8], gt_vertex_2d[9]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[2], gt_vertex_2d[3], gt_vertex_2d[10], gt_vertex_2d[11]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[4], gt_vertex_2d[5], gt_vertex_2d[12], gt_vertex_2d[13]], fill=(255, 0, 255), width=2)
                        draw_gt.line([gt_vertex_2d[6], gt_vertex_2d[7], gt_vertex_2d[14], gt_vertex_2d[15]], fill=(255, 0, 255), width=2)
            del draw_gt

        for i in range(len(top_label_indices)):
            predicted_class = self.class_names[int(top_label_indices[i])]
            score = top_conf[i]
            cx, cy = top_norm_center[i].astype(np.int32)
            vertex = top_norm_vertex[i].astype(np.int32)
            l, w, h = top_box_size[i]

            if cx > np.shape(image)[1] or cx < 0 or cy > np.shape(image)[0] or cy < 0:
                continue

            for j in range(len(top_label_indices)):
                predicted_class_d = self.class_names[int(top_label_indices[j])]
                score_d = top_conf[j]
                cx_d, cy_d = top_norm_center[j].astype(np.int32)
                vertex_d = top_norm_vertex[j].astype(np.int32)
                l_d, w_d, h_d = top_box_size[j]

                if (vertex[14]+vertex_d[2])/2 - 10 <= cx_d <= (vertex[14]+vertex_d[2])/2 + 10 and (vertex[13]+vertex[1])/2 -10 <= cy_d <= (vertex[13]+vertex[1])/2 +10:

                    # cls, conf
                    label = '{} {:.2f}'.format(predicted_class, score)
                    size = 'l:{:.2f}, w:{:.2f}, h:{:.2f}'.format(l, w, h)
                    size = size.encode('utf-8')
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
                    label = label.encode('utf-8')
                    # print(label)
                    draw.text((cx_d, cy_d), str(label,'UTF-8'), fill=(0, 0, 0), font=font)

                    # size
                    # draw.text((cx_d-60, cy_d-60), str(size,'UTF-8'), fill=(0, 0, 0), font=font)

                    draw.ellipse((cx_d -3 , cy_d - 3, cx_d + 3, cy_d + 3), outline=(0,0,255), width=2)

                    # 3D box
                    # width
                    # 0-1  2-3  4-5  6-7
                    # if (vertex[14] <= cx < = vertex[2]) and (vertex[13] <= cy <= vertex[1]):
                    draw.line([vertex[0], vertex[1], vertex[2], vertex[3]], fill=(255, 0, 0), width=2)
                    draw.line([vertex[4], vertex[5], vertex[6], vertex[7]], fill=(255, 0, 0), width=2)
                    draw.line([vertex[8], vertex[9], vertex[10], vertex[11]], fill=(255, 0, 0), width=2)
                    draw.line([vertex[12], vertex[13], vertex[14], vertex[15]], fill=(255, 0, 0), width=2)

                    # length
                    # 0-3 1-2 4-7 5-6
                    draw.line([vertex[0], vertex[1], vertex[6], vertex[7]], fill=(0, 0, 255), width=2)
                    draw.line([vertex[2], vertex[3], vertex[4], vertex[5]], fill=(0, 0, 255), width=2)
                    draw.line([vertex[8], vertex[9], vertex[14], vertex[15]], fill=(0, 0, 255), width=2)
                    draw.line([vertex[10], vertex[11], vertex[12], vertex[13]], fill=(0, 0, 255), width=2)

                    # height
                    # 0-4 1-5 2-6 3-7
                    draw.line([vertex[0], vertex[1], vertex[8], vertex[9]], fill=(0, 255, 0), width=2)
                    draw.line([vertex[2], vertex[3], vertex[10], vertex[11]], fill=(0, 255, 0), width=2)
                    draw.line([vertex[4], vertex[5], vertex[12], vertex[13]], fill=(0, 255, 0), width=2)
                    draw.line([vertex[6], vertex[7], vertex[14], vertex[15]], fill=(0, 255, 0), width=2)

                    # draw vehicle size values
                    draw.text([(vertex[0] + vertex[6]) // 2-25, (vertex[1] + vertex[7]) // 2-25], "{:.2f}m".format(l),
                              fill=(255, 0, 0), font=font)
                    draw.text([(vertex[0] + vertex[2]) // 2-25, (vertex[1] + vertex[3]) // 2], "{:.2f}m".format(w),
                              fill=(255, 0, 0), font=font)
                    draw.text([(vertex[2] + vertex[10]) // 2, (vertex[3] + vertex[11]) // 2-20], "{:.2f}m".format(h),
                              fill=(255, 0, 0), font=font)

                    # save record
                    if is_record_result:
                        if vertex[14] < vertex[2]:  # right perspective(7,1)
                            left, top, right, bottom = vertex[14], vertex[15], vertex[2], vertex[3]
                            f_2d.write("%s %s %s %s %s %s\n" % (predicted_class, str(score)[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
                        else:  # left perspective  (1x,6y,7x,0y)
                            left, top, right, bottom = vertex[2], vertex[13], vertex[14], vertex[1]
                            f_2d.write("%s %s %s %s %s %s\n" % (predicted_class, str(score)[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

                        vertex_3d = cal_pred_3dvertex(vertex, h, calib_matrix)
                        line_3d, line_size, line_ct_loc = "", "", ""
                        for i in vertex_3d:
                            line_3d += " " + str(i)
                        for i in (l, w, h):
                            line_size += " " + str(i)
                        # calc 3d centroid
                        cx_3d, cy_3d, cz_3d = RDUVtoXYZ(calib_matrix, cx_d, cy_d, 1000*h/2)
                        for i in (cx_3d, cy_3d, cz_3d):
                            line_ct_loc += " " + str(i)

                        f_3d.write("%s %s %s %s %s\n" % (predicted_class, str(score)[:6], str(line_3d.strip()), str(line_size.strip()), str(line_ct_loc.strip())))

                    del draw
        return image, heatmap, process_time

        # for i, c in enumerate(top_label_indices):
        #     predicted_class = self.class_names[int(c)]
        #     score = top_conf[i]

        #     cx, cy = top_norm_center[i].astype(np.int32)
        #     vertex = top_norm_vertex[i].astype(np.int32)
        #     l, w, h = top_box_size[i]

        #     if cx > np.shape(image)[1] or cx < 0 or cy > np.shape(image)[0] or cy < 0:
        #         continue

        #     for j, d in enumerate(top_label_indices):
        #         predicted_class = self.class_names[int(c)]
        #         score = top_conf[i]

        #         cx, cy = top_norm_center[i].astype(np.int32)
        #         vertex = top_norm_vertex[i].astype(np.int32)
        #         l, w, h = top_box_size[i]

        #                     # if (vertex[14] <= cx <= vertex[2]) and (vertex[13] <= cy <= vertex[1]):
        #     #     continue

        #     # cls, conf
        #     label = '{} {:.2f}'.format(predicted_class, score)
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label, font)
        #     label = label.encode('utf-8')
        #     # print(label)
        #     draw.text((cx, cy), str(label,'UTF-8'), fill=(0, 0, 0), font=font)

        #     # 3D box
        #     # width
        #     # 0-1  2-3  4-5  6-7
        #     # if (vertex[14] <= cx < = vertex[2]) and (vertex[13] <= cy <= vertex[1]):
        #     draw.line([vertex[0], vertex[1], vertex[2], vertex[3]], fill=128, width=2)
        #     draw.line([vertex[4], vertex[5], vertex[6], vertex[7]], fill=128, width=2)
        #     draw.line([vertex[8], vertex[9], vertex[10], vertex[11]], fill=128, width=2)
        #     draw.line([vertex[12], vertex[13], vertex[14], vertex[15]], fill=128, width=2)

        #     # length
        #     # 0-3 1-2 4-7 5-6
        #     draw.line([vertex[0], vertex[1], vertex[6], vertex[7]], fill=128, width=2)
        #     draw.line([vertex[2], vertex[3], vertex[4], vertex[5]], fill=128, width=2)
        #     draw.line([vertex[8], vertex[9], vertex[14], vertex[15]], fill=128, width=2)
        #     draw.line([vertex[10], vertex[11], vertex[12], vertex[13]], fill=128, width=2)

        #     # height
        #     # 0-4 1-5 2-6 3-7
        #     draw.line([vertex[0], vertex[1], vertex[8], vertex[9]], fill=128, width=2)
        #     draw.line([vertex[2], vertex[3], vertex[10], vertex[11]], fill=128, width=2)
        #     draw.line([vertex[4], vertex[5], vertex[12], vertex[13]], fill=128, width=2)
        #     draw.line([vertex[6], vertex[7], vertex[14], vertex[15]], fill=128, width=2)
        #     del draw
        # return image
