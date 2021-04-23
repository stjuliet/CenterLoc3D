# predict and decode results
import os
import cv2 as cv
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn
from torch.autograd import Variable

from fpn import KeyPointDetection
from hourglass import HourglassNet, HgResBlock, Hourglass
from utils import decode_bbox, letterbox_image, nms


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


# 使用自己训练好的模型预测
# model_path、classes_path和backbone
class Bbox3dPred(object):
    _defaults = {
        "model_path"        : 'logs\Epoch120-Total_train_Loss2.7936-Val_Loss4.7216.pth',
        "classes_path"      : 'model_data/classes.txt',
        "backbone"          : "resnet50",
        "image_size"        : [512,512,3],
        "confidence"        : 0.15,
        # backbone为resnet50时建议设置为True
        # backbone为hourglass时建议设置为False
        # 也可以根据检测效果自行选择
        "nms"               : True,
        "nms_threhold"      : 0.5,
        "cuda"              : True,
        "letterbox_image"   : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    

    # 初始化Bbox3d
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    # 获得所有的分类
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # 载入模型
    def generate(self):
        self.backbone_resnet_index = {"resnet18": 0, "resnet34": 1, "resnet50": 2, "resnet101": 3, "resnet152": 4}
        # 计算类别数
        self.num_classes = len(self.class_names)
        # 创建模型
        if self.backbone[:-2] == "resnet":
            self.model = KeyPointDetection(model_index=self.backbone_resnet_index[self.backbone], num_classes=self.num_classes, pretrained_weights=False)
        if self.backbone == "hourglass":
            self.model = HourglassNet(2, 1, 256, 3, HgResBlock, inplanes=3)

        # 载入权值
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=True)
        # 验证模式
        self.model = self.model.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.model.cuda()

        print('{} model, classes loaded.'.format(self.model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # 检测图片
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        # 给图像增加灰条，实现不失真的resize
        # 也可以直接resize进行识别
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.image_size[1], self.image_size[0])))
            crop_img = np.array(crop_img, dtype = np.float32)[:,:,::-1]
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)

        photo = np.array(crop_img, dtype = np.float32)
        letter_img = np.array(crop_img, dtype = np.float32)
        
        # 图片预处理，归一化。获得的photo的shape为[1, 3, 512, 512]
        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)), [1, self.image_size[2], self.image_size[0], self.image_size[1]])
        
        with torch.no_grad():
            images = Variable(torch.from_numpy(np.asarray(photo)).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            # [bt, num_classes, 128, 128]
            # [bt, 2, 128, 128]
            # [bt, 16, 128, 128]
            # [bt, 3, 128, 128]
            output_hm, output_center, output_vertex, output_size = self.model(images)

            # 保存热力图
            # 提取属于第0类的热力图
            # hotmaps = output_hm[0].cpu().numpy().transpose(1, 2, 0)[..., 0]
            # print(hotmaps.shape)

            # import matplotlib.pyplot as plt

            # heatmap = np.maximum(hotmaps, 0)
            # heatmap /= np.max(heatmap)
            # plt.matshow(heatmap)
            # plt.show()

            # heatmap = cv.resize(heatmap, (self.image_size[0], self.image_size[1]))
            # heatmap = np.uint8(255 * heatmap)
            # heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
            # superimposed_img = heatmap * 0.4 + letter_img
            # cv.imwrite('img/hotmap.jpg', superimposed_img)

            # 利用预测结果进行解码
            outputs = decode_bbox(output_hm, output_center, output_vertex, output_size, self.image_size, self.confidence, self.cuda, 50)

            #-------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            #-------------------------------------------------------#
            empty_box = []
            try:
                if self.nms:
                    for i in range(len(outputs)):
                        empty_box.append([outputs[i][16], outputs[i][17], outputs[i][4], outputs[i][5], outputs[i][22]])
                    np_det_results = np.array(empty_box, dtype=np.float32)
                    outputs = np.array(nms(np_det_results, self.nms_threhold))
                    # pass
                # 后续添加3d box iou 计算公式
                # outputs = np.array(nms(outputs, self.nms_threhold))
            except:
                pass
            
            output = outputs[0]
            if len(output) <= 0:
                return image
            
            norm_center, norm_vertex, box_size, det_conf, det_cls = output[:,:2], output[:,2:18], output[:,18:21], output[:,21], output[:,22]
            
            # 筛选出其中得分高于confidence的框
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_cls[top_indices].tolist()
            top_norm_center = norm_center[top_indices]
            top_norm_vertex = norm_vertex[top_indices]
            top_box_size = box_size[top_indices]
            
            # 将坐标还原至原图像
            top_norm_center[:, 0] = top_norm_center[:, 0] * max(image_shape[0], image_shape[1])
            top_norm_center[:, 1] = top_norm_center[:, 1] * max(image_shape[0], image_shape[1]) - abs(image_shape[0]-image_shape[1])//2.

            top_norm_vertex[:, 0:16:2] = top_norm_vertex[:, 0:16:2] * max(image_shape[0], image_shape[1])
            top_norm_vertex[:, 1:16:2] = top_norm_vertex[:, 1:16:2] * max(image_shape[0], image_shape[1]) - abs(image_shape[0]-image_shape[1])//2.


        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32')//2)

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.image_size[0], 1)


        keep = []
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


                    # 类别, 置信度
                    label = '{} {:.2f}'.format(predicted_class, score)
                    size = 'l:{:.2f}, w:{:.2f}, h:{:.2f}'.format(l, w, h)
                    size = size.encode('utf-8')
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
                    label = label.encode('utf-8')
                    # print(label)
                    draw.text((cx_d, cy_d), str(label,'UTF-8'), fill=(0, 0, 0), font=font)

                    # 长宽高
                    draw.text((cx_d-60, cy_d-60), str(size,'UTF-8'), fill=(0, 0, 0), font=font)

                    draw.ellipse((cx_d -3 , cy_d - 3, cx_d + 3, cy_d + 3), outline=(0,0,255), width=2)

                    # 3D box
                    # 宽度方向
                    # 0-1  2-3  4-5  6-7
                    # if (vertex[14] <= cx < = vertex[2]) and (vertex[13] <= cy <= vertex[1]):
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
                    del draw
        return image

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

        #     # 类别, 置信度
        #     label = '{} {:.2f}'.format(predicted_class, score)
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label, font)
        #     label = label.encode('utf-8')
        #     # print(label)
        #     draw.text((cx, cy), str(label,'UTF-8'), fill=(0, 0, 0), font=font)

        #     # 3D box
        #     # 宽度方向
        #     # 0-1  2-3  4-5  6-7
        #     # if (vertex[14] <= cx < = vertex[2]) and (vertex[13] <= cy <= vertex[1]):
        #     draw.line([vertex[0], vertex[1], vertex[2], vertex[3]], fill=128, width=2)
        #     draw.line([vertex[4], vertex[5], vertex[6], vertex[7]], fill=128, width=2)
        #     draw.line([vertex[8], vertex[9], vertex[10], vertex[11]], fill=128, width=2)
        #     draw.line([vertex[12], vertex[13], vertex[14], vertex[15]], fill=128, width=2)

        #     # 长度方向
        #     # 0-3 1-2 4-7 5-6
        #     draw.line([vertex[0], vertex[1], vertex[6], vertex[7]], fill=128, width=2)
        #     draw.line([vertex[2], vertex[3], vertex[4], vertex[5]], fill=128, width=2)
        #     draw.line([vertex[8], vertex[9], vertex[14], vertex[15]], fill=128, width=2)
        #     draw.line([vertex[10], vertex[11], vertex[12], vertex[13]], fill=128, width=2)

        #     # 高度方向
        #     # 0-4 1-5 2-6 3-7
        #     draw.line([vertex[0], vertex[1], vertex[8], vertex[9]], fill=128, width=2)
        #     draw.line([vertex[2], vertex[3], vertex[10], vertex[11]], fill=128, width=2)
        #     draw.line([vertex[4], vertex[5], vertex[12], vertex[13]], fill=128, width=2)
        #     draw.line([vertex[6], vertex[7], vertex[14], vertex[15]], fill=128, width=2)
        #     del draw
        # return image
