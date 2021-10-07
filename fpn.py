# resnets + fpn
import math
import torch
import torch.nn as nn
from torchsummary import summary
from nets.resnets import resnet18, resnet34, resnet50, resnet101, resnet152
from nets.efficientnet import EfficientNet as EffNet
from nets.darknet import darknet53
from deform_conv import DeformConv2d


class Resnet(nn.Module):
    '''
    input: model index
    output: 3 feature maps (for FPN)
    '''
    def __init__(self, model_index, load_weights=False):
        super(Resnet, self).__init__()
        self.model_edition = [resnet18, resnet34, resnet50, resnet101, resnet152]
        model = self.model_edition[model_index](load_weights)
        # remove avgpool, fc layers for future structure
        del model.avgpool
        del model.fc
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        feat1 = self.model.layer2(x)
        feat2 = self.model.layer3(feat1)
        feat3 = self.model.layer4(feat2)
        # size: 64, 32, 16  channels: 512, 1024, 2048
        return feat1, feat2, feat3


class EfficientNet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            #------------------------------------------------------#
            #   取出对应的特征层，如果某个EffcientBlock的步长为2的话
            #   意味着它的前一个特征层为有效特征层
            #   除此之外，最后一个EffcientBlock的输出为有效特征层
            #------------------------------------------------------#
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[2:]


class FPN(nn.Module):
    '''
    input: 3 feature maps
    output: 5 features maps -> ConvTranspose2d to 1 feature map (for detection head)
    '''
    def __init__(self, C3_channels, C4_channels, C5_channels, out_channels = 256, deform=False):
        super(FPN, self).__init__()
        self.final_out_channels = 64  # 最终特征图尺寸
        self.fpchannels = [64, 32, 16, 8, 4]  # 金字塔网络多尺度特征层尺寸列表

        if deform:
            self.C3_conv1 = DeformConv2d(C3_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False, modulation=False) # 1*1卷积,保证feature map尺寸不变
            self.C3_conv2 = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, modulation=False) # 3*3卷积,保证feature map尺寸不变
            self.C4_conv1 = DeformConv2d(C4_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False, modulation=False) # 1*1卷积,保证feature map尺寸不变
            self.C4_conv2 = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, modulation=False) # 3*3卷积,保证feature map尺寸不变
            self.C5_conv1 = DeformConv2d(C5_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False, modulation=False) # 1*1卷积,保证feature map尺寸不变
            self.C5_conv2 = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, modulation=False) # 3*3卷积,保证feature map尺寸不变
            self.P6_conv = DeformConv2d(C5_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False, modulation=False) # 3*3卷积,stride=2, 减小feature map尺寸
            self.P7_conv = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False, modulation=False) # 3*3卷积,stride=2, 减小feature map尺寸
        else:
            self.C3_conv1 = nn.Conv2d(C3_channels, out_channels, kernel_size=1, stride=1, padding=0) # 1*1卷积,保证feature map尺寸不变
            self.C3_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) # 3*3卷积,保证feature map尺寸不变
            self.C4_conv1 = nn.Conv2d(C4_channels, out_channels, kernel_size=1, stride=1, padding=0) # 1*1卷积,保证feature map尺寸不变
            self.C4_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) # 3*3卷积,保证feature map尺寸不变
            self.C5_conv1 = nn.Conv2d(C5_channels, out_channels, kernel_size=1, stride=1, padding=0) # 1*1卷积,保证feature map尺寸不变
            self.C5_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) # 3*3卷积,保证feature map尺寸不变
            self.P6_conv = nn.Conv2d(C5_channels, out_channels, kernel_size=3, stride=2, padding=1) # 3*3卷积,stride=2, 减小feature map尺寸
            self.P7_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) # 3*3卷积,stride=2, 减小feature map尺寸
        # inplace=True 节约反复申请和释放内存的资源消耗, 但是在训练时反向传播会导致无法求导, pytorch 0.4之后的版本均会有该问题
        self.P7_relu = nn.ReLU()

        self.unsample = nn.Upsample(scale_factor=2)  # 2倍上采样

        # 针对P3--P7特征图 做反卷积
        # forward函数中只能包含前向传播, 不能定义layer
        # 原始尺寸 -> 128*128*64
        self.P3_convtrans = self._make_convtrans_sequence(out_channels, self.final_out_channels, self.fpchannels[0])
        self.P4_convtrans = self._make_convtrans_sequence(out_channels, self.final_out_channels, self.fpchannels[1])
        self.P5_convtrans = self._make_convtrans_sequence(out_channels, self.final_out_channels, self.fpchannels[2])
        self.P6_convtrans = self._make_convtrans_sequence(out_channels, self.final_out_channels, self.fpchannels[3])
        self.P7_convtrans = self._make_convtrans_sequence(out_channels, self.final_out_channels, self.fpchannels[4])

        # self.no_merge_conv = nn.Conv2d(C3_channels//2, out_channels//4, kernel_size=3, stride=1, padding=1)


    def _make_convtrans_sequence(self, in_channels, final_out_channels, in_sizes, out_sizes = 128):
        sequences = []
        length = math.log2(out_sizes//in_sizes)  # 默认float类型
        for i in range(int(length)):
            if length == 1:
                out_channels = final_out_channels
            else:
                in_channels = in_channels//pow(2, i) if in_channels//pow(2, i) > self.final_out_channels else self.final_out_channels
                out_channels = in_channels//pow(2, i+1) if in_channels//pow(2, i+1) > self.final_out_channels else self.final_out_channels
            sequences.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,kernel_size=4,stride=2, padding=1))
            sequences.append(nn.BatchNorm2d(out_channels))
            sequences.append(nn.ReLU())
        return nn.Sequential(*sequences)

    
    def forward(self, input_1, input_2, input_3):
        C3, C4, C5 = input_1, input_2, input_3
        C3_x1 = self.C3_conv1(C3)
        C4_x1 = self.C4_conv1(C4)
        C5_x1 = self.C5_conv1(C5)
        P5 = self.C5_conv2(C5_x1)

        C5_conv_unsample = self.unsample(C5_x1)
        C4_x2 = C4_x1 + C5_conv_unsample
        P4 = self.C4_conv2(C4_x2)

        C4_unsample = self.unsample(C4_x2)
        C3_x2 = C3_x1 + C4_unsample
        P3 = self.C3_conv2(C3_x2)

        P6 = self.P6_conv(C5)

        P7_x = self.P7_relu(P6)
        P7 = self.P7_conv(P7_x)

        # # -------------------------多尺度特征融合-----------------------------#
        # 反卷积提取高分辨率特征层
        P3 = self.P3_convtrans(P3)
        P4 = self.P4_convtrans(P4)
        P5 = self.P5_convtrans(P5)
        P6 = self.P6_convtrans(P6)
        P7 = self.P7_convtrans(P7)

        # [64, 128, 128]
        # 分配权重
        P_merge = 0.5 * P3 + 0.2 * P4 + 0.1 * P5 + 0.1 * P6 + 0.1 * P7
        # P_merge = 0.5 * P3 + 0.3 * P4 + 0.2 * P5

        return P_merge
        # # -------------------------多尺度特征融合-----------------------------#

        # 去多尺度特征融合
        # 将P3上采样+卷积,从64*64*256到128*128*64
        # P3_out = self.P3_convtrans(P3)

        # return P3_out




class DetectionHead(nn.Module):
    '''
    input: 1 features maps
    output: detection results (type, center, vertex, phy_size)
    '''
    def __init__(self, in_channels, out_channels = 64, num_classes = 3, center = 2, num_vertex = 16, phy_size = 3):
        super(DetectionHead, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.pred_dimension = num_classes + center + num_vertex + phy_size
        # type (num_classes)
        self.type_sequence = self._make_sequence(in_channels, out_channels, num_classes)
        # regression (center_offset, vertex, size)
        self.center_sequence = self._make_sequence(in_channels, out_channels, center)
        self.vertex_sequence = self._make_sequence(in_channels, out_channels, num_vertex)
        self.phy_size_sequence = self._make_sequence(in_channels, out_channels, phy_size)
        self.attention_avgpool, self.attention_fc = self._se_attention(out_channels)
    
    def _make_sequence(self, in_channels, out_channels, final_out_channels):
        '''[4 groups of conv and relu] + [1 group of output conv]'''
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(),
                             nn.Conv2d(out_channels, final_out_channels, kernel_size=3, stride=1, padding=1))

    def _se_attention(self, out_channels, reduction=16):
        '''attention机制'''
        return nn.AdaptiveAvgPool2d(1), nn.Sequential(nn.Linear(out_channels, out_channels // reduction, bias=False),
                             nn.ReLU(),
                             nn.Linear(out_channels // reduction, out_channels, bias=False),
                             nn.Sigmoid())

    def forward(self, x):
        # 使用注意力机制的类别预测
        # 提取make_sequence除最后一层的所有层, 后接注意力机制, 
        # 使用make_sequence最后一层卷积, 后接sigmoid函数, 完成预测

        for i in range(len(self.type_sequence)-1):
            temp_out_type = self.type_sequence[i](x)
        temp_out_type = self.attention_avgpool(temp_out_type).view(x.shape[0], x.shape[1])
        temp_out_type = self.attention_fc(temp_out_type).view(x.shape[0], x.shape[1], 1, 1)
        out_type = x * temp_out_type.expand_as(x)
        out_type = self.sigmoid(self.type_sequence[len(self.type_sequence)-1](out_type))

        out_center = self.center_sequence(x)
        out_vertex = self.vertex_sequence(x)
        out_phy_size = self.phy_size_sequence(x)
        # output = torch.cat([out_type, out_center, out_vertex, out_phy_size], dim = 1)  # 在预测值维度上拼接(type + center + vertex + size)

        return out_type, out_center, out_vertex, out_phy_size


class KeyPointDetection(nn.Module):
    '''
    inplementation of the network (4 components)
    '''
    def __init__(self, model_name, model_index, num_classes, pretrained_weights = False, deform=False):
        super(KeyPointDetection, self).__init__()
        self.pretrained_weights = pretrained_weights
        self.deform = deform
        if model_name == "resnet":
            self.backbone = Resnet(model_index, pretrained_weights)
            fpn_size_dict = {
            0: [128, 256, 512],
            1: [128, 256, 512],
            2: [512, 1024, 2048],
            3: [512, 1024, 2048],
            4: [512, 1024, 2048],
            }[model_index]
        if model_name == "efficientnet":
            self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 6]
            fpn_size_dict = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            }[model_index]

            #-------------------------------------------#
            #   获得三个shape的有效特征层
            #   分别是C3  64, 64, 40
            #         C4  32, 32, 112
            #         C5  16, 16, 320
            #-------------------------------------------#
            self.backbone = EfficientNet(self.backbone_phi[model_index], pretrained_weights)
        
        if model_name == "darknet":
            self.backbone = darknet53(pretrained_weights, deform)
            fpn_size_dict = {0: [256, 512, 1024]}[model_index]

        self.fpn = FPN(fpn_size_dict[0], fpn_size_dict[1], fpn_size_dict[2], 256, self.deform)
        self.detection_head = DetectionHead(in_channels=64, num_classes=num_classes)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        resnet_features_1, resnet_features_2, resnet_features_3 = self.backbone(x)
        fpn_features = self.fpn(resnet_features_1, resnet_features_2, resnet_features_3)
        bt_hm, bt_center, bt_vertex, bt_size = self.detection_head(fpn_features)
        return bt_hm, bt_center, bt_vertex, bt_size


if __name__ == "__main__":
    # feature = torch.randn((1, 3, 512, 512))
    # resnet-50
    # model = KeyPointDetection("resnet", 2, num_classes=3)
    # efficientnet-b5
    # model = KeyPointDetection("efficientnet", 5, num_classes=3)


    # darknet-53(input:416*416)
    feature = torch.randn((1, 3, 416, 416))
    model = KeyPointDetection("darknet", 0, num_classes=3, pretrained_weights=True, deform=True)

    bt_hm, bt_center, bt_vertex, bt_size = model(feature)
    print(bt_center.shape)
    print(bt_size.shape)
    # 输出summary的时候，model中返回特征图不能以list形式打包返回
    print(summary(model,(3, 416, 416), batch_size=1, device='cpu'))
