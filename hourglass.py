# hourglass
# - reference: http://t.zoukankan.com/xxxxxxxxx-p-11651437.html

import torch
import torch.nn as nn
from torchsummary import summary

class HgResBlock(nn.Module):
    """ single resnet block """
    def __init__(self, inplanes, outplanes):
        super(HgResBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.conv_1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, stride=1)
        self.bn_2 = nn.BatchNorm2d(midplanes)
        self.conv_2 = nn.Conv2d(midplanes, midplanes, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(midplanes)
        self.conv_3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.conv_skip = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1)
    
    def forward(self, x):
        residual = x
        out = self.bn_1(x)
        out = self.conv_1(out)
        out = self.relu(out)
        out = self.bn_2(out)
        out = self.conv_2(out)
        out = self.relu(out)
        out = self.bn_3(out)
        out = self.conv_3(out)
        out = self.relu(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out

class Hourglass(nn.Module):
    def __init__(self, depth, nFeat, nModules, resBlocks):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.nFeat = nFeat
        self.nModules = nModules
        self.resBlocks = resBlocks
        self.hg = self._make_hourglass()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlocks(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_hourglass(self):
        hg = []
        for i in range(self.depth):
            res = [self._make_residual(self.nModules) for _ in range(3)]
            if i == (self.depth - 1):
                res.append(self._make_residual(self.nModules))      # extra one for the middle
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hourglass_forward(self, depth_id, x):
        up_1 = self.hg[depth_id][0](x)
        low_1 = self.downsample(x)
        low_1 = self.hg[depth_id][1](low_1)
        if depth_id == (self.depth - 1):
            low_2 = self.hg[depth_id][3](low_1)
        else:
            low_2 = self._hourglass_forward(depth_id + 1, low_1)
        low_3 = self.hg[depth_id][2](low_2)
        up_2 = self.upsample(low_3)
        return up_1 + up_2

    def forward(self, x):
        return self._hourglass_forward(0, x)

class HourglassNet(nn.Module):
    def __init__(self, nStacks, nModules, nFeat, nClasses, resBlock=HgResBlock, inplanes=3):
        super(HourglassNet, self).__init__()
        self.nStacks = nStacks
        self.nModules = nModules
        self.nFeat = nFeat
        self.nClasses = nClasses
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.reg_hm = nn.Conv2d(nFeat, nClasses, kernel_size=3, padding=1)
        self.reg_center = nn.Conv2d(nFeat, 2, kernel_size=3, padding=1)
        self.reg_vertex = nn.Conv2d(nFeat, 16, kernel_size=3, padding=1)
        self.reg_size = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1)

        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(nStacks):
            hg.append(Hourglass(depth=4, nFeat=nFeat, nModules=nModules, resBlocks=resBlock))
            res.append(self._make_residual(nModules))
            fc.append(self._make_fc(inplanes=nFeat, outplanes=nFeat))
            score.append(nn.Conv2d(nFeat, nClasses, kernel_size=1))
            if i < (nStacks - 1):
                fc_.append(nn.Conv2d(nFeat, nFeat, kernel_size=1))
                score_.append(nn.Conv2d(nClasses, nFeat, kernel_size=1))

        self.head = self._make_head()
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_head(self):
        self.conv_1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2, padding=3) # downsample scale - 2
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.res_1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)  # downsample scale - 2
        self.res_2 = self.resBlock(128, 128)
        self.res_3 = self.resBlock(128, self.nFeat)
        return nn.Sequential(*[self.conv_1, self.bn_1, self.relu, self.res_1, self.pool, self.res_2, self.res_3])

    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU())
    
    def freeze_backbone(self):
        freeze_list = [self.head, self.hg]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        freeze_list = [self.head, self.hg]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x):
        # head
        x = self.head(x)
        output = []
        for i in range(self.nStacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            output.append(score)
            if i < (self.nStacks - 1):
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
            else:
                # last regression
                bt_hm = self.reg_hm(y)
                bt_center = self.reg_center(y)
                bt_vertex = self.reg_vertex(y)
                bt_size = self.reg_size(y)
        return bt_hm, bt_center, bt_vertex, bt_size


if __name__ == "__main__":
    fp = torch.randn((1, 3, 512, 512))
    model = HourglassNet(5, 2, 256, 3, HgResBlock, inplanes=3)
    bt_hm, bt_center, bt_vertex, bt_size = model(fp)
    print(bt_center.shape)
    print(bt_size.shape)
    # 输出summary的时候，model中返回特征图不能以list形式打包返回
    print(summary(model,(3, 512, 512), batch_size=1, device='cpu'))
    