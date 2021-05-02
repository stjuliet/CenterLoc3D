# train datasets
import os
import time
import cv2 as cv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping

from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from fpn import KeyPointDetection
from hourglass import HourglassNet, HgResBlock
from loss import focal_loss, reg_l1_loss, reproject_l1_loss
from dataloader import Bbox3dDatasets, bbox3d_dataset_collate

mean = [0.40789655, 0.44719303, 0.47026116]
std = [0.2886383, 0.27408165, 0.27809834]


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_lr(optimizer):
    '''get lr value'''
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, writer):
    """
    func: 训练一个epoch
    epoch: 当前轮数
    epoch_size: 总训练迭代数
    epoch_size_val: 总验证迭代数
    gen: 训练数据
    genval: 验证数据
    Epoch: 总训练轮数
    cuda: 是否使用cuda
    writer: tensorboard绘制loss曲线图
    """
    global train_tensorboard_step, val_tensorboard_step
    total_cls_loss, total_center_off_loss, total_vertex_loss, total_size_loss, total_reproj_loss, total_loss = 0, 0, 0, 0, 0, 0
    total_train_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

            batch_imgs, batch_calib_matrixs, batch_hms, batch_center_regs, batch_vertex_regs, batch_size_regs, batch_center_masks, batch_box_perspectives, batch_raw_box_base_points, raw_img_ws, raw_img_hs = batch
            optimizer.zero_grad()

            pred_hm, pred_center, pred_vertex, pred_size = net(batch_imgs)

            cls_loss = focal_loss(pred_hm, batch_hms)
            center_off_loss = reg_l1_loss(pred_center, batch_center_regs, batch_center_masks, index = 2)
            vertex_loss = 0.1*reg_l1_loss(pred_vertex, batch_vertex_regs, batch_center_masks, index = 16)
            size_loss = 0.1*reg_l1_loss(pred_size, batch_size_regs, batch_center_masks, index = 3)
            reproj_loss = 0.1*reproject_l1_loss(pred_vertex, batch_calib_matrixs, pred_size, batch_center_masks, batch_raw_box_base_points, 16, batch_box_perspectives, output_shape, input_shape, raw_img_hs, raw_img_ws)
            
            loss = cls_loss + center_off_loss + vertex_loss + size_loss + reproj_loss

            total_train_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_center_off_loss += center_off_loss.item()
            total_vertex_loss += vertex_loss.item()
            total_size_loss += size_loss.item()
            total_reproj_loss += reproj_loss.item()

            loss.backward()
            optimizer.step()
            
            # 每一个iteration都写入  step从1开始
            writer.add_scalar('Train_loss', loss, train_tensorboard_step)
            train_tensorboard_step += 1

            pbar.set_postfix(**{'loss'  : total_train_loss / (iteration + 1),
                                'cls_loss'  : total_cls_loss / (iteration + 1),
                                'center_loss'  : total_center_off_loss / (iteration + 1),
                                'vertex_loss'  : total_vertex_loss / (iteration + 1),
                                'size_loss'  : total_size_loss / (iteration + 1),
                                'reproj_loss'  : total_reproj_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)

    # 将loss写入tensorboard，下面注释的是每个世代保存一次
    # writer.add_scalar('Train_loss', total_train_loss/(iteration+1), epoch)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]
                
                batch_imgs, batch_calib_matrixs, batch_hms, batch_center_regs, batch_vertex_regs, batch_size_regs, batch_center_masks, batch_box_perspectives, batch_raw_box_base_points, raw_img_ws, raw_img_hs = batch

                pred_hm, pred_center, pred_vertex, pred_size = net(batch_imgs)

                cls_loss = focal_loss(pred_hm, batch_hms)
                center_off_loss = reg_l1_loss(pred_center, batch_center_regs, batch_center_masks, index = 2)
                vertex_loss = 0.1*reg_l1_loss(pred_vertex, batch_vertex_regs, batch_center_masks, index = 16)
                size_loss = 0.1*reg_l1_loss(pred_size, batch_size_regs, batch_center_masks, index = 3)
                reproj_loss = 0.1*reproject_l1_loss(pred_vertex, batch_calib_matrixs, pred_size, batch_center_masks, batch_raw_box_base_points, 16, batch_box_perspectives, output_shape, input_shape, raw_img_hs, raw_img_ws)
                
                loss = cls_loss + center_off_loss + vertex_loss + size_loss + reproj_loss

                val_loss += loss.item()

                # writer.add_scalar('Val_loss', loss, val_tensorboard_step)
                # val_tensorboard_step += 1

                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    # 每个epoch写入一次  epoch从1开始
    writer.add_scalar('Val_loss', val_loss / (epoch_size_val+1), epoch + 1)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total train loss: %.4f || Val loss: %.4f ' % (total_train_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_train_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_train_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)
    
if __name__ == "__main__":
    # 输入图片的大小
    input_shape = (512, 512, 3)
    output_shape = (128, 128)

    # 类别文件
    classes_path = 'model_data/classes.txt'

    # 获取classes和数量
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    # 是否使用imagenet-resnet50的预训练权重。
    pretrain = True

    # 指定backbone
    backbone = "resnet50"

    # 是否使用Cuda
    Cuda = True

    # 获取模型
    backbone_resnet_index = {"resnet18": 0, "resnet34": 1, "resnet50": 2, "resnet101": 3, "resnet152": 4}
    if backbone[:-2] == "resnet":
        model = KeyPointDetection(model_index=backbone_resnet_index[backbone], num_classes=num_classes, pretrained_weights=pretrain)
    if backbone == "hourglass":
        model = HourglassNet(2, 1, 256, 3, HgResBlock, inplanes=3)


    # 加载训练好的模型权重
    # 断点续训练
    model_path = r"logs\Epoch60-Total_train_Loss8.5380-Val_Loss7.4408.pth"
    print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict,strict=False)

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    # 设置早停
    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 获得图片路径和标签
    annotation_path = 'DATA2021_train.txt'

    # 划分验证集
    # 验证集和训练集的比例为1:9
    # 8:1:1
    val_split = 1/9
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # 构建绘制loss曲线图writer
    writer = SummaryWriter(log_dir='train-logs',flush_secs=60)
    if Cuda:
        graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor).cuda()
    else:
        graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor)
    writer.add_graph(model, (graph_inputs,))

    train_tensorboard_step = 3300 + 1  # 断点续训练
    val_tensorboard_step = 1
    
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size

    if True:
        # 超参数设置
        lr = 3.125e-5
        Batch_size = 4
        Init_Epoch = 50
        Freeze_Epoch = 60
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

        train_dataset = Bbox3dDatasets(lines[:num_train], input_shape, num_classes, True)
        val_dataset = Bbox3dDatasets(lines[num_train:], input_shape, num_classes, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                                drop_last=True, collate_fn=bbox3d_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8,pin_memory=True,
                                drop_last=True, collate_fn=bbox3d_dataset_collate)

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size

        # 冻结一定部分训练
        model.freeze_backbone()

        # # 开始训练前展示训练样本
        # for step, data in enumerate(gen, start = 0):
        #     # batch_center_reg: 中心点偏移量
        #     img, calib_matrix, batch_hm, batch_center_reg, batch_vertex_reg, batch_size_reg, batch_center_mask, batch_box_perspective, batch_base_point, raw_img_w, raw_img_h = data
            
        #     for num in range(len(img)):
        #         plt.figure()
        #         raw_img = img[num] # numpy格式, BGR, CHW
        #         raw_img = raw_img.transpose(1, 2, 0)  # CHW->HWC
        #         raw_img = np.array((raw_img * std + mean) * 255.).astype(np.uint8)
        #         raw_img = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)  # RGB

        #         max_size=max(raw_img_w[num],raw_img_h[num])
        #         minus_size=abs(raw_img_w[num]-raw_img_h[num])

        #         pil_raw_img=Image.fromarray(raw_img).resize((max_size,max_size)) # 返回PIL格式带灰条原图
        #         draw=ImageDraw.Draw(pil_raw_img)

        #         # 绘制中心点
        #         t_list = np.where(batch_center_mask[num] == 1.0)  # 遍历图像域中所有目标点
        #         for y, x in zip(t_list[0], t_list[1]):
        #             # 显示到带灰条原图上
        #             center = ([x, y] + batch_center_reg[num, y, x]) * max_size // output_shape[0]
        #             vertex = batch_vertex_reg[num, y, x] * max_size // output_shape[0]

        #             draw.ellipse([center[0], center[1], center[0]+5, center[1]+5], outline=(0, 0, 255), width = 1)
        #             cls_id = np.argmax(batch_hm[num, y, x])
        #             cls_name = class_names[cls_id]
        #             draw.text([center[0], center[1]-10], cls_name, fill=(255, 0, 0))

        #             # 宽度方向
        #             # 0-1  2-3  4-5  6-7
        #             draw.line([vertex[0], vertex[1], vertex[2], vertex[3]], fill=128, width=2)
        #             draw.line([vertex[4], vertex[5], vertex[6], vertex[7]], fill=128, width=2)
        #             draw.line([vertex[8], vertex[9], vertex[10], vertex[11]], fill=128, width=2)
        #             draw.line([vertex[12], vertex[13], vertex[14], vertex[15]], fill=128, width=2)

        #             # 长度方向
        #             # 0-3 1-2 4-7 5-6
        #             draw.line([vertex[0], vertex[1], vertex[6], vertex[7]], fill=128, width=2)
        #             draw.line([vertex[2], vertex[3], vertex[4], vertex[5]], fill=128, width=2)
        #             draw.line([vertex[8], vertex[9], vertex[14], vertex[15]], fill=128, width=2)
        #             draw.line([vertex[10], vertex[11], vertex[12], vertex[13]], fill=128, width=2)

        #             # 高度方向
        #             # 0-4 1-5 2-6 3-7
        #             draw.line([vertex[0], vertex[1], vertex[8], vertex[9]], fill=128, width=2)
        #             draw.line([vertex[2], vertex[3], vertex[10], vertex[11]], fill=128, width=2)
        #             draw.line([vertex[4], vertex[5], vertex[12], vertex[13]], fill=128, width=2)
        #             draw.line([vertex[6], vertex[7], vertex[14], vertex[15]], fill=128, width=2)
        #         # del draw

        #         plt.subplot(2,2,1)
        #         plt.imshow(pil_raw_img)  # RGB

        #         # 绘制热力图
        #         plt.subplot(2,2,2)
        #         hotmaps = batch_hm[num][...,0]
        #         print(hotmaps.shape)
        #         heatmap = np.maximum(hotmaps, 0)
        #         heatmap /= np.max(heatmap)
        #         plt.imshow(heatmap)

        #         # 将灰度图转换为伪彩色图
        #         plt.subplot(2,2,3)
        #         heatmap = cv.resize(heatmap, (512, 512))
        #         heatmap = np.uint8(255 * heatmap)
        #         heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        #         raw_img = cv.cvtColor(raw_img, cv.COLOR_RGB2BGR)
        #         superimposed_img = heatmap * 0.4 + raw_img * 0.8  # BGR

        #         cv.imwrite("img/hot.jpg", superimposed_img)
        #         superimposed_img = np.array(superimposed_img, dtype=np.uint8)
        #         superimposed_img = cv.cvtColor(superimposed_img, cv.COLOR_BGR2RGB)
        #         plt.imshow(superimposed_img)

        #     plt.show()


        for epoch in range(Init_Epoch,Freeze_Epoch):
            val_loss = fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda, writer)
            lr_scheduler.step(val_loss)

    if True:
        # 超参数设置
        lr = 1e-4
        Batch_size = 8
        Freeze_Epoch = 60
        Unfreeze_Epoch = 120

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

        train_dataset = Bbox3dDatasets(lines[:num_train], input_shape, num_classes, True)
        val_dataset = Bbox3dDatasets(lines[num_train:], input_shape, num_classes, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                                drop_last=True, collate_fn=bbox3d_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8,pin_memory=True, 
                                drop_last=True, collate_fn=bbox3d_dataset_collate)

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size

        # 解冻后训练
        model.unfreeze_backbone()

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            val_loss = fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda, writer)
            lr_scheduler.step(val_loss)
            
            early_stopping(val_loss, net)
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                break

