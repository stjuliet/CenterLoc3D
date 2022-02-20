# train datasets
import os
import time
import cv2 as cv
import numpy as np

import argparse
import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.pytorchtools import EarlyStopping

from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from nets.fpn import KeyPointDetection
from nets.hourglass_official import HourglassNet, Bottleneck
from utils.loss import focal_loss, reg_l1_loss, reproject_l1_loss, reg_iou_loss
from dataset.dataloader import Bbox3dDatasets, bbox3d_dataset_collate


mean = [0.40789655, 0.44719303, 0.47026116]
std = [0.2886383, 0.27408165, 0.27809834]


def get_classes(classes_path):
    """ loads the classes """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_lr(optimizer):
    """ get lr value """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, backbone, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, iou_type, cuda, writer):
    """
    func: train one epoch
    net:
    backbone:
    epoch: current epoch
    epoch_size: all train epoch
    epoch_size_val: all val epoch
    gen: train data
    genval: val data
    Epoch: all epoch
    cuda: gpu use
    writer: tensorboard loss visualization
    """
    global train_tensorboard_step, val_tensorboard_step
    total_cls_loss, total_center_off_loss, total_vertex_loss, total_size_loss, total_reproj_loss = 0, 0, 0, 0, 0
    total_iou_loss = 0
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

            if iou_type:
                iou_loss = reg_iou_loss(iou_type, pred_vertex, batch_vertex_regs, batch_center_masks, batch_box_perspectives, output_shape, input_shape, raw_img_hs, raw_img_ws)
                if iou_loss > 1.0:
                    iou_loss = torch.log(iou_loss)
                else:
                    pass
                loss = cls_loss + center_off_loss + vertex_loss + size_loss + reproj_loss + iou_loss
            else:
                iou_loss = 0.0
                loss = cls_loss + center_off_loss + vertex_loss + size_loss + reproj_loss

            total_train_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_center_off_loss += center_off_loss.item()
            total_vertex_loss += vertex_loss.item()
            total_size_loss += size_loss.item()
            total_reproj_loss += reproj_loss.item()

            if iou_type:
                total_iou_loss += iou_loss.item()

            loss.backward()
            optimizer.step()
            
            # each iteration, step start from 1
            writer.add_scalar('Train_loss', loss, train_tensorboard_step)
            train_tensorboard_step += 1

            pbar.set_postfix(**{'loss'                  : total_train_loss / (iteration + 1),
                                'cls_loss'              : total_cls_loss / (iteration + 1),
                                'center_loss'           : total_center_off_loss / (iteration + 1),
                                'vertex_loss'           : total_vertex_loss / (iteration + 1),
                                'size_loss'             : total_size_loss / (iteration + 1),
                                'reproj_loss'           : total_reproj_loss / (iteration + 1),
                                '%s_loss' % iou_type    : total_iou_loss / (iteration + 1),
                                'lr'                    : get_lr(optimizer)})
            pbar.update(1)

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

                if iou_type:
                    iou_loss = reg_iou_loss(iou_type, pred_vertex, batch_vertex_regs, batch_center_masks, batch_box_perspectives, output_shape, input_shape, raw_img_hs, raw_img_ws)
                    if iou_loss > 1.0:
                        iou_loss = torch.log(iou_loss)
                    else:
                        pass
                    loss = cls_loss + center_off_loss + vertex_loss + size_loss + reproj_loss + iou_loss
                else:
                    iou_loss = 0.0
                    loss = cls_loss + center_off_loss + vertex_loss + size_loss + reproj_loss

                val_loss += loss.item()

                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    # each epoch, step start from 1
    writer.add_scalar('Val_loss', val_loss / (epoch_size_val+1), epoch + 1)

    print('Finish Validation')
    print('Epoch:' + str(epoch+1) + '/' + str(Epoch))
    print('Total train loss: %.4f || Val loss: %.4f ' % (total_train_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))

    # save epoch, model parameters, optimizer into a dict
    # for reserve training
    state = {"epoch": epoch + 1, 
             "state_dict": model.state_dict(), 
             "optimizer": optimizer.state_dict()}

    torch.save(state, '../logs/%s-Epoch%d-%s-Total_train_Loss%.4f-Val_Loss%.4f.pth'%(backbone,(epoch+1),iou_type,total_train_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)


def train_from_checkpoint(model_path, model):
    """ reserve training """
    checkpoint = torch.load(model_path)

    pretrained_model_dict = checkpoint["state_dict"]
    optimizer_dict = checkpoint["optimizer"]
    start_epoch = checkpoint["epoch"]
    start_lr = optimizer_dict['param_groups'][0]['lr']

    model_dict = model.state_dict()
    update_model_dict = {k: v for k, v in pretrained_model_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(update_model_dict)
    model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=5e-4)
    optimizer.load_state_dict(optimizer_dict)

    print('checkpoint loaded!')

    Batch_size = batch_size_list[1]
    Freeze_Epoch = start_epoch
    Unfreeze_Epoch = 120 - start_epoch + 1

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    train_dataset = Bbox3dDatasets(lines[:num_train], input_shape, num_classes, True)
    val_dataset = Bbox3dDatasets(lines[num_train:], input_shape, num_classes, False)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                            drop_last=True, collate_fn=bbox3d_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8,pin_memory=True, 
                            drop_last=True, collate_fn=bbox3d_dataset_collate)

    epoch_size = num_train//Batch_size
    epoch_size_val = num_val//Batch_size

    model.unfreeze_backbone()

    for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
        val_loss = fit_one_epoch(net,args.backbone,optimizer,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,args.iou_loss_type,args.gpu, writer)
        lr_scheduler.step(val_loss)
        
        early_stopping(val_loss, net)
        # early stopping
        if early_stopping.early_stop:
            print("Early stopping")
            # stop training
            break


def get_arguments_for_train(list_backbones):
    """ get training parameters """
    parser = argparse.ArgumentParser("CenterLoc3D")
    # dataset
    parser.add_argument("-classes_path", type=str, default="../model_data/classes.txt", help="classes file path")
    parser.add_argument("-num_workers", type=int, default=8, help="dataloader num workers")
    # model
    parser.add_argument("-input_size", type=str, default="512, 512, 3", help="network input img size")
    parser.add_argument("-backbone", type=str, default="resnet50", choices=list_backbones, help="network backbone")
    parser.add_argument("-pretrain", type=bool, default=True, help="use pretrained model for fine-tuning")
    parser.add_argument("-deform", type=bool, default=False, help="use deformable convs in network")
    parser.add_argument("-iou_loss_type", type=str, default="ciou", choices=["iou", "giou", "diou", "ciou", "cdiou"], help="use iou loss type in training")
    # hyper params in training
    parser.add_argument("-reverse_train", type=bool, default=False, help="train from previous weights status")
    parser.add_argument("-reverse_train_model_path", type=str, default=None, help="previous weights filepath")
    parser.add_argument("-freeze_lr", type=float, default=1e-3, help="learning rate in freeze training step")
    parser.add_argument("-unfreeze_lr", type=float, default=1e-4, help="learning rate in unfreeze training step")
    parser.add_argument("-freeze_epoch", type=int, default=60, help="freeze_epoch in freeze training step")
    parser.add_argument("-unfreeze_epoch", type=int, default=100, help="unfreeze_epoch in unfreeze training step")
    parser.add_argument("-freeze_batch_size", type=int, default=16, help="batch size in training step")
    parser.add_argument("-unfreeze_batch_size", type=int, default=8, help="batch size in training step")
    parser.add_argument("-early_stopping_patience", type=int, default=7, help="early stopping patience")
    # others (path, etc.)
    parser.add_argument("-model_train_txt_path", type=str, default="../dataset/DATA2021_train.txt", help="model train txt file path")
    parser.add_argument("-model_save_dir", type=str, default="../logs", help="model save directory")
    parser.add_argument("-train_logs_save_dir", type=str, default="../train-logs", help="train logs for tensorboard save directory")
    parser.add_argument('-gpu', type=bool, default=True, help="gpu status")
    parser.add_argument('-gpu_index', type=str, default="0", help="gpu index")

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    # model index
    backbone_resnet_index = {"resnet18": 0, "resnet34": 1, "resnet50": 2, "resnet101": 3, "resnet152": 4}
    backbone_efficientnet_index = {"efficientnetb0": 0, "efficientnetb1": 1, "efficientnetb2": 2,
                     "efficientnetb3": 3, "efficientnetb4": 4, "efficientnetb5": 5, "efficientnetb6": 6, "efficientnetb7": 7}
    list_backbones = list(backbone_resnet_index.keys()) + list(backbone_efficientnet_index.keys()) + ["hourglass", "darknet"]

    # model - batch_size
    batch_size_dict = {"resnet":[16, 8], "efficientnet":[16, 4], "hourglass":[2, 1], "darknet":[16, 8]}
    # ------------------------------------------------------------------------------------

    # add parameters !!!
    args = get_arguments_for_train(list_backbones)

    # ------------------------dataset-----------------------------------------------------
    # get cls, num_cls
    class_names = get_classes(args.classes_path)
    num_classes = len(class_names)

    # train txt
    annotation_path = args.model_train_txt_path

    # train/val/test
    # 8:1:1
    val_split = 1/9
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    with open("../DATAdevkit/DATA2021/ImageSets/Main/val.txt", "w") as fval:
        for line in lines[num_train:]:
            fval.write(line.split(" ")[0].split("/")[-1].split(".")[0] + "\n")
    # ------------------------dataset-----------------------------------------------------

    # ------------------------model-----------------------------------------------------
    # backbone
    if args.backbone == "darknet":
        # input size
        input_shape = tuple(map(int, args.input_size.split(",")))
        print(input_shape)
        output_shape = (input_shape[0] // 4, input_shape[1] // 4)
    else:
        # input size
        input_shape = tuple(map(int, args.input_size.split(",")))
        output_shape = (input_shape[0] // 4, input_shape[1] // 4)

    # build model, hyper-parameters
    batch_size_list = []
    if args.backbone[:-2] == "resnet":
        model = KeyPointDetection(model_name=args.backbone[:-2], model_index=backbone_resnet_index[args.backbone], num_classes=num_classes, pretrained_weights=args.pretrain)
        batch_size_list = batch_size_dict[args.backbone[:-2]]
    if args.backbone[:-2] == "efficientnet":
        model = KeyPointDetection(model_name=args.backbone[:-2], model_index=backbone_efficientnet_index[args.backbone], num_classes=num_classes, pretrained_weights=args.pretrain)
        batch_size_list = batch_size_dict[args.backbone[:-2]]
    if args.backbone == "hourglass":
        model = HourglassNet(Bottleneck, num_stacks=8, num_blocks=1, num_classes=args.num_classes)
        batch_size_list = batch_size_dict[args.backbone]
        if args.pretrain:  # hourglass
            model_path = "../model_data/hourglass-s8b1-best.pth.tar"
            print('Loading pretrained weights into state dict...')

            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path)
            state_dict = pretrained_dict['state_dict']
            matched_model_dict = ["module." + mdk for mdk in model_dict.keys()]
            final_model_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if k in matched_model_dict and model_dict[k.replace("module.", "")].shape == v.shape}
            model_dict.update(final_model_dict)
            model.load_state_dict(model_dict, strict=False)
            print('Finished!')
    if args.backbone == "darknet":
        if args.deform:
            batch_size_dict.update({"darknet": [16, 8]})
        else:
            batch_size_dict.update({"darknet": [16, 8]})
        model = KeyPointDetection(model_name=args.backbone, model_index=0, num_classes=args.num_classes, pretrained_weights=args.pretrain, deform=args.deform)
        batch_size_list = batch_size_dict[args.backbone]

    # gpu training
    if args.gpu:
        # multi-gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            net = net.cuda()
        else:
            net = model.cuda()
    else:
        net = model
    # ------------------------model-----------------------------------------------------

    # ------------------------hyper parameters-----------------------------------------------------
    # batch_size
    if batch_size_list:
        args.freeze_batch_size = batch_size_list[0]
        args.unfreeze_batch_size = batch_size_list[1]

    # early stopping
    # prevent overfitting
    patience = args.early_stopping_patience
    best_model_path = "../logs/%s-best-checkpoint.pth" % args.backbone
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=best_model_path)

    # ------------------------hyper parameters-----------------------------------------------------

    # ------------------------others-----------------------------------------------------
    # model save dir
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    # tensorboard writer
    writer = SummaryWriter(log_dir=args.train_logs_save_dir, flush_secs=60)
    if not args.deform:
        if args.gpu:
            graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor)
        writer.add_graph(model, (graph_inputs,))

    train_tensorboard_step = 1
    val_tensorboard_step = 1

    train_cont = args.reverse_train
    train_cont_model_path = args.reverse_train_model_path

    # ------------------------others-----------------------------------------------------
    
    if train_cont:
        train_from_checkpoint(train_cont_model_path, model)
    else:
        if True:
            lr = args.freeze_lr
            Batch_size = args.freeze_batch_size
            Init_Epoch = 0
            Freeze_Epoch = args.freeze_epoch
            
            optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

            train_dataset = Bbox3dDatasets(lines[:num_train], input_shape, num_classes, True)
            val_dataset = Bbox3dDatasets(lines[num_train:], input_shape, num_classes, False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=args.num_workers, pin_memory=False,
                                    drop_last=True, collate_fn=bbox3d_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=args.num_workers, pin_memory=False,
                                    drop_last=True, collate_fn=bbox3d_dataset_collate)

            epoch_size = num_train//Batch_size
            epoch_size_val = num_val//Batch_size

            model.freeze_backbone()
            #
            # # show train samples, and save
            # if not os.path.exists("../batch_samples"):
            #     os.makedirs("../batch_samples")
            # for step, data in enumerate(gen, start=0):
            #     # batch_center_reg: center offset
            #     img, calib_matrix, batch_hm, batch_center_reg, batch_vertex_reg, batch_size_reg, batch_center_mask, batch_box_perspective, batch_base_point, raw_img_w, raw_img_h = data
            #
            #     for num in range(len(img)):
            #         plt.figure()
            #         raw_img = img[num] # numpy, BGR, CHW
            #         raw_img = raw_img.transpose(1, 2, 0)  # CHW->HWC
            #         raw_img = np.array((raw_img * std + mean) * 255.).astype(np.uint8)
            #         raw_img = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)  # RGB
            #
            #         max_size = max(raw_img_w[num],raw_img_h[num])
            #         minus_size = abs(raw_img_w[num]-raw_img_h[num])
            #
            #         pil_raw_img = Image.fromarray(raw_img).resize((max_size,max_size))
            #         draw=ImageDraw.Draw(pil_raw_img)
            #
            #         t_list = np.where(batch_center_mask[num] == 1.0)  # object
            #         for y, x in zip(t_list[0], t_list[1]):
            #             # show in resized img
            #             center = ([x, y] + batch_center_reg[num, y, x]) * max_size // output_shape[0]
            #             vertex = batch_vertex_reg[num, y, x] * max_size // output_shape[0]
            #
            #             draw.ellipse([center[0], center[1], center[0]+5, center[1]+5], outline=(0, 0, 255), width = 1)
            #             cls_id = np.argmax(batch_hm[num, y, x])
            #             cls_name = class_names[cls_id]
            #             draw.text([center[0], center[1]-10], cls_name, fill=(255, 0, 0))
            #
            #             # width
            #             # 0-1  2-3  4-5  6-7
            #             draw.line([vertex[0], vertex[1], vertex[2], vertex[3]], fill=128, width=2)
            #             draw.line([vertex[4], vertex[5], vertex[6], vertex[7]], fill=128, width=2)
            #             draw.line([vertex[8], vertex[9], vertex[10], vertex[11]], fill=128, width=2)
            #             draw.line([vertex[12], vertex[13], vertex[14], vertex[15]], fill=128, width=2)
            #
            #             # length
            #             # 0-3 1-2 4-7 5-6
            #             draw.line([vertex[0], vertex[1], vertex[6], vertex[7]], fill=128, width=2)
            #             draw.line([vertex[2], vertex[3], vertex[4], vertex[5]], fill=128, width=2)
            #             draw.line([vertex[8], vertex[9], vertex[14], vertex[15]], fill=128, width=2)
            #             draw.line([vertex[10], vertex[11], vertex[12], vertex[13]], fill=128, width=2)
            #
            #             # height
            #             # 0-4 1-5 2-6 3-7
            #             draw.line([vertex[0], vertex[1], vertex[8], vertex[9]], fill=128, width=2)
            #             draw.line([vertex[2], vertex[3], vertex[10], vertex[11]], fill=128, width=2)
            #             draw.line([vertex[4], vertex[5], vertex[12], vertex[13]], fill=128, width=2)
            #             draw.line([vertex[6], vertex[7], vertex[14], vertex[15]], fill=128, width=2)
            #         # del draw
            #
            #         plt.subplot(2,2,1)
            #         plt.imshow(pil_raw_img)  # RGB
            #         pil_raw_img.save("../batch_samples/batch_sample_%s.png" % str(num))
            #
            #         # heatmap
            #         plt.subplot(2,2,2)
            #         hotmaps = batch_hm[num][..., 0] # cls_index
            #         print(hotmaps.shape)
            #         heatmap = np.maximum(hotmaps, 0)
            #         heatmap /= np.max(heatmap)
            #         plt.imshow(heatmap)
            #
            #         # pseudo color map
            #         plt.subplot(2,2,3)
            #         heatmap = cv.resize(heatmap, (512, 512))
            #         heatmap = np.uint8(255 * heatmap)
            #         heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
            #         cv.imwrite("../batch_samples/batch_heatmap_%s.png" % str(num), heatmap)
            #         raw_img = cv.cvtColor(raw_img, cv.COLOR_RGB2BGR)
            #         superimposed_img = heatmap * 0.4 + raw_img * 0.8  # BGR
            #
            #         cv.imwrite("../batch_samples/batch_merge_%s.png" % str(num), superimposed_img)
            #         superimposed_img = np.array(superimposed_img, dtype=np.uint8)
            #         superimposed_img = cv.cvtColor(superimposed_img, cv.COLOR_BGR2RGB)
            #         plt.imshow(superimposed_img)
            #
            #     plt.show()

            for epoch in range(Init_Epoch, Freeze_Epoch):
                val_loss = fit_one_epoch(net,args.backbone,optimizer,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,args.iou_loss_type,args.gpu, writer)
                lr_scheduler.step(val_loss)

        if True:
            # 超参数设置
            lr = args.unfreeze_lr
            Batch_size = args.unfreeze_batch_size
            Freeze_Epoch = args.freeze_epoch
            Unfreeze_Epoch = args.unfreeze_epoch

            optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

            train_dataset = Bbox3dDatasets(lines[:num_train], input_shape, num_classes, True)
            val_dataset = Bbox3dDatasets(lines[num_train:], input_shape, num_classes, False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=args.num_workers, pin_memory=False,
                                    drop_last=True, collate_fn=bbox3d_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=args.num_workers, pin_memory=False,
                                    drop_last=True, collate_fn=bbox3d_dataset_collate)

            epoch_size = num_train//Batch_size
            epoch_size_val = num_val//Batch_size

            model.unfreeze_backbone()

            for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
                val_loss = fit_one_epoch(net,args.backbone,optimizer,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,args.iou_loss_type,args.gpu, writer)
                lr_scheduler.step(val_loss)
                
                early_stopping(val_loss, net)
                # early stopping
                if early_stopping.early_stop:
                    print("Early stopping")
                    # stop training
                    break
