import os
import numpy as np
from utils import basic_3diou
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

mode = "test"

MINOVERLAP = 0.7
MATCHED_NUM = 0

# 是否导出定位可视化
vis_pos = False
vis_curve = True

det_txt_dir = "%s/input-3D/detection-results" % mode
gt_txt_dir = "%s/input-3D/ground-truth" % mode

if not os.path.exists("%s/input-3D/visualize-pos" % mode):
    os.makedirs("%s/input-3D/visualize-pos" % mode)

if not os.path.exists("%s/input-3D/visualize-loc-curve" % mode):
    os.makedirs("%s/input-3D/visualize-loc-curve" % mode)

list_det_txt = sorted(os.listdir(det_txt_dir))
list_gt_txt = sorted(os.listdir(gt_txt_dir))

total_size_error, total_loc_error = 0.0, 0.0
single_size_error, single_loc_error = [], []
l_error, w_error, h_error = [], [], []

tp_sizes_dt, tp_locs_dt, tp_sizes_gt, tp_locs_gt = [], [], [], []  # 保存TP的预测值和真实值

# 统计一种场景下的xc,yc,zc, l,w,h的误差，单位：m
s_xc_error, s_yc_error, s_zc_error, s_loc_error = [], [], [], []
s_l_error, s_w_error, s_h_error, s_size_error = [], [], [], []
s_gt_yc = []


font_legend = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 40,
        }

font_label = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 40,
        }

for i in tqdm(range(len(list_det_txt))):  # 循环文件
    # 每个场景生成一个折线图保存！
    if vis_curve:
        fig, ax = plt.subplots(figsize=(20,15),dpi=100)
        plt.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()

        [label.set_fontname('Times New Roman') for label in labels]

        plt.xlabel("ground truth distance/m", font_label)
        plt.ylabel("error/m", font_label)

        # 坐标刻度朝内
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
    
    # 每个文件生成一个散点图保存！
    if vis_pos:
        fig, ax = plt.subplots(figsize=(15,20),dpi=100)
        plt.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()

        [label.set_fontname('Times New Roman') for label in labels]

        plt.xlabel("x/m", font_label)
        plt.ylabel("y/m", font_label)

        # 坐标刻度朝内
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.grid(True,linestyle='--',alpha=0.5)

    f_det = open(os.path.join(det_txt_dir, list_det_txt[i]), "r")
    list_dets = f_det.readlines()
    f_gt = open(os.path.join(gt_txt_dir, list_gt_txt[i]), "r")
    list_gts = f_gt.readlines()

    # 根据不同场景确定有效视野范围
    if list_det_txt[i].split("_")[0] == "session0":
        valid_pers = 150.0*1000 /2  # 有效视野范围
    elif list_det_txt[i].split("_")[0] == "session6":
        valid_pers = 100.0*1000 /2  # 有效视野范围

    for line_det in list_dets:  # 循环检测值
        for line_gt in list_gts:  # 循环真值
            det = line_det.split()
            gt = line_gt.split()

            bb_det = np.array(det[2:26]).astype(np.float)
            bb_gt = np.array(gt[1:25]).astype(np.float)

            if vis_pos:
                # 无论是否匹配到都画出结果
                cx_gt_unmatched, cy_gt_unmatched, cz_gt_unmatched = np.array(gt[28:31]).astype(np.float)
                cx_det_unmatched, cy_det_unmatched, cz_det_unmatched = np.array(det[29:32]).astype(np.float)
                det_plot = plt.scatter(cx_det_unmatched/1000, cy_det_unmatched/1000, s=800, c="b", marker="s", label="detection")   # det 蓝色方块表示
                gt_plot = plt.scatter(cx_gt_unmatched/1000, cy_gt_unmatched/1000, s=1000, c="r", marker="*", label="ground truth")   # gt 红色五角星表示

                plt.legend(handles=[gt_plot, det_plot], prop=font_legend)
            
            ov = basic_3diou(bb_det, bb_gt)

            if ov > MINOVERLAP:  # 标记匹配到
                MATCHED_NUM += 1
                l_dt, w_dt, h_dt = np.array(det[26:29]).astype(np.float)
                cx_dt, cy_dt, cz_dt = np.array(det[29:32]).astype(np.float)
                l_gt, w_gt, h_gt = np.array(gt[25:28]).astype(np.float)
                cx_gt, cy_gt, cz_gt = np.array(gt[28:31]).astype(np.float)
                # 保存tp的预测值和真实值，用于记录至txt中
                tp_sizes_dt.append([l_dt, w_dt, h_dt])
                tp_sizes_gt.append([l_gt, w_gt, h_gt])
                tp_locs_dt.append([cx_dt, cy_dt, cz_dt])
                tp_locs_gt.append([cx_gt, cy_gt, cz_gt])
                l_error.append(abs(l_dt-l_gt))
                w_error.append(abs(w_dt-w_gt))
                h_error.append(abs(h_dt-h_gt))
                # 保存单个样本误差，用于记录至txt中
                tmp_size_error = abs(l_dt-l_gt)/l_gt + abs(w_dt-w_gt)/w_gt + abs(h_dt-h_gt)/h_gt
                tmp_loc_error = math.sqrt((cx_dt-cx_gt)**2+(cy_dt-cy_gt)**2+(cz_dt-cz_gt)**2)/valid_pers
                single_size_error.append(tmp_size_error)
                single_loc_error.append(tmp_loc_error)
                # 累计误差
                total_loc_error += tmp_loc_error
                total_size_error += tmp_size_error

                # 匹配上才记录真实位置和误差
                if vis_curve:
                    s_xc_error.append(abs(cx_dt/1000-cx_gt/1000))
                    s_yc_error.append(abs(cy_dt/1000-cy_gt/1000))
                    s_zc_error.append(abs(cz_dt/1000-cz_gt/1000))
                    s_loc_error.append(abs(cx_dt/1000-cx_gt/1000)+abs(cy_dt/1000-cy_gt/1000)+abs(cz_dt/1000-cz_gt/1000))
                    s_l_error.append(abs(l_dt-l_gt))
                    s_w_error.append(abs(w_dt-w_gt))
                    s_h_error.append(abs(h_dt-h_gt))
                    s_size_error.append(abs(l_dt-l_gt)+abs(w_dt-w_gt)+abs(h_dt-h_gt))
                    s_gt_yc.append(cy_gt/1000)
    if vis_curve and i > 0 and list_det_txt[i].split("_")[:2] != list_det_txt[i-1].split("_")[:2]:
        # 绘图保存后清空所有变量
        # xc_plot = plt.plot(sorted(s_gt_yc), sorted(s_xc_error), color="green", linewidth=2, label="X error")
        # yc_plot = plt.plot(sorted(s_gt_yc), sorted(s_yc_error), color="red", linewidth=2, label="Y error")
        # zc_plot = plt.plot(sorted(s_gt_yc), sorted(s_zc_error), color="blue", linewidth=2, label="Z error")
        # loc_plot = plt.plot(sorted(s_gt_yc), sorted(s_loc_error), color="orange", linewidth=2, label="total error")

        l_plot = plt.plot(sorted(s_gt_yc), sorted(s_l_error), color="green", linewidth=2, label="l error")
        w_plot = plt.plot(sorted(s_gt_yc), sorted(s_w_error), color="red", linewidth=2, label="w error")
        h_plot = plt.plot(sorted(s_gt_yc), sorted(s_h_error), color="blue", linewidth=2, label="h error")
        size_plot = plt.plot(sorted(s_gt_yc), sorted(s_size_error), color="orange", linewidth=2, label="total error")

        plt.legend(loc="best", prop=font_legend)

        # plt.savefig(os.path.join("%s/input-3D/visualize-loc-curve" % mode, list_det_txt[i].split(".")[0] + "_vis_loc_curve.png"))
        plt.savefig(os.path.join("%s/input-3D/visualize-loc-curve" % mode, list_det_txt[i].split(".")[0] + "_vis_size_curve.png"))
        plt.close()

        s_xc_error.clear()
        s_yc_error.clear()
        s_zc_error.clear()
        s_loc_error.clear()
        s_l_error.clear()
        s_w_error.clear()
        s_h_error.clear()
        s_size_error.clear()
        s_gt_yc.clear()

    if vis_curve and i == len(list_det_txt)-1:
        # xc_plot = plt.plot(sorted(s_gt_yc), sorted(s_xc_error), color="green", linewidth=2, label="X error")
        # yc_plot = plt.plot(sorted(s_gt_yc), sorted(s_yc_error), color="red", linewidth=2, label="Y error")
        # zc_plot = plt.plot(sorted(s_gt_yc), sorted(s_zc_error), color="blue", linewidth=2, label="Z error")
        # loc_plot = plt.plot(sorted(s_gt_yc), sorted(s_loc_error), color="orange", linewidth=2, label="total error")

        l_plot = plt.plot(sorted(s_gt_yc), sorted(s_l_error), color="green", linewidth=2, label="l error")
        w_plot = plt.plot(sorted(s_gt_yc), sorted(s_w_error), color="red", linewidth=2, label="w error")
        h_plot = plt.plot(sorted(s_gt_yc), sorted(s_h_error), color="blue", linewidth=2, label="h error")
        size_plot = plt.plot(sorted(s_gt_yc), sorted(s_size_error), color="orange", linewidth=2, label="total error")

        plt.legend(loc="best", prop=font_legend)

        # plt.savefig(os.path.join("%s/input-3D/visualize-loc-curve" % mode, list_det_txt[i].split(".")[0] + "_vis_loc_curve.png"))
        plt.savefig(os.path.join("%s/input-3D/visualize-loc-curve" % mode, list_det_txt[i].split(".")[0] + "_vis_size_curve.png"))
        plt.close()
    
    if vis_curve:
        plt.close()


    if vis_pos:
        plt.savefig(os.path.join("%s/input-3D/visualize-pos" % mode, list_det_txt[i].split(".")[0] + "_vispos.png"))
        plt.close()

# 记录三维尺寸和三维质心(定位)最后误差及精度
avg_size_error = total_size_error / MATCHED_NUM
avg_loc_error = total_loc_error / MATCHED_NUM

avg_size_precision = 1.0 - avg_size_error
avg_loc_precision = 1.0 - avg_loc_error

with open("./%s/input-3D/size_and_loc_precision.txt"%mode, "w") as f:
    f.write("Head: ".ljust(35) + "L".ljust(22) + "W".ljust(22) + "H".ljust(22) + "CX".ljust(22) + "CY".ljust(22) + "CZ".ljust(22) +"\n")
    for i in range(len(tp_sizes_dt)):
        f.write("TP_SIZES_LOCS_DT: " + str("\t".join(map("{:20}".format, tp_sizes_dt[i]))) + "\t" + str("\t".join(map("{:20}".format,tp_locs_dt[i]))) + "\n")
        f.write("TP_SIZES_LOCS_GT: " + str("\t".join(map("{:20}".format, tp_sizes_gt[i]))) + "\t" + str("\t".join(map("{:20}".format,tp_locs_gt[i]))) + "\n")
        f.write("TP_SIZES_LOCS_ERROR_PRECISION: " + "\t" + str(single_size_error[i]).zfill(15) + "\t\t\t" + str(single_loc_error[i]).zfill(15) + "\t\t\t" + str(1.0-single_size_error[i]).zfill(15) + "\t\t\t" + str(1.0-single_loc_error[i]).zfill(15) + "\n")
        f.write("LWH_ERROR/m: " + "\t" + str(l_error[i]).zfill(5) + "\t\t\t" + str(w_error[i]).zfill(5) + "\t\t\t" + str(h_error[i]).zfill(5) + "\n\n")
    f.write("Avg_l_error: " + str(np.mean(l_error)) + "\n")
    f.write("Avg_w_error: " + str(np.mean(w_error))+ "\n")
    f.write("Avg_h_error: " + str(np.mean(h_error))+ "\n")
    f.write("Avg_size_error: " + str(avg_size_error) + "\n")
    f.write("Avg_loc_error: " + str(avg_loc_error) + "\n")
    f.write("Avg_size_precision: " + str(avg_size_precision) + "\n")
    f.write("Avg_loc_precision: " + str(avg_loc_precision) + "\n")


print("Avg_l_error: " + str(np.mean(l_error)))
print("Avg_w_error: " + str(np.mean(w_error)))
print("Avg_h_error: " + str(np.mean(h_error)))

print("Avg_size_error: " + str(avg_size_error))
print("Avg_loc_error: " + str(avg_loc_error))

print("Avg_size_precision: " + str(avg_size_precision))
print("Avg_loc_precision: " + str(avg_loc_precision))






