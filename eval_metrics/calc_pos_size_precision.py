import os
import numpy as np
from utils.utils import basic_3diou
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

mode = "test"

record_txt = False  # do not change

MINOVERLAP = 0.7
MATCHED_NUM = 0

# scene nums
SCENE_NUM = 0

# postion visualization (set once to True only)
vis_pos = True
vis_curve = False

det_txt_dir = "../%s/input-3D/detection-results" % mode
gt_txt_dir = "../%s/input-3D/ground-truth" % mode

if not os.path.exists("../%s/input-3D/visualize-pos-%s" % (mode, str(MINOVERLAP))):
    os.makedirs("../%s/input-3D/visualize-pos-%s" % (mode, str(MINOVERLAP)))

if not os.path.exists("../%s/input-3D/visualize-loc-curve" % mode):
    os.makedirs("../%s/input-3D/visualize-loc-curve" % mode)

# save results to txt files
if not os.path.exists("../%s/input-3D/size_and_loc_precision-%s.txt"%(mode, str(MINOVERLAP))):
    record_txt = True
    f = open("../%s/input-3D/size_and_loc_precision-%s.txt"%(mode, str(MINOVERLAP)), "w")
    f.write("Head: ".ljust(35) + "L".ljust(22) + "W".ljust(22) + "H".ljust(22) + "CX".ljust(22) + "CY".ljust(22) + "CZ".ljust(22) +"\n")

list_det_txt = sorted(os.listdir(det_txt_dir))
list_gt_txt = sorted(os.listdir(gt_txt_dir))

match_img_ids = []

total_size_precision, total_loc_precision = 0.0, 0.0
single_size_precision, single_loc_precision = [], []

tp_sizes_dt, tp_locs_dt, tp_sizes_gt, tp_locs_gt = [], [], [], []  # save tp gt and pred

# error of xc,yc,zc,l,w,h in a single scene, m
s_xc_error, s_yc_error, s_zc_error, s_loc_error = [], [], [], []
s_l_error, s_w_error, s_h_error, s_size_error = [], [], [], []
s_gt_yc = []

# error of xc,yc,zc,l,w,h in all scenes, m
all_s_xc_error, all_s_yc_error, all_s_zc_error, all_s_loc_error = [], [], [], []
all_s_l_error, all_s_w_error, all_s_h_error, all_s_size_error = [], [], [], []


font_legend = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 40,
        }

font_label = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 40,
        }

# effective distance for different scenes
valid_pers = [[120, 25], [120, 25], [60, 15]]

for i in tqdm(range(len(list_det_txt))):  # img files
    # save a single curve for each scene!
    if vis_curve:
        fig, ax = plt.subplots(figsize=(20, 15), dpi=100)
        plt.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()

        [label.set_fontname('Times New Roman') for label in labels]

        plt.xlabel("ground truth distance/m", font_label)
        plt.ylabel("error/m", font_label)

        # tick direction
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
    
    # save a single scatter for each img file!
    if vis_pos:
        fig, ax = plt.subplots(figsize=(15, 20), dpi=100)
        plt.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()

        [label.set_fontname('Times New Roman') for label in labels]

        plt.xlabel("x/m", font_label)
        plt.ylabel("y/m", font_label)

        # tick direction
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.grid(True, linestyle='--', alpha=0.5)

    f_det = open(os.path.join(det_txt_dir, list_det_txt[i]), "r")
    list_dets = f_det.readlines()
    f_gt = open(os.path.join(gt_txt_dir, list_gt_txt[i]), "r")
    list_gts = f_gt.readlines()

    if i > 0 and list_det_txt[i].split("_")[:2] != list_det_txt[i-1].split("_")[:2]:
        SCENE_NUM += 1
    pers_r = int(valid_pers[SCENE_NUM][0]) * 1000 / 2
    pers_pr = int(valid_pers[SCENE_NUM][1]) * 1000 / 2
    # if list_det_txt[i].split("_")[0] == "session0":
    #     valid_pers = 150.0*1000 /2  # effective distance for different scenes
    # elif list_det_txt[i].split("_")[0] == "session6":
    #     valid_pers = 100.0*1000 /2  # effective distance for different scenes

    for line_det in list_dets:  # dets
        for line_gt in list_gts:  # gts
            det = line_det.split()
            gt = line_gt.split()

            bb_det = np.array(det[2:26]).astype(np.float)
            bb_gt = np.array(gt[1:25]).astype(np.float)

            if vis_pos:
                # draw whenever matched
                cx_gt_unmatched, cy_gt_unmatched, cz_gt_unmatched = np.array(gt[28:31]).astype(np.float)
                cx_det_unmatched, cy_det_unmatched, cz_det_unmatched = np.array(det[29:32]).astype(np.float)
                det_plot = plt.scatter(cx_det_unmatched/1000, cy_det_unmatched/1000, s=800, c="b", marker="s", label="detection")   # det
                gt_plot = plt.scatter(cx_gt_unmatched/1000, cy_gt_unmatched/1000, s=1000, c="r", marker="*", label="ground truth")   # gt

                plt.legend(handles=[gt_plot, det_plot], prop=font_legend)
            
            ov = basic_3diou(bb_det, bb_gt)  # 3d iou

            if ov > MINOVERLAP:  # marked as matched
                MATCHED_NUM += 1
                # save matched img ids
                match_img_ids.append(list_det_txt[i])

                l_dt, w_dt, h_dt = np.array(det[26:29]).astype(np.float)
                cx_dt, cy_dt, cz_dt = np.array(det[29:32]).astype(np.float)
                l_gt, w_gt, h_gt = np.array(gt[25:28]).astype(np.float)
                cx_gt, cy_gt, cz_gt = np.array(gt[28:31]).astype(np.float)
                # save tp gt and pred to txt
                tp_sizes_dt.append([l_dt, w_dt, h_dt])
                tp_sizes_gt.append([l_gt, w_gt, h_gt])
                tp_locs_dt.append([cx_dt, cy_dt, cz_dt])
                tp_locs_gt.append([cx_gt, cy_gt, cz_gt])

                # save single img precision to txt
                tmp_size_error = abs(l_dt-l_gt)/l_gt + abs(w_dt-w_gt)/w_gt + abs(h_dt-h_gt)/h_gt
                tmp_loc_error = abs(cx_dt-cx_gt)/pers_pr+abs(cy_dt-cy_gt)/pers_r
                single_size_precision.append(1.0 - tmp_size_error)
                single_loc_precision.append(1.0 - tmp_loc_error)
                # total error
                total_loc_precision += (1.0 - tmp_loc_error)
                total_size_precision += (1.0 - tmp_size_error)

                # gt pos and error (only for matched)
                s_xc_error.append(abs(cx_dt/1000-cx_gt/1000))
                s_yc_error.append(abs(cy_dt/1000-cy_gt/1000))
                s_zc_error.append(abs(cz_dt/1000-cz_gt/1000))
                s_loc_error.append(abs(cx_dt/1000-cx_gt/1000)+abs(cy_dt/1000-cy_gt/1000)+abs(cz_dt/1000-cz_gt/1000))
                s_l_error.append(abs(l_dt-l_gt))
                s_w_error.append(abs(w_dt-w_gt))
                s_h_error.append(abs(h_dt-h_gt))
                s_size_error.append(abs(l_dt-l_gt)+abs(w_dt-w_gt)+abs(h_dt-h_gt))
                s_gt_yc.append(cy_gt/1000)

                all_s_l_error.append(abs(l_dt-l_gt))
                all_s_w_error.append(abs(w_dt-w_gt))
                all_s_h_error.append(abs(h_dt-h_gt))

                all_s_xc_error.append(abs(cx_dt/1000-cx_gt/1000))
                all_s_yc_error.append(abs(cy_dt/1000-cy_gt/1000))
                all_s_zc_error.append(abs(cz_dt/1000-cz_gt/1000))

                all_s_loc_error.append(abs(cx_dt/1000-cx_gt/1000)+abs(cy_dt/1000-cy_gt/1000)+abs(cz_dt/1000-cz_gt/1000))
                all_s_size_error.append(abs(l_dt-l_gt)+abs(w_dt-w_gt)+abs(h_dt-h_gt))

    if vis_curve and i > 0 and list_det_txt[i].split("_")[:2] != list_det_txt[i-1].split("_")[:2]:
        if SCENE_NUM == 1:
            plt.yticks(np.arange(-0.1,1.6,0.3))
            plt.ylim(-0.1,1.6)
        if SCENE_NUM == 2:
            plt.yticks(np.arange(-0.5,3.5,0.5))
            plt.ylim(-0.5,3.5)
            
        xc_plot = plt.plot(sorted(s_gt_yc), sorted(s_xc_error), color="green", linewidth=2, label="X error")
        yc_plot = plt.plot(sorted(s_gt_yc), sorted(s_yc_error), color="red", linewidth=2, label="Y error")
        zc_plot = plt.plot(sorted(s_gt_yc), sorted(s_zc_error), color="blue", linewidth=2, label="Z error")
        loc_plot = plt.plot(sorted(s_gt_yc), sorted(s_loc_error), color="orange", linewidth=2, label="total error")

        # # 尺寸刻度
        # if SCENE_NUM == 1:
        #     plt.yticks(np.arange(-0.2,1.2,0.2))
        # if SCENE_NUM == 2:
        #     plt.yticks(np.arange(-0.2,1.2,0.2))
        # l_plot = plt.plot(sorted(s_gt_yc), sorted(s_l_error), color="green", linewidth=2, label="l error")
        # w_plot = plt.plot(sorted(s_gt_yc), sorted(s_w_error), color="red", linewidth=2, label="w error")
        # h_plot = plt.plot(sorted(s_gt_yc), sorted(s_h_error), color="blue", linewidth=2, label="h error")
        # size_plot = plt.plot(sorted(s_gt_yc), sorted(s_size_error), color="orange", linewidth=2, label="total error")
        
        if record_txt:
            f.write("Scene: " + str(i) + "\n")
            f.write("Avg_l_error_single_scene/m: " + str(np.mean(s_l_error)) + "\n")
            f.write("Avg_w_error_single_scene/m: " + str(np.mean(s_w_error))+ "\n")
            f.write("Avg_h_error_single_scene/m: " + str(np.mean(s_h_error))+ "\n")
            f.write("Avg_xc_error_single_scene/m: " + str(np.mean(s_xc_error)) + "\n")
            f.write("Avg_yc_error_single_scene/m: " + str(np.mean(s_yc_error))+ "\n")
            f.write("Avg_zc_error_single_scene/m: " + str(np.mean(s_zc_error))+ "\n")
            f.write("Avg_size_error_single_scene/m: " + str(np.mean(s_size_error))+ "\n")
            f.write("Avg_loc_error_single_scene/m: " + str(np.mean(s_loc_error))+ "\n")

        plt.legend(loc="best", prop=font_legend)

        plt.savefig(os.path.join("../%s/input-3D/visualize-loc-curve" % mode, list_det_txt[i].split(".")[0] + "_vis_loc_curve-%s.eps"%str(MINOVERLAP)), format="eps")
        # plt.savefig(os.path.join("%s/input-3D/visualize-loc-curve" % mode, list_det_txt[i].split(".")[0] + "_vis_size_curve-%s.eps"%str(MINOVERLAP)), format="eps")
        plt.close()

        # clear all variables after figure saved
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

        plt.yticks(np.arange(-0.2,1,0.2))
        plt.ylim(-0.2, 1.0)
        xc_plot = plt.plot(sorted(s_gt_yc), sorted(s_xc_error), color="green", linewidth=2, label="X error")
        yc_plot = plt.plot(sorted(s_gt_yc), sorted(s_yc_error), color="red", linewidth=2, label="Y error")
        zc_plot = plt.plot(sorted(s_gt_yc), sorted(s_zc_error), color="blue", linewidth=2, label="Z error")
        loc_plot = plt.plot(sorted(s_gt_yc), sorted(s_loc_error), color="orange", linewidth=2, label="total error")

        # plt.yticks(np.arange(-0.2,1.2,0.2))
        # l_plot = plt.plot(sorted(s_gt_yc), sorted(s_l_error), color="green", linewidth=2, label="l error")
        # w_plot = plt.plot(sorted(s_gt_yc), sorted(s_w_error), color="red", linewidth=2, label="w error")
        # h_plot = plt.plot(sorted(s_gt_yc), sorted(s_h_error), color="blue", linewidth=2, label="h error")
        # size_plot = plt.plot(sorted(s_gt_yc), sorted(s_size_error), color="orange", linewidth=2, label="total error")
        if record_txt:
            f.write("Scene: " + str(i) + "\n")
            f.write("Avg_l_error_single_scene/m: " + str(np.mean(s_l_error)) + "\n")
            f.write("Avg_w_error_single_scene/m: " + str(np.mean(s_w_error))+ "\n")
            f.write("Avg_h_error_single_scene/m: " + str(np.mean(s_h_error))+ "\n")
            f.write("Avg_xc_error_single_scene/m: " + str(np.mean(s_xc_error)) + "\n")
            f.write("Avg_yc_error_single_scene/m: " + str(np.mean(s_yc_error))+ "\n")
            f.write("Avg_zc_error_single_scene/m: " + str(np.mean(s_zc_error))+ "\n")
            f.write("Avg_size_error_single_scene/m: " + str(np.mean(s_size_error))+ "\n")
            f.write("Avg_loc_error_single_scene/m: " + str(np.mean(s_loc_error))+ "\n")

        plt.legend(loc="best", prop=font_legend)

        plt.savefig(os.path.join("../%s/input-3D/visualize-loc-curve" % mode, list_det_txt[i].split(".")[0] + "_vis_loc_curve-%s.eps"%str(MINOVERLAP)), format="eps")
        # plt.savefig(os.path.join("%s/input-3D/visualize-loc-curve" % mode, list_det_txt[i].split(".")[0] + "_vis_size_curve-%s.eps"%str(MINOVERLAP)), format="eps")
        plt.close()
    
    if vis_curve:
        plt.close()

    if vis_pos:
        plt.savefig(os.path.join("../%s/input-3D/visualize-pos-%s" % (mode, str(MINOVERLAP)), list_det_txt[i].split(".")[0] + "_vispos-%s.eps"%str(MINOVERLAP)), format="eps")
        plt.close()

# save final error and precision of 3d size and centroid
avg_size_precision = total_size_precision / MATCHED_NUM
avg_loc_precision = total_loc_precision / MATCHED_NUM

if record_txt:
    for i in range(len(tp_sizes_dt)):  # single matched data
        f.write("IMG_ID: " + match_img_ids[i] + "\n")
        f.write("TP_SIZES_LOCS_DT: " + str("\t".join(map("{:20}".format, tp_sizes_dt[i]))) + "\t" + str("\t".join(map("{:20}".format,tp_locs_dt[i]))) + "\n")
        f.write("TP_SIZES_LOCS_GT: " + str("\t".join(map("{:20}".format, tp_sizes_gt[i]))) + "\t" + str("\t".join(map("{:20}".format,tp_locs_gt[i]))) + "\n")
        f.write("TP_SIZES_LOCS_PRECISION: " + "\t" + str(single_size_precision[i]).zfill(15) + "\t\t\t" + str(single_loc_precision[i]).zfill(15) + "\n")
        f.write("LWH_ERROR/m: " + "\t" + str(all_s_l_error[i]).zfill(5) + "\t\t\t" + str(all_s_w_error[i]).zfill(5) + "\t\t\t" + str(all_s_h_error[i]).zfill(5) + "\n")
        f.write("XYZ_ERROR/m: " + "\t" + str(all_s_xc_error[i]).zfill(5) + "\t\t\t" + str(all_s_yc_error[i]).zfill(5) + "\t\t\t" + str(all_s_zc_error[i]).zfill(5) + "\n\n")
    f.write("Avg_size_precision: " + str(avg_size_precision) + "\n")
    f.write("Avg_loc_precision: " + str(avg_loc_precision) + "\n")

    f.write("Avg_l_error/m: " + str(np.mean(all_s_l_error))+ "\n")
    f.write("Avg_w_error/m: " + str(np.mean(all_s_w_error))+ "\n")
    f.write("Avg_h_error/m: " + str(np.mean(all_s_h_error))+ "\n")
    f.write("Avg_xc_error/m: " + str(np.mean(all_s_xc_error))+ "\n")
    f.write("Avg_yc_error/m: " + str(np.mean(all_s_yc_error))+ "\n")
    f.write("Avg_zc_error/m: " + str(np.mean(all_s_zc_error))+ "\n")
    f.write("Avg_size_error/m: " + str(np.mean(all_s_size_error))+ "\n")
    f.write("Avg_loc_error/m: " + str(np.mean(all_s_loc_error))+ "\n")

    f.close()
