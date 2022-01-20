import os
import json
import cv2 as cv
import numpy as np

# dataset_path_list = ["E:/PythonCodes/bbox3d_annotation_tools/session0_center_data", 
#                      "E:/PythonCodes/bbox3d_annotation_tools/session0_right_data", 
#                      "E:/PythonCodes/bbox3d_annotation_tools/session6_right_data"]

raw_fps = 50.00
raw_frame_start = 5*60
frame_len = int(25.00 * 5*60)
img_root_dir = "E:/PythonCodes/bbox3d_annotation_tools/session0_right_data"
gt_pos_json_file = "gt_pos_json/system_dubska_bmvc14_session0_right.json"

start_frame_index = int(raw_frame_start*raw_fps)

with open(gt_pos_json_file, "r") as f:
    gt_pos_data_dict = json.load(f)

# 挑选5分钟时的gt 显示至对应的图像上
# frames id posX posY
list_cars = gt_pos_data_dict["cars"]

img_index = 0

all_list_frames = []
all_ids = []
all_pos_xs = []
all_pos_ys = []

indices = []

for single_car_dict in list_cars:
    list_frames = np.array(list(single_car_dict["frames"])).astype(np.int)
    id = np.array(int(single_car_dict["id"])).astype(np.int)
    list_pos_xs = np.array(list(single_car_dict["posX"])).astype(np.float)
    list_pos_ys = np.array(list(single_car_dict["posY"])).astype(np.float)
    all_list_frames.append(list_frames)
    all_ids.append(id)
    all_pos_xs.append(list_pos_xs)
    all_pos_ys.append(list_pos_ys)

np_list_frames = np.array(all_list_frames)
np_ids = np.array(all_ids)
np_pos_xs = np.array(all_pos_xs)
np_pos_ys = np.array(all_pos_ys)

for frm_idx in range(img_index, img_index + frame_len):
    indices.clear()
    img_path = os.path.join(img_root_dir, img_root_dir.split("/")[-1][:-4] + "%06d" % (frm_idx) + ".jpg")
    img = cv.imread(img_path)
    for i in range(len(np_list_frames)):
        for j in range(len(np_list_frames[i])):
            if start_frame_index == np_list_frames[i][j]:
                indices.append((i, j))
    for index in indices:
        x, y = index
        pos_x, pos_y = int(np_pos_xs[x][y]), int(np_pos_ys[x][y])
        cv.circle(img, (pos_x, pos_y), 3, (0, 0, 255), -1)
    cv.imshow("test", img)
    cv.waitKey(50)
    start_frame_index += 1

print("o")


    # start_frame_index

    # for frame_index in list_frames:
    #     img_path = os.path.join(img_root_dir, img_root_dir.split("/")[-1][:-4] + "%06d" % (img_index) + ".jpg")
    #     img = cv.imread(img_path)
    #     index = list_frames.index(frame_index)
    #     pos_x = int(list_pos_xs[index])
    #     pos_y = int(list_pos_ys[index])
    #     cv.circle(img, (pos_x, pos_y), 3, (0, 0, 255), -1)
    #     cv.imshow("test", img)
    #     cv.waitKey(50)


print(gt_pos_data_dict)
