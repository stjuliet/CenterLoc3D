# 无标注文件预测

import os
import shutil
from box_predict import Bbox3dPred
from PIL import Image
from tqdm import tqdm
import cv2 as cv

model = Bbox3dPred()

# 绝对路径
dataset_path_list = ["E:/PythonCodes/bbox3d_annotation_tools/session0_center_data", 
                     "E:/PythonCodes/bbox3d_annotation_tools/session0_right_data", 
                     "E:/PythonCodes/bbox3d_annotation_tools/session6_right_data"]

# 指定数据集图片类型
IMAGE_TYPES = [".jpg", ".png", ".jpeg"]

# 是否记录测试结果
record_result = True
# 是否保存测试结果图片
save_test_img = True

total_proc_time = 0

for dt_index, single_dataset_path_list in enumerate(dataset_path_list):
    file_list = os.listdir(single_dataset_path_list)
    # 标定文件路径
    calib_file_dir = os.path.join(dataset_path_list[dt_index], file_list[0])
    calib_list = os.listdir(calib_file_dir)
    calib_raw_path = os.path.join(calib_file_dir, calib_list[0])
    # 原始图像路径
    for file_name in tqdm(file_list):
        if not os.path.exists(os.path.join(single_dataset_path_list, file_name[:-4] + ".xml")):
            for img_type in IMAGE_TYPES:
                img_raw_path = os.path.join(single_dataset_path_list, file_name[:-4] + img_type)
                if os.path.exists(img_raw_path):
                    image = Image.open(img_raw_path)
                    image_id = file_name.split(".")[0]
                    r_image, r_heatmap, proc_time = model.detect_image(image, image_id, record_result, calib_raw_path)
                    total_proc_time += proc_time
                    if save_test_img:
                        r_image.save("./input-2D/images-optional/" + str(img_raw_path.split("\\")[-1]))
                        cv.imwrite("./input-2D/images-heatmap/" + str(img_raw_path.split("\\")[-1][:-4] + "_heatmap.png"), r_heatmap)
                else:
                    continue

print("Average FPS: ", round(len(test_file_path_list)/total_proc_time, 4))
print("batch predict without annotations finished!")

if record_result:
    print("results recorded!")

if save_test_img:
    print("images saved!")
