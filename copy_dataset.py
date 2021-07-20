# 将标注文件copy到VOC格式数据集下
# 区分训练验证集和测试集

import os
import shutil

# 绝对路径
dataset_path_list = ["E:/PythonCodes/bbox3d_annotation_tools/session0_center_data", 
                     "E:/PythonCodes/bbox3d_annotation_tools/session0_right_data", 
                     "E:/PythonCodes/bbox3d_annotation_tools/session6_right_data"]

# 相对路径
voc_trainval_img_dir = "DATAdevkit/DATA2021/JPEGImages"
voc_trainval_anno_dir = "DATAdevkit/DATA2021/Annotations"
voc_trainval_calib_dir = "DATAdevkit/DATA2021/Calib"

voc_test_img_dir = "DATAdevkit/TESTDATA2021/JPEGImages"
voc_test_anno_dir = "DATAdevkit/TESTDATA2021/Annotations"
voc_test_calib_dir = "DATAdevkit/TESTDATA2021/Calib"

# 指定训练验证集开始位置
trainval_len_list = [2852*3, 3183*3, 1014*3]

# 指定数据集图片类型
IMAGE_TYPES = [".jpg", ".png", ".jpeg"]


if not os.path.exists(voc_trainval_img_dir):
    os.makedirs(voc_trainval_img_dir)
if not os.path.exists(voc_trainval_anno_dir):
    os.makedirs(voc_trainval_anno_dir)
if not os.path.exists(voc_test_img_dir):
    os.makedirs(voc_test_img_dir)
if not os.path.exists(voc_test_anno_dir):
    os.makedirs(voc_test_anno_dir)
if not os.path.exists(voc_trainval_calib_dir):
    os.makedirs(voc_trainval_calib_dir)
if not os.path.exists(voc_test_calib_dir):
    os.makedirs(voc_test_calib_dir)


for dt_index, single_dataset_path_list in enumerate(dataset_path_list):
    file_list = os.listdir(single_dataset_path_list)
    # copy calib files
    calib_file_dir = os.path.join(dataset_path_list[dt_index], file_list[0])
    calib_list = os.listdir(calib_file_dir)

    calib_raw_path = os.path.join(calib_file_dir, calib_list[0])
    calib_trainval_new_path = os.path.join(voc_trainval_calib_dir, calib_list[0])
    calib_test_new_path = os.path.join(voc_test_calib_dir, calib_list[0])

    if not os.path.exists(calib_trainval_new_path):
        shutil.copy(calib_raw_path, calib_trainval_new_path)
    if not os.path.exists(calib_test_new_path):
        shutil.copy(calib_raw_path, calib_test_new_path)

    # trainval
    for file_name in file_list[:trainval_len_list[dt_index]]:
        if file_name.endswith(".xml"):
            # print("trainval: " + file_name)
            xml_raw_path = os.path.join(single_dataset_path_list, file_name)
            xml_new_path = os.path.join(voc_trainval_anno_dir, file_name)
            # copy trainval xml
            if not os.path.exists(xml_new_path):
                shutil.copy(xml_raw_path, xml_new_path)

            for img_type in IMAGE_TYPES:
                img_raw_path = os.path.join(single_dataset_path_list, file_name[:-4] + img_type)
                img_new_path = os.path.join(voc_trainval_img_dir, file_name[:-4] + img_type)
                if os.path.exists(img_raw_path):
                    # copy trainval img
                    if not os.path.exists(img_new_path):
                        shutil.copy(img_raw_path, img_new_path)
    # test
    for file_name in file_list[trainval_len_list[dt_index]:len(file_list)]:
        if file_name.endswith(".xml"):
            # print("test: " + file_name)
            xml_raw_path = os.path.join(single_dataset_path_list, file_name)
            xml_new_path = os.path.join(voc_test_anno_dir, file_name)
            # copy test xml
            if not os.path.exists(xml_new_path):
                shutil.copy(xml_raw_path, xml_new_path)

            for img_type in IMAGE_TYPES:
                img_raw_path = os.path.join(single_dataset_path_list, file_name[:-4] + img_type)
                img_new_path = os.path.join(voc_test_img_dir, file_name[:-4] + img_type)
                if os.path.exists(img_raw_path):
                    # copy test img
                    if not os.path.exists(img_new_path):
                        shutil.copy(img_raw_path, img_new_path)


print("finish copy calib files, imgs and annotations!")
