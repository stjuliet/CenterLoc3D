# 将标注文件copy到VOC格式数据集下
# 区分训练验证集和测试集

import os
import shutil

dataset_path_list = ["E:\\PythonCodes\\bbox3d_annotation_tools\\session0_center_data", 
                     "E:\\PythonCodes\\bbox3d_annotation_tools\\session0_right_data", 
                     "D:\\3Dmark2\\session6_right_data"]

voc_trainval_img_dir = "E:\\PythonCodes\\bbox3d_recognition_v2\\DATAdevkit\\DATA2021\\JPEGImages"
voc_trainval_anno_dir = "E:\\PythonCodes\\bbox3d_recognition_v2\\DATAdevkit\\DATA2021\\Annotations"

voc_test_img_dir = "E:\\PythonCodes\\bbox3d_recognition_v2\\DATAdevkit\\TESTDATA2021\\JPEGImages"
voc_test_anno_dir = "E:\\PythonCodes\\bbox3d_recognition_v2\\DATAdevkit\\TESTDATA2021\\Annotations"

# 指定训练验证集开始位置
trainval_len_list = [2852*3, 3183*3, 117*3]


if not os.path.exists(voc_trainval_img_dir):
    os.makedirs(voc_trainval_img_dir)
if not os.path.exists(voc_trainval_anno_dir):
    os.makedirs(voc_trainval_anno_dir)
if not os.path.exists(voc_test_img_dir):
    os.makedirs(voc_test_img_dir)
if not os.path.exists(voc_test_anno_dir):
    os.makedirs(voc_test_anno_dir)


for dt_index, single_dataset_path_list in enumerate(dataset_path_list):
    file_list = os.listdir(single_dataset_path_list)

    # trainval
    for file_name in file_list[:trainval_len_list[dt_index]]:
        if file_name.endswith(".xml"):
            # print("trainval: " + file_name)
            xml_raw_path = single_dataset_path_list + "\\" + file_name
            xml_new_path = voc_trainval_anno_dir + "\\" + file_name
            # copy trainval xml
            if not os.path.exists(xml_new_path):
                shutil.copy(xml_raw_path, xml_new_path)

            img_raw_path = single_dataset_path_list + "\\" + file_name[:-4] + ".jpg"
            img_new_path = voc_trainval_img_dir + "\\" + file_name[:-4] + ".jpg"
            # copy trainval img
            if not os.path.exists(img_new_path):
                shutil.copy(img_raw_path, img_new_path)
    # test
    for file_name in file_list[trainval_len_list[dt_index]:]:
        if file_name.endswith(".xml"):
            # print("test: " + file_name)
            xml_raw_path = single_dataset_path_list + "\\" + file_name
            xml_new_path = voc_test_anno_dir + "\\" + file_name
            # copy test xml
            if not os.path.exists(xml_new_path):
                shutil.copy(xml_raw_path, xml_new_path)

            img_raw_path = single_dataset_path_list + "\\" + file_name[:-4] + ".jpg"
            img_new_path = voc_test_img_dir + "\\" + file_name[:-4] + ".jpg"
            # copy test img
            if not os.path.exists(img_new_path):
                shutil.copy(img_raw_path, img_new_path)


print("finish copy imgs and annotations!")
