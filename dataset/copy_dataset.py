# copy annatation files to VOC folder
# trainval/test

import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

# absolute path
dataset_path_list = ["E:/PythonCodes/bbox3d_annotation_tools/session0_center_data",
                     "E:/PythonCodes/bbox3d_annotation_tools/session0_right_data",
                     "E:/PythonCodes/bbox3d_annotation_tools/session6_right_data",
                     "E:/PythonCodes/bbox3d_annotation_tools/real_scene_cam0",
                     "E:/PythonCodes/bbox3d_annotation_tools/real_scene_cam1"]

# relative path
voc_trainval_img_dir = "../DATAdevkit/DATA2021/JPEGImages"
voc_trainval_anno_dir = "../DATAdevkit/DATA2021/Annotations"
voc_trainval_calib_dir = "../DATAdevkit/DATA2021/Calib"

voc_test_img_dir = "../DATAdevkit/TESTDATA2021/JPEGImages"
voc_test_anno_dir = "../DATAdevkit/TESTDATA2021/Annotations"
voc_test_calib_dir = "../DATAdevkit/TESTDATA2021/Calib"

# start index of trainval
trainval_len_list = [2852*3, 3183*3, 1014*3, 10053, 8948]

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

for dt_index, single_dataset_path_list in tqdm(enumerate(dataset_path_list)):
    file_list = os.listdir(single_dataset_path_list)
    # copy calib files
    if "calib" in file_list:
        calib_file_dir = os.path.join(dataset_path_list[dt_index], "calib")
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
            if single_dataset_path_list.split("/")[-1] == "real_scene_cam0" or single_dataset_path_list.split("/")[-1] == "real_scene_cam1":
                # raw xml filename element revise
                raw_xml_tree = ET.parse(xml_raw_path)
                raw_root = raw_xml_tree.getroot()
                raw_filename = raw_xml_tree.find("filename").text
                new_base_filename_dir = raw_filename.replace("bg", single_dataset_path_list.split("/")[-1])
                raw_xml_tree.find("filename").text = new_base_filename_dir
                # new filename revise
                xml_new_path = os.path.join(voc_trainval_anno_dir, new_base_filename_dir.split("/")[-1][:-4] + ".xml")

                with open(xml_new_path, "w") as xml:
                    raw_xml_tree.write(xml_new_path, encoding="utf-8", xml_declaration=True)
            else:
                xml_new_path = os.path.join(voc_trainval_anno_dir, file_name)

            # copy trainval xml
            if not os.path.exists(xml_new_path):
                shutil.copy(xml_raw_path, xml_new_path)

            for img_type in IMAGE_TYPES:
                img_raw_path = os.path.join(single_dataset_path_list, file_name[:-4] + img_type)
                if os.path.exists(img_raw_path):
                    if single_dataset_path_list.split("/")[-1] == "real_scene_cam0" or single_dataset_path_list.split("/")[-1] == "real_scene_cam1":
                        img_new_path = os.path.join(voc_trainval_img_dir, new_base_filename_dir.split("/")[-1][:-4] + img_type)
                    else:
                        img_new_path = os.path.join(voc_trainval_img_dir, file_name[:-4] + img_type)
                    # copy trainval img
                    if not os.path.exists(img_new_path):
                        shutil.copy(img_raw_path, img_new_path)
    # test
    for file_name in file_list[trainval_len_list[dt_index]:len(file_list)]:
        if file_name.endswith(".xml"):
            # print("test: " + file_name)
            xml_raw_path = os.path.join(single_dataset_path_list, file_name)
            if single_dataset_path_list.split("/")[-1] == "real_scene_cam0" or single_dataset_path_list.split("/")[-1] == "real_scene_cam1":
                # raw xml filename element revise
                raw_xml_tree = ET.parse(xml_raw_path)
                raw_root = raw_xml_tree.getroot()
                raw_filename = raw_xml_tree.find("filename").text
                new_base_filename_dir = raw_filename.replace("bg", single_dataset_path_list.split("/")[-1])
                raw_xml_tree.find("filename").text = new_base_filename_dir
                # new filename revise
                xml_new_path = os.path.join(voc_test_anno_dir, new_base_filename_dir.split("/")[-1][:-4] + ".xml")

                with open(xml_new_path, "w") as xml:
                    raw_xml_tree.write(xml_new_path, encoding="utf-8", xml_declaration=True)
            else:
                xml_new_path = os.path.join(voc_test_anno_dir, file_name)
            # copy test xml
            if not os.path.exists(xml_new_path):
                shutil.copy(xml_raw_path, xml_new_path)

            for img_type in IMAGE_TYPES:
                img_raw_path = os.path.join(single_dataset_path_list, file_name[:-4] + img_type)
                if os.path.exists(img_raw_path):
                    if single_dataset_path_list.split("/")[-1] == "real_scene_cam0" or single_dataset_path_list.split("/")[-1] == "real_scene_cam1":
                        img_new_path = os.path.join(voc_test_img_dir, new_base_filename_dir.split("/")[-1][:-4] + img_type)
                    else:
                        img_new_path = os.path.join(voc_test_img_dir, file_name[:-4] + img_type)
                    # copy test img
                    if not os.path.exists(img_new_path):
                        shutil.copy(img_raw_path, img_new_path)

print("finish copy calib files, imgs and annotations to VOC folder!")
