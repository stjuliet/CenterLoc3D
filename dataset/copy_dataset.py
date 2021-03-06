# copy annatation files to VOC folder
# trainval/test

import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
random.seed(0)  # keep split results same

# absolute path
# dataset_path_list = ["E:/PythonCodes/bbox3d_annotation_tools/session0_center_data",
#                      "E:/PythonCodes/bbox3d_annotation_tools/session0_right_data",
#                      "E:/PythonCodes/bbox3d_annotation_tools/session6_right_data",
#                      "E:/PythonCodes/bbox3d_annotation_tools/real_scene_cam0",
#                      "E:/PythonCodes/bbox3d_annotation_tools/real_scene_cam1"]

dataset_path_list = ["E:/PythonCodes/bbox3d_annotation_tools/real_scene_cam0",
                     "E:/PythonCodes/bbox3d_annotation_tools/real_scene_cam1"]

# relative path
voc_trainval_img_dir = "../DATAdevkit/DATA2021/JPEGImages"
voc_trainval_anno_dir = "../DATAdevkit/DATA2021/Annotations"
voc_trainval_calib_dir = "../DATAdevkit/DATA2021/Calib"

voc_test_img_dir = "../DATAdevkit/TESTDATA2021/JPEGImages"
voc_test_anno_dir = "../DATAdevkit/TESTDATA2021/Annotations"
voc_test_calib_dir = "../DATAdevkit/TESTDATA2021/Calib"

# start index of trainval
# 2852*3, 3183*3, 1014*3,
trainval_len_list = [10053, 8948]

trainval_num_list = [3851, 3701]

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
    trainval_bmp_files = []

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

    # bmp files list
    for i in range(trainval_num_list[dt_index]):
        file_name = "bg_%06d" % i
        bmp_anno_file_path = os.path.join(single_dataset_path_list, file_name + "_drawbbox_result.bmp")
        if not os.path.exists(bmp_anno_file_path):
            trainval_bmp_files.append(file_name[:9])
    declude_xml_list_without_bmps = random.sample(trainval_bmp_files, int(len(trainval_bmp_files) * 0.9))

    # trainval
    for file_name in file_list[:trainval_len_list[dt_index]]:
        if file_name.endswith(".xml"):
            xml_raw_path = os.path.join(single_dataset_path_list, file_name)
            if single_dataset_path_list.split("/")[-1] == "real_scene_cam0" or single_dataset_path_list.split("/")[-1] == "real_scene_cam1":
                # if file_name[:-4] not in declude_xml_list_without_bmps:
                # raw xml filename element revise
                raw_xml_tree = ET.parse(xml_raw_path)
                raw_root = raw_xml_tree.getroot()
                raw_filename = raw_xml_tree.find("filename").text
                new_base_filename_dir = raw_filename.replace("bg", single_dataset_path_list.split("/")[-1])
                if new_base_filename_dir.endswith(".png"):
                    new_base_filename_dir = new_base_filename_dir.replace(".png", ".jpg")
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
                        # if file_name[:-4] not in declude_xml_list_without_bmps:
                        img_new_path = os.path.join(voc_trainval_img_dir, new_base_filename_dir.split("/")[-1][:-4] + ".jpg")
                    else:
                        img_new_path = os.path.join(voc_trainval_img_dir, file_name[:-4] + ".jpg")
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
                if new_base_filename_dir.endswith(".png"):
                    new_base_filename_dir = new_base_filename_dir.replace(".png", ".jpg")
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
                        img_new_path = os.path.join(voc_test_img_dir, new_base_filename_dir.split("/")[-1][:-4] + ".jpg")
                    else:
                        img_new_path = os.path.join(voc_test_img_dir, file_name[:-4] + ".jpg")
                    # copy test img
                    if not os.path.exists(img_new_path):
                        shutil.copy(img_raw_path, img_new_path)

print("finish copy calib files, imgs and annotations to VOC folder!")
