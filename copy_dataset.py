# 将标注文件copy到VOC格式数据集下
import os
import shutil

dataset_path_list = ["E:\\PythonCodes\\bbox3d_annotation_tools\\session0_center_data"]
voc_img_dir = "E:\\PythonCodes\\bbox3d_recognition_v2\\DATAdevkit\\DATA2021\\JPEGImages"
voc_anno_dir = "E:\\PythonCodes\\bbox3d_recognition_v2\\DATAdevkit\\DATA2021\\Annotations"

for single_dataset_path_list in dataset_path_list:
    file_list = os.listdir(single_dataset_path_list)
    for file_name in file_list:
        if file_name.endswith(".xml"):
            xml_raw_path = single_dataset_path_list + "\\" + file_name
            xml_new_path = voc_anno_dir + "\\" + file_name
            # copy xml
            shutil.copy(xml_raw_path, xml_new_path)

            img_raw_path = single_dataset_path_list + "\\" + file_name[:-4] + ".jpg"
            img_new_path = voc_img_dir + "\\" + file_name[:-4] + ".jpg"
            # copy img
            shutil.copy(img_raw_path, img_new_path)
