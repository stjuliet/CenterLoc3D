# 将标注文件copy到VOC格式数据集下
import os
import shutil

dataset_path_list = ["E:\\PythonCodes\\bbox3d_annotation_tools\\session0_center_data", 
                     "E:\\PythonCodes\\bbox3d_annotation_tools\\session0_right_data", 
                     "D:\\3Dmark2\\session6_right_data"]
voc_img_dir = "E:\\PythonCodes\\bbox3d_recognition_v2\\DATAdevkit\\DATA2021\\JPEGImages"
voc_anno_dir = "E:\\PythonCodes\\bbox3d_recognition_v2\\DATAdevkit\\DATA2021\\Annotations"

if not os.path.exists(voc_img_dir):
    os.makedirs(voc_img_dir)
if not os.path.exists(voc_anno_dir):
    os.makedirs(voc_anno_dir)

for single_dataset_path_list in dataset_path_list:
    file_list = os.listdir(single_dataset_path_list)
    for file_name in file_list:
        if file_name.endswith(".xml"):
            xml_raw_path = single_dataset_path_list + "\\" + file_name
            xml_new_path = voc_anno_dir + "\\" + file_name
            # copy xml
            if not os.path.exists(xml_new_path):
                shutil.copy(xml_raw_path, xml_new_path)

            img_raw_path = single_dataset_path_list + "\\" + file_name[:-4] + ".jpg"
            img_new_path = voc_img_dir + "\\" + file_name[:-4] + ".jpg"
            # copy img
            if not os.path.exists(img_new_path):
                shutil.copy(img_raw_path, img_new_path)

print("finish copy imgs and annotations!")
