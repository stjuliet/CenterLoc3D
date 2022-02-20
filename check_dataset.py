import os
import xml.etree.ElementTree as ET

train_data_dir = "DATAdevkit/DATA2021/Annotations"

train_data_list = os.listdir(train_data_dir)

train_data_anno_dict = {}

count = 0

for i, anno in enumerate(train_data_list):
    anno_path = os.path.join(train_data_dir, anno)
    if os.path.exists(anno_path):
        xml_tree = ET.parse(anno_path)
        xml_root = xml_tree.getroot()
        per_frame_valid_check_count = 0
        anno_num = len(xml_root.findall("object"))

        if train_data_list[i].split("_")[2] == "cam0" and anno_num == 2:
            # print(anno)
            count += 1
        elif anno_num == 1:
            # print(anno)
            count += 1

        if i > 0:
            if (train_data_list[i].split("_")[0][:7] == "session" and train_data_list[i].split("_")[:2] != train_data_list[i-1].split("_")[:2]) or (train_data_list[i].split("_")[0][:4] == "real" and train_data_list[i].split("_")[:3] != train_data_list[i - 1].split("_")[:3]):
                train_data_anno_dict.update({train_data_list[i]: count})
                count = 0
            if i == len(train_data_list)-1:
                train_data_anno_dict.update({train_data_list[i]: count})
                count = 0

print(train_data_anno_dict)
