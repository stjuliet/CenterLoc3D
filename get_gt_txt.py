# 获取测试集的ground-truth

import os
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm


# 获得类
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

image_ids = open('./DATAdevkit/DATA2021/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in tqdm(image_ids):
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse("./DATAdevkit/DATA2021/Annotations/"+image_id+".xml").getroot()
        for obj in root.findall('object'):
            # 读取类型、视角及顶点信息
            #  车辆类型(0，1，2) (str)
            veh_type_data = obj.find('type').text

            #  车辆三维框顶点坐标(int)
            veh_vertex_data = [] # for each box (8 vertex)
            box_2dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex2d').text)
            for x in box_2dvertex:
                veh_vertex_data += [int(item) for item in x.split(", ")]  # 合并为[x1,y1,x2,y2,...,x8,y8]的形式

            #  视角(left, right) (str)
            veh_view_data = obj.find('perspective').text

            if veh_view_data == "right":  # right perspective  (7,1)
                left, top, right, bottom = veh_vertex_data[14], veh_vertex_data[15], veh_vertex_data[2], veh_vertex_data[3]
                new_f.write("%s %s %s %s %s\n" % (str(veh_type_data), str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
            else:  # left perspective  (1x,6y,7x,0y)
                left, top, right, bottom = veh_vertex_data[2], veh_vertex_data[13], veh_vertex_data[14], veh_vertex_data[1]
                new_f.write("%s %s %s %s %s\n" % (str(veh_type_data), str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
