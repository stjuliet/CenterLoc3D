# get ground-truth of val/test

import os
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils.utils import *

mode = "test"  # val/test


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


image_ids = open('../DATAdevkit/DATA2021/ImageSets/Main/%s.txt'%mode).read().strip().split()
gt_annos = open('../dataset/DATA2021_%s.txt'%mode).readlines()

# 2D
if not os.path.exists("../%s/input-2D"%mode):
    os.makedirs("../%s/input-2D"%mode)
if not os.path.exists("../%s/input-2D/ground-truth"%mode):
    os.makedirs("../%s/input-2D/ground-truth"%mode)

# 3D
if not os.path.exists("../%s/input-3D"%mode):
    os.makedirs("../%s/input-3D"%mode)
if not os.path.exists("../%s/input-3D/ground-truth"%mode):
    os.makedirs("../%s/input-3D/ground-truth"%mode)

# gt for draw
if not os.path.exists("../%s/gt-for-draw"%mode):
    os.makedirs("../%s/gt-for-draw"%mode)


ct = 0
for image_id in tqdm(image_ids):
    with open("../%s/input-2D/ground-truth/"%mode+image_id+".txt", "w") as f_2d:
        with open("../%s/input-3D/ground-truth/"%mode+image_id+".txt", "w") as f_3d:
            with open("../%s/gt-for-draw/"%mode+image_id+"_vt2d.txt", "w") as f_loc_size_gt:
                if mode == "test":
                    root_dir = "../DATAdevkit/TESTDATA2021/Annotations/"
                else:
                    root_dir = "../DATAdevkit/DATA2021/Annotations/"
                root = ET.parse(root_dir + image_id + ".xml").getroot()
                width = int(root.find("size").find("width").text)
                height = int(root.find("size").find("height").text)
                calib_xml_path = gt_annos[ct].split(" ")[1]
                calib_matrix = read_calib_params(calib_xml_path, width, height)
                ct += 1
                for obj in root.findall('object'):
                    #  get cls, view, vertex
                    #  vehicle type (0，1，2) (str)
                    veh_type_data = obj.find('type').text

                    #  vertex of 3d box in img (int)  --- 2d
                    veh_vertex_data_2d = []  # for each box (8 vertex)
                    box_2dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex2d').text)
                    for x in box_2dvertex:
                        veh_vertex_data_2d += [int(item) for item in x.split(", ")]  # [x1,y1,x2,y2,...,x8,y8]

                    #  vertex of 3d box in world (int)  --- 3d
                    veh_vertex_data_3d = []  # for each box (8 vertex)
                    box_3dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex3d').text)
                    for x in box_3dvertex:
                        veh_vertex_data_3d += [int(float(item)) for item in x.split(", ")]  # [x1,y1,z1,x2,y2,z2,...,x8,y8,z8]

                    #  vehicle size
                    veh_physical_size = obj.find('veh_size').text.split(" ")
                    veh_physical_size = list(map(float, veh_physical_size))

                    #  vehicle centroid in img
                    veh_proj_loc = obj.find('veh_loc_2d').text.split(" ")
                    veh_proj_loc = list(map(int, veh_proj_loc))

                    #  view (left, right) (str)
                    veh_view_data = obj.find('perspective').text

                    if veh_view_data == "right":  # right perspective  (7,1)
                        left, top, right, bottom = veh_vertex_data_2d[14], veh_vertex_data_2d[15], veh_vertex_data_2d[2], veh_vertex_data_2d[3]
                        f_2d.write("%s %s %s %s %s\n" % (str(veh_type_data), str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
                    else:  # left perspective  (1x,6y,7x,0y)
                        left, top, right, bottom = veh_vertex_data_2d[2], veh_vertex_data_2d[13], veh_vertex_data_2d[14], veh_vertex_data_2d[1]
                        f_2d.write("%s %s %s %s %s\n" % (str(veh_type_data), str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

                    line_3d, line_size, line_ct_loc = "", "", ""
                    for i in veh_vertex_data_3d:
                        line_3d += " " + str(i)
                    for i in veh_physical_size:
                        line_size += " " + str(i)
                    # calculate 3d centroid in world
                    cx_3d, cy_3d, cz_3d = RDUVtoXYZ(calib_matrix, veh_proj_loc[0], veh_proj_loc[1], 1000*veh_physical_size[2]/2)
                    for i in (cx_3d, cy_3d, cz_3d):
                        line_ct_loc += " " + str(i)
                    f_3d.write("%s %s %s %s\n" % (str(veh_type_data), str(line_3d.strip()), str(line_size.strip()), str(line_ct_loc.strip())))

                    # save gt 3d box and centroid for drawing
                    line_vt2d = ""
                    for i in veh_vertex_data_2d:
                        line_vt2d += " " + str(i)
                    for i in veh_proj_loc:
                        line_vt2d += " " + str(i)
                    f_loc_size_gt.write("%s\n" % (str(line_vt2d.strip())))


print("finish getting gt files!")
