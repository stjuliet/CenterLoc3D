# generate train script
import xml.etree.ElementTree as ET
from os import getcwd
import os
import re

# change with dataset
sets = [('DATA2021', 'train'), ('DATA2021', 'val'), ('DATA2021', 'test')]
classes = ["Car", "Truck", "Bus"]


def convert_annotation(year, image_id, list_file):
    in_file = open('DATAdevkit/%s/Annotations/%s.xml'%(year, image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    # calib xml path (only reserve file name)
    calib_xml_path = str(tree.find('calibfile').text).split("/")[-1]

    # box
    for idx, obj in enumerate(root.iter('object')):
        # 1、2D box [left, top, width, height] (int)
        bbox2d_data = obj.find('bbox2d').text.split()
        # 2、vehicle type (0，1，2) (str)
        veh_type_data = obj.find('type').text
        if veh_type_data not in classes:
            continue
        veh_cls_id = classes.index(veh_type_data)
        # 3、centroid in img (int)
        veh_centre_data = obj.find('veh_loc_2d').text.split()
        # 4、vertex of 3D box in img (int)
        veh_vertex_data = []  # for each box (8 vertex)
        box_2dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex2d').text)
        for x in box_2dvertex:
            veh_vertex_data += [int(item) for item in x.split(", ")]  # 合并为[x1,y1,x2,y2,...,x8,y8]的形式
        # 5、vehicle size (float, m)
        veh_size_data = obj.find('veh_size').text.split()
        # 6、vehicle view (left, right) (int)
        veh_view_data = obj.find('perspective').text
        if veh_view_data == 'right':
            veh_view_data = 1
        else:
            veh_view_data = 0
        # 7、vehicle base point (int)
        veh_base_point_data = obj.find('base_point').text.split()

        # save data for each object
        # line
        # file_path (outer write)
        # calib_xml_path left,top,width,height,cls_id,cx1,cy1,u0,v0,...,u7,v7,v_l,v_w,v_h,pers,bpx1,bpx2  (29 items)
        if idx == 0:
            if image_set == "test":
                year = "TESTDATA2021"
            list_file.write(" " + os.path.join('DATAdevkit/%s/Calib'%(year), calib_xml_path))
        single_line = " " + ",".join([box for box in bbox2d_data]) + "," + str(veh_cls_id) + "," + \
                        ",".join([centre for centre in veh_centre_data]) + "," + ",".join([str(vertex) for vertex in veh_vertex_data]) + "," + \
                            ",".join([size for size in veh_size_data]) + "," + str(veh_view_data) + "," + ",".join([base_point for base_point in veh_base_point_data])
        list_file.write(single_line)


if __name__ == "__main__":
    wd = getcwd()

    for year, image_set in sets:  # train/val/test
        image_ids = open('DATAdevkit/%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()  # train/val/test
        list_file = open('%s_%s.txt'%(year, image_set), 'w')  # write to a new txt
        for image_id in image_ids:  # train/val/test imgs
            # file_path calib_xml_path left,top,width,height,cls_id,cx1,cy1,u0,v0,...,u7,v7,v_l,v_w,v_h,pers,bpx1,bpx2
            if image_set == "test":
                year = "TESTDATA2021"
            list_file.write('%s/DATAdevkit/%s/JPEGImages/%s.jpg'%(wd, year, image_id))  # write absolute path of imgs to txt
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()

    print("finish convert scripts!")
