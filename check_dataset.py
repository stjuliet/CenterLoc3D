# check scene change and img index continuous in a single scene

import os

f = open("DATAdevkit/DATA2021/ImageSets/Main/trainval.txt", "r")
list_n = f.readlines()
for index, element in enumerate(list_n):
    cur_num = int(element.split("_")[-1])
    if index > 0:
        last_num = int(list_n[index-1].split("_")[-1])
        if abs(cur_num - last_num) != 1:
            print(list_n[index])
