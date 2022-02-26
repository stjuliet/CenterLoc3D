# split train/val/test datasets
# val will be re-splitted in train.py
# test.txt and val.txt is empty

import os
import random
random.seed(0)  # keep split results same

trainval_xmlfilepath = "../voc_format/DATA2022/Annotations"
# test_xmlfilepath = "../voc_format/TESTDATA2022/Annotations"
saveBasePath = "../voc_format/DATA2022/ImageSets/Main"

# usually 9:1
trainval_percent = 1  # test from TESTDATA2021
train_percent = 1

temp_trainval_xml = os.listdir(trainval_xmlfilepath)
total_trainval_xml = []
for xml in temp_trainval_xml:
    if xml.endswith(".xml"):
        total_trainval_xml.append(xml)

# temp_test_xml = os.listdir(test_xmlfilepath)
# total_test_xml = []
# for xml in temp_test_xml:
#     if xml.endswith(".xml"):
#         total_test_xml.append(xml)

num_trainval = len(total_trainval_xml)
list_trainval = range(num_trainval)
tv = int(num_trainval*trainval_percent)
tr = int(tv*train_percent)
trainval = random.sample(list_trainval, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train size", tr)
ftrainval = open(os.path.join(saveBasePath, "trainval.txt"), "w")
ftest = open(os.path.join(saveBasePath, "test.txt"), "w")
ftrain = open(os.path.join(saveBasePath, "train.txt"), "w")
fval = open(os.path.join(saveBasePath, "val.txt"), "w")

# train val
for i in list_trainval:
    name = total_trainval_xml[i][:-4] + "\n"
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

# # test
# for test_xml in total_test_xml:
#     ftest.write(test_xml[:-4] + "\n")

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

print("finish split datasets!")
