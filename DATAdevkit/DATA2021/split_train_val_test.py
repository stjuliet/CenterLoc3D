# split train/val/test datasets
# 验证集的划分在train.py代码里面进行
# 执行完此代码test.txt和val.txt里面没有内容

import os
import random
random.seed(0)  # 每次划分结果一样

xmlfilepath = "DATAdevkit/DATA2021/Annotations"
saveBasePath = "DATAdevkit/DATA2021/ImageSets/Main"

# 想要增加测试集修改trainval_percent
# train_percent不需要修改
# 训练验证、测试 9:1
trainval_percent = 0.95
train_percent = 1

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num = len(total_xml)
list_n = range(num)
tv = int(num*trainval_percent)
tr = int(tv*train_percent)
trainval = random.sample(list_n, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train size", tr)
ftrainval = open(os.path.join(saveBasePath, "trainval.txt"), "w")
ftest = open(os.path.join(saveBasePath, "test.txt"), "w")
ftrain = open(os.path.join(saveBasePath, "train.txt"), "w")
fval = open(os.path.join(saveBasePath, "val.txt"), "w")

for i in list_n:
    name = total_xml[i][:-4] + "\n"
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

print("finish split!")