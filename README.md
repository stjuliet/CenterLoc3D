# bbox3d_recognition

单目交通场景车辆三维目标检测网络的Pytorch实现✌✌

## 运行环境

- Windows 10

- Python 3.6.2

- PyTorch 1.4.0

- torchvision 0.4.2

- CUDA 10.1

- cuDNN 7.6.5

- tensorboard 2.5.0、tensorboardX 2.2.0、tensorflow-gpu 1.9.0

## 数据集及训练配置流程

1. 采用Labelimg3D标注工具进行标注，格式类VOC，存放于任意路径

2. 使用copy_dataset.py将数据集图片及标注文件复制到DATAdevkit/DATA2021/Annotations和DATAdevkit/DATA2021/JPEGImages两个文件夹下

3. 运行DATAdevkit/DATA2021/split_train_val_test.py划分训练集\验证集\测试集，在DATAdevkit/DATA2021/ImageSets/Main文件夹下生成train.txt、trainval.txt、val.txt、test.txt

4. 运行data2script.py将真实标签生成DATA2021_train.txt、DATA2021_val.txt、DATA2021_test.txt用于训练读取标签数据

5. 设置超参数（包括batch size, epoch等），运行train.py开始训练（可设置断点恢复训练）

## 预测流程

预测时均可设置record_result，控制是否记录预测结果到txt文件用于评价网络mAP

- single_predict.py --- 单张图像预测

- batch_predict.py --- 批量图像预测（带标签，可用于指标评价）

- batch_predict_no_anno.py --- 批量图像预测（不带标签，不可用于指标评价，仅能用于可视化）

## 评价指标

- 2d/3d mAP (阈值参数分别为0.5，0.7)

step 1. 在box_predict.py中设置训练好的模型路径、backbone名称等参数

step 2. 运行get_gt_txt_2d3d.py获得真实标签结果（写入(val/test)/input-2d(3d)/ground-truth中，包括真实标签txt），运行batch_predict.py获得预测结果（写入(val/test)/input-2d(3d)/detection-results中，包括检测结果可视化、热力图可视化和检测结果txt），其中可指定在val/test数据集上

ground-truth/detection-results 2D txt文件格式：
```
gt:
type xmin ymin xmax ymax
type xmin ymin xmax ymax
...

dt:
type score xmin ymin xmax ymax
type score xmin ymin xmax ymax
...
```

ground-truth/detection-results 3D txt文件格式：
```
gt:
     [        mm         ] [   m  ] [  mm  ]
type x1 y1 z1 ... x8 y8 z8 lv wv hv cx cy cz
type x1 y1 z1 ... x8 y8 z8 lv wv hv cx cy cz
...

dt:
           [        mm         ] [   m  ] [  mm  ]
type score x1 y1 z1 ... x8 y8 z8 lv wv hv cx cy cz
type score x1 y1 z1 ... x8 y8 z8 lv wv hv cx cy cz
...
```

step 3. 分别运行get_map_2d.py、get_map_3d.py获得评价结果，保存至(val/test)文件夹下（带阈值区分）

- 定位精度、三维物理尺寸精度

在评价完2d/3d mAP的基础上，运行calc_pos_size_precision.py，可获得测试集中定位、三维物理尺寸的精度和误差，对于误差可以细化至分场景，并且能够输出定位俯视图、定位和尺寸误差曲线图（误差与车辆-相机距离）
