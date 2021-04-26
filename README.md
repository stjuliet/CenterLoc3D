# bbox3d_recognition

单目交通场景三维目标检测✌✌

## 数据集及训练配置流程

1. 采用Labelimg3D标注工具进行标注，格式类VOC

2. 使用copy_dataset.py将数据集图片及标注文件复制到DATAdevkit/DATA2021/Annotations和DATAdevkit/DATA2021/JPEGImages两个文件夹下

3. 运行DATAdevkit/DATA2021/split_train_val_test.py划分训练集\验证集\测试集，在DATAdevkit/DATA2021/ImageSets/Main文件夹下生成train.txt、trainval.txt、val.txt、test.txt

4. 运行data2script.py将真实标签生成DATA2021_train.txt、DATA2021_val.txt、DATA2021_test.txt用于训练读取标签数据

5. 设置超参数，运行train.py开始训练

## 预测流程

预测时均可设置record_result，控制是否记录预测结果到txt文件用于评价网络mAP

- single_predict.py --- 单张图像预测

- batch_predict.py --- 批量图像预测
