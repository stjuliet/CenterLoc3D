# 批量预测test.txt文件中的图片
import os
from box_predict import Bbox3dPred
from PIL import Image

model = Bbox3dPred()

test_txt_path = "DATA2021_test.txt"

if not os.path.exists("test_results"):
    os.makedirs("test_results")


with open(test_txt_path, "r") as fread:
    test_file_path_list = fread.readlines()
    for single_test_file_path in test_file_path_list:
        image = Image.open(single_test_file_path.split(" ")[0])
        r_image = model.detect_image(image)
        r_image.save("test_results/test_img_" + str(single_test_file_path.split(" ")[0].split("/")[-1]))

