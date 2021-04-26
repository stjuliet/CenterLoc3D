# 批量预测test.txt文件中的图片, 并记录测试结果, 用于评价map

from box_predict import Bbox3dPred
from PIL import Image
from tqdm import tqdm

model = Bbox3dPred()

# 测试集文件路径
test_txt_path = "DATA2021_test.txt"
# 是否记录测试结果，用于评价map
record_result = True
# 是否保存测试结果图片
save_test_img = False

total_proc_time = 0
with open(test_txt_path, "r") as fread:
    test_file_path_list = fread.readlines()
    with tqdm(total=len(test_file_path_list),postfix=dict) as pbar:
        for single_test_file_path in test_file_path_list:
            image = Image.open(single_test_file_path.split(" ")[0])
            image_id = single_test_file_path.split(" ")[0].split("/")[-1][:-4]
            r_image, proc_time = model.detect_image(image, image_id, record_result)
            total_proc_time += proc_time
            if save_test_img:
                r_image.save("./input/images-optional/"+str(single_test_file_path.split(" ")[0].split("/")[-1]))
            
            pbar.set_postfix(**{'single fps' : 1/proc_time})
            pbar.update(1)

print("Average FPS: ", len(test_file_path_list)/total_proc_time)
