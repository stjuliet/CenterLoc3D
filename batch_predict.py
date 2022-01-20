# bacth predict imgs in test.txt
# and save results for eval

from box_predict import Bbox3dPred
from PIL import Image
from tqdm import tqdm
import cv2 as cv

model = Bbox3dPred()

mode = "test"  # val/test datasets

test_txt_path = "DATA2021_%s.txt" % mode

record_result = True
# draw gt boxes in imgs
draw_gt = True
# visualization
save_test_img = True

total_proc_time = 0
with open(test_txt_path, "r") as fread:
    test_file_path_list = fread.readlines()
    with tqdm(total=len(test_file_path_list), postfix=dict) as pbar:
        for single_test_file_path in test_file_path_list:
            image = Image.open(single_test_file_path.split(" ")[0])
            image_id = single_test_file_path.split(" ")[0].split("/")[-1][:-4]
            calib_path = single_test_file_path.split(" ")[1]
            r_image, r_heatmap, proc_time = model.detect_image(image, image_id, draw_gt, record_result, calib_path, mode)
            total_proc_time += proc_time
            if save_test_img:
                r_image.save("./%s/input-2D/images-optional/"%mode + str(single_test_file_path.split(" ")[0].split("/")[-1]))
                cv.imwrite("./%s/input-2D/images-heatmap/"%mode + str(single_test_file_path.split(" ")[0].split("/")[-1][:-4] + "_heatmap.png"), r_heatmap)
                # r_heatmap.save("./input-2D/images-heatmap/" + str(single_test_file_path.split(" ")[0].split("/")[-1][:-4] + "_heatmap.png"))
                # r_image.save("./input-3D/images-optional/" + str(single_test_file_path.split(" ")[0].split("/")[-1]))
            
            pbar.set_postfix(**{"single fps" : round(1 / proc_time, 4)})  # **.****
            pbar.update(1)

print("Average FPS: ", round(len(test_file_path_list)/total_proc_time, 4))
print("batch predict finished!")

if record_result:
    print("results recorded!")

if save_test_img:
    print("images saved!")
