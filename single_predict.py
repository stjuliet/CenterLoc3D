# predict func
from box_predict import Bbox3dPred
from PIL import Image

model = Bbox3dPred()
# 单张图片，不记录测试结果，只显示
record_result = False

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image, single_proc_time = model.detect_image(image, img.split("/")[-1][:-4], record_result)
        r_image.show()
        print("Single FPS: ", round(1 / single_proc_time, 4))
