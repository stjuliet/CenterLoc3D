# predict func
from box_predict import Bbox3dPred
from PIL import Image

model = Bbox3dPred()
# predict single img for show
record_result = False
# draw gt boxes in imgs
draw_gt = False


while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image, _, single_proc_time = model.detect_image(image, img.split("/")[-1][:-4], draw_gt, record_result, None, "test")
        r_image.show()
        # r_image.save("test.jpg")
        print("Single FPS: ", round(1 / single_proc_time, 4))
