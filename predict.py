
from box_predict import Bbox3dPred
from PIL import Image

model = Bbox3dPred()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = model.detect_image(image)
        # r_image.save("img/img.jpg")
        r_image.show()
        