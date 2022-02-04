# predict videos
from box_predict import Bbox3dPred
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2 as cv
import time

model = Bbox3dPred()

video_path      = "../videos/session0_center.avi"
video_save_path = video_path[:-4] + "_out.mp4"
calib_path = video_path[:-4] + "_calibParams.xml"

capture = cv.VideoCapture(video_path)
start_frame = 0
capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
video_fps = capture.get(cv.CAP_PROP_FPS)
print(video_fps)
frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

if video_save_path != "":
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
    out = cv.VideoWriter(video_save_path, fourcc, video_fps, size)

ref, frame = capture.read()
if not ref:
    raise ValueError("fail to load videos!")

fps = 0.0
for index in tqdm(range(frame_count-start_frame*8)):
    ref, frame = capture.read()
    if not ref:
        break
    # BGR -> RGB
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))
    r_image, _, process_time = model.detect_image(frame, video_path.split("/")[-1][:-4]+"_"+str(index), False, False, calib_path, "test")
    frame = np.array(r_image)
    # RGB -> BGR
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    fps = 1. / process_time
    # print("fps=%.2f" % fps)
    frame = cv.putText(frame, "fps=%.2f" % fps, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("video", frame)
    c = cv.waitKey(1) & 0xff
    if video_save_path != "":
        out.write(frame)

    if c == 27:
        capture.release()
        break

print("Video Detection Done!")
capture.release()
if video_save_path != "":
    print("Save processed video to the path :" + video_save_path)
    out.release()
cv.destroyAllWindows()
