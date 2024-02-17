# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:39:42 2024

@author: abubakar.siddique
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("people.mp4")

assert cap.isOpened(), "Error reading video file"
# assert cap1.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w, h, fps)
line_points = region_points = [(200, 400), (750, 400)]#[(0, 0), (1280, 0), (1280, 720), (0, 720)]##[(0, 0), (2560, 0), (2560, 1440), (0, 1440)]#  # line or region points
classes_to_count = [0]#[0, 2]  # person and car classes for count

# Video writer
# video_writer = cv2.VideoWriter("C:/Users/abubakar.siddique/Downloads/object_counting_output-office.avi",
#                         cv2.VideoWriter_fourcc(*'mp4v'),
#                         fps,
#                         (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                  reg_pts=line_points,
                  classes_names=model.names,
                  draw_tracks=True)

# counter1 = object_counter.ObjectCounter()
# counter1.set_args(view_img=False,
#                   reg_pts=line_points,
#                   classes_names=model.names,
#                   draw_tracks=True)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

while cap.isOpened():
    success, im0 = cap.read()
    # print(im0.shape[:2])
    im0 = ResizeWithAspectRatio(im0, width=1280)
    # print(im0.shape[:2])
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False,
                          classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    

    # video_writer.write(im0)

cap.release()
#video_writer.release()
cv2.destroyAllWindows()