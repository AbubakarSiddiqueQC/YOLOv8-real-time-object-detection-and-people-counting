# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:10:13 2024

@author: abubakar.siddique
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2 # Import OpenCV Library
from ultralytics import YOLO # Import Ultralytics Package
import threading # Threading module import
# Define the video files for the trackers
video_file1 = "rtsp://live.video.cam" 
video_file2 = "rtsp://live.video.cam" 
model = YOLO("yolov8n.pt")
classes_to_count = [0]
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

def run_tracker_in_thread(filename, model, file_index):
    """
    This function is designed to run a video file or webcam stream
    concurrently with the YOLOv8 model, utilizing threading.

    - filename: The path to the video file or the webcam/external
    camera source.
    - model: The file path to the YOLOv8 model.
    - file_index: An argument to specify the count of the
    file being processed.
    """

    video = cv2.VideoCapture(filename)  # Read the video file
    # start_frame_number = 10
    # video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    while True:
        ret, frame = video.read()  # Read the video frames
        
        frame = ResizeWithAspectRatio(frame, width=1280)
        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True,classes=classes_to_count)
        res_plotted = results[0].plot()
        print(res_plotted.shape[:2])
        cv2.imshow("Tracking_Stream_"+str(file_index), res_plotted)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()


# Create the tracker thread

# Thread used for the video file
tracker_thread1 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file1, model, 1),
                                   daemon=True)

# Thread used for the webcam
# tracker_thread2 = threading.Thread(target=run_tracker_in_thread,
#                                    args=(video_file2, model, 2),
#                                    daemon=True)

# Start the tracker thread

# Start thread that run video file
tracker_thread1.start()

# Start thread that run webcam
# tracker_thread2.start()

# Wait for the tracker thread to finish

# Tracker thread 1
tracker_thread1.join()

# Tracker thread 2
# tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()