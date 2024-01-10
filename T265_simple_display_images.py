import pyrealsense2 as rs
import cv2
import cv2
import time
import numpy as np
import os
import glob

def display_t265():

    pipeline = rs.pipeline()
    pipeline.start()
    print(" ESC key to terminate")
    time.sleep(2)
    print("GO!")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            f1 = frames.get_fisheye_frame(1)
            f2 = frames.get_fisheye_frame(2)

            if not f1:
                continue
            if not f2:
                continue

            image1 = np.asanyarray(f1.get_data())
            image2 = np.asanyarray(f2.get_data())
            im_v = cv2.hconcat([image1, image2])

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', im_v)
            key = cv2.waitKey(1)
            if key == 27: # ESC
                return



    finally:
        pipeline.stop()

try:
    display_t265()
except Exception as e:
    print("ERROR T265  : ",e)

