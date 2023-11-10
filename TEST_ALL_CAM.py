import pyrealsense2 as rs
import cv2
import cv2
import time
import numpy as np
import os
import glob
from CONFIG_ODOMETRY_SYSTEM import *



def capture_frames_CAM1():

    pipeline = rs.pipeline()
    pipeline.start()

    time.sleep(1)
    print("GO!")
    fr = 0

    while fr < 30:
        frames = pipeline.wait_for_frames()
        f1 = frames.get_fisheye_frame(1)


        image1 = np.asanyarray(f1.get_data())

        cv2.namedWindow('RealSense cam1', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense cam1', image1)
        fr = fr + 1
        key = cv2.waitKey(1)
        if key == 27: # ESC
            return


    pipeline.stop()


def capture_frames_CAM2():

    pipeline = rs.pipeline()
    pipeline.start()

    time.sleep(1)
    print("GO!")
    fr = 0

    while fr < MAX_FRAME_TEST:
        frames = pipeline.wait_for_frames()
        f2 = frames.get_fisheye_frame(2)


        image1 = np.asanyarray(f2.get_data())

        cv2.namedWindow('RealSense cam2', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense cam2', image1)
        fr = fr + 1
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            return


    pipeline.stop()



try:
    capture_frames_CAM1()
except Exception as e:
    print("ERROR T265 CAM1 : ",e)

try:
    capture_frames_CAM2()
except Exception as e:
    print("ERROR T265 CAM2 : ",e)