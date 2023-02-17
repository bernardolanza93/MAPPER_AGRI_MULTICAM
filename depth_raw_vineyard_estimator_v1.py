
import time

from numpy.linalg import norm
import os

import os
import sys
import math
import cv2
import csv
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from datetime import datetime
import pyrealsense2 as rs
from evaluator_utils import *




PATH_HERE = os.getcwd()
PATH_2_FILE = "/data/"
PATH_2_AQUIS = "/aquisition/"


for folders in os.listdir(PATH_HERE + PATH_2_AQUIS):
    print("files:", os.listdir(PATH_HERE + PATH_2_AQUIS))
    folder_name = folders
    #videos = os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name)
    writeCSVdata(folder_name, ["image_array"])

    for videos in os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name):

        #print(videos)
        print("ITERATION:", folder_name)




        if videos.endswith(".mkv"):
            print(videos.split(".")[0])

            if videos.split(".")[0] == "RGB":

                path_rgb= os.path.join(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name, videos)
                video1 = cv2.VideoCapture(path_rgb)
            elif videos.split(".")[0] == "DEPTH":

                path_depth= os.path.join(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name, videos)
                video2 = cv2.VideoCapture(path_depth)



    # We need to check if camera
    # is opened previously or not
    if (video1.isOpened() == False):
        print("Error reading video rgb")
    if (video2.isOpened() == False):
        print("Error reading video depth")
    # We need to set resolutions.
    # so, convert them from float to integer.

    frame_width = int(video1.get(3))
    frame_height = int(video1.get(4))
    frame_width2 = int(video2.get(3))
    frame_height2 = int(video2.get(4))
    print(frame_height2, " x ", frame_width2)
    #print("height : ", frame_height)
    #print("width : ", frame_width)



    time.sleep(1)
    while video2.isOpened():


        ret2, frame2 = video2.read()
        print(ret2)

        if ret2 == True:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            #this frame is ready to save data of depth
            # Multiplying arrays
            one_dim_image = frame2.flatten()
            # printing result
            writeCSVdata("dept_intensity_" + folder_name, one_dim_image )
            cv2.imshow("d", frame2)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                sys.exit()
                break

            if cv2.getWindowProperty("d", cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            break
    video2.release()
    cv2.destroyAllWindows()
    print(".")
    print("terminated")



