from scipy.spatial import distance as dist
import os
import cv2
import math
import time
import numpy as np
import statistics
import pyrealsense2 as rs
import csv
from scipy.spatial import cKDTree
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from CONFIGURATION_VISION import *
from pointcloud_utility_library import *
import open3d as o3d
from sklearn.preprocessing import RobustScaler
import pyntcloud
import pandas as pd
import random
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from misc_utility_library import *

print(cv2.__version__)


custom_width = 1920  # Set your desired width
custom_height = 1080  # Set your desired height


def UNDO_convert_u8_img_to_u16_d435_depth_image(u16image):
    resized = u16image / 10
    # rescale without first 50 cm of offset unwanted
    resized = resized - OFFSET_CM_COMPRESSION

    # stretchin all in the 0-255 cm interval
    maxi = np.clip(resized, 0, 255)
    # convert to 8 bit
    intcm = maxi.astype('uint8')
    return intcm






def convert_u8_img_to_u16_d435_depth_image(u8_image):

    u8_image = u8_image + OFFSET_CM_COMPRESSION
    u16_image = u8_image.astype('uint16')
    u16_image_off = u16_image
    u16_image_off_mm = u16_image_off * 10
    return u16_image_off_mm





for folders in os.listdir(PATH_HERE + PATH_2_AQUIS):
    #print("files:", os.listdir(PATH_HERE + PATH_2_AQUIS))
    folder_name = folders
    # videos = os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name)
    # writeCSVdata(folder_name, ["frame", "pixel", "volume", "distance_med", "volumes", "distances"])

    for videos in os.listdir(PATH_HERE + PATH_2_AQUIS + "/" + folder_name):

        # print(videos)
        print("FILENAME:", folder_name)

        if videos.endswith(".mkv"):
            #print(videos.split(".")[0])

            if videos.split(".")[0] == "RGB":

                path_rgb = os.path.join(PATH_HERE + PATH_2_AQUIS + "/" + folder_name, videos)
                # creo l oggetto per lo streaming
                video1 = cv2.VideoCapture(path_rgb)
            elif videos.split(".")[0] == "DEPTH":

                path_depth = os.path.join(PATH_HERE + PATH_2_AQUIS + "/" + folder_name, videos)
                video2 = cv2.VideoCapture(path_depth)

        # We need to check if camera
        # is opened previously or not
    if (video1.isOpened() == False):
        print("Error reading video rgb")
    if (video2.isOpened() == False):
        print("Error reading video depth")

    frame_width = int(video1.get(3))
    frame_height = int(video1.get(4))
    frame_width2 = int(video2.get(3))
    frame_height2 = int(video2.get(4))
    # print("height : ", frame_height)
    # print("width : ", frame_width)
    size = (frame_width, frame_height)
    size2 = (frame_width2, frame_height2)

    nrfr = 0
    time.sleep(1)

    volumes_extracted = []

    while (video1.isOpened() and video2.isOpened()):


        ret, frame = video1.read()
        ret2, frame2 = video2.read()
        nrfr = nrfr + 1
        #print(nrfr)


        if ret and ret2:
            if nrfr > 3 and nrfr < 29:


                depth_image = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                # maschero il depth con la maschera rgb


                original_d = depth_image

                # Split the RGB guide image into separate color channels
                b, g, r = cv2.split(frame)

                # Apply the guided filter separately to each color channel of the depth image
                radius = 15  # Radius of the filter
                epsilon = 0.1  # Regularization parameter

                filtered_depth_r = cv2.ximgproc.guidedFilter(r, depth_image, radius, epsilon)
                filtered_depth_g = cv2.ximgproc.guidedFilter(g, depth_image, radius, epsilon)
                filtered_depth_b = cv2.ximgproc.guidedFilter(b, depth_image, radius, epsilon)

                # Merge the filtered color channels back into a single depth image
                filtered_depth_image = cv2.merge((filtered_depth_b, filtered_depth_g, filtered_depth_r))

                depth_image_8u = filtered_depth_image.astype(np.uint8)
                depth_image_8u = cv2.cvtColor(depth_image_8u, cv2.COLOR_BGR2GRAY)


                concatenated_image = cv2.hconcat([original_d, depth_image_8u])
                #
                conc = resize_image(concatenated_image,50)
                #






                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', conc)
                key = cv2.waitKey(CONTINOUS_STREAM)
                if key == ord('q') or key == 27:
                    break






    cv2.destroyAllWindows()