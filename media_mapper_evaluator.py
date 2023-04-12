



import numpy as np


import time

from numpy.linalg import norm
import os

import os
import sys
import math
import cv2
import csv
from datetime import datetime


from evaluator_utils import *

"""
D435's FOV is 91.2 x 65.5 x 100.6. 

"""







PATH_HERE = os.getcwd()
PATH_2_FILE = "/data/"
PATH_2_AQUIS = "/aquisition/"
SAVE_VIDEO = False
TRACKBAR = False
THRESHOLD = True
OPENING = False
PIXEL_COUNTING = True
MASK_DEPTH = True
CONVERT_DEPTH_TO_1CH = False
CROPPING = False
MEDIUM_DEPTH_DISPLAY = True
BLOB_DETECTOR = True
FILTER_DEPTH = False





BOT = (0, 8, 11)
TOP = (180, 218, 126)

#print("thres_value = ",THRES_VALUE)


check_folder("/aquisition/")
dimension = 50
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


intrinsics = obtain_intrinsics()
print(intrinsics)



for folders in os.listdir(PATH_HERE + PATH_2_AQUIS):
    print("files:", os.listdir(PATH_HERE + PATH_2_AQUIS))
    folder_name = folders
    #videos = os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name)
    writeCSVdata(folder_name, ["frame", "pixel", "volume", "distance_med", "volumes", "distances"])

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
    #print("height : ", frame_height)
    #print("width : ", frame_width)

    size = (frame_width, frame_height)
    size2 = (frame_width2, frame_height2)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    if SAVE_VIDEO:
        result = cv2.VideoWriter('filename.avi',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 20, size)


    nrfr = 0
    total1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    print("frames", total1)
    print("frames", total2)
    time.sleep(1)
    while(video1.isOpened() and video2.isOpened()):
        ret, frame = video1.read()
        ret2, frame2 = video2.read()
        print(ret,ret2)





        if ret == True and ret2 == True:

            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


            # Writ ethe frame into the
            # file 'filename.avi'
            if SAVE_VIDEO:

                result.write(frame)


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            #sistemi di mask generation
            if TRACKBAR:
                frame_HSV = frame
                mask = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            elif THRESHOLD:
                ret, mask = cv2.threshold(gray, THRES_VALUE, 255, cv2.THRESH_BINARY)
                imask = mask < 255
                imagem = (255 - mask)
                green = 255 * np.ones_like(frame, np.uint8)
                green[imask] = frame[imask]  # dentro i mask metto frame
            #frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #mask = cv2.inRange(frame_HSV, BOT, TOP)

            if OPENING:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            if FILTER_DEPTH:
                max_depth = 260
                min_depth = 60
                frame = set_white_extreme_depth_area (frame, frame2 ,max_depth, min_depth)
            SAVE = False
            if BLOB_DETECTOR:
                #print("___1___",mask,"____2__", frame,"___3___", green ,"___4___",frame2)
                #print("frames s"Right"hapes", mask.shape, frame.shape, green .shape, frame2.shape)
                frame,mask,frame2, pixel, volume, cylindrification_results, completation , dA, dB = blob_detector(frame ,frame2, intrinsics)
                if completation:
                    SAVE = True

            if MASK_DEPTH:
                imask = mask < 255
                frame22 = 255 * np.ones_like(frame2, np.uint8) #all white
                frame22[imask] = frame2[imask]

                distance_med = extract_medium_from_depth_segmented(frame2[imask])
                print("distance", distance_med)
                #print("dist ", distance_med)


                frame2 = frame22




            if PIXEL_COUNTING:
                if SAVE:

                    writeCSVdata(folder_name,[nrfr,pixel,int(volume),int(distance_med), cylindrification_results[0], cylindrification_results[1]])
                    #writeCSVdata(folder_name + "geometrical", [int(dA), int(dB) , int(distance_med)])

                    #print(len(cylindrification_results[1]))
                #print([nrfr,pixel])
            nrfr += 1



            #frame = resize_image(frame,dimension)
            #green = resize_image(green, dimension)
            #mask = resize_image(mask, dimension)

            #frame2 = resize_image(frame2, dimension)



            cv2.imshow("or", frame)
            # cv2.imshow("mask", mask)
            # #cv2.imshow("green", green)
            # cv2.imshow("frame2", frame2)
            # #cv2.imshow("edge", edge)
            # #cv2.imshow("skel", skel)
            cv2.moveWindow("mask", 500, 200)
            # cv2.moveWindow("frame2", 200, 500)



            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                sys.exit()
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture and video
    # write objects
    video1.release()
    video2.release()
    if SAVE_VIDEO:
        result.release()

    # Closes all the frames
    cv2.destroyAllWindows()


sys.exit()
