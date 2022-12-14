#!/usr/bin/python
import sys
import time

import pyrealsense2 as rs
import cv2
import numpy as np
import os
from datetime import datetime
from evaluator_utils import *

#png uint 16#

'''
pip install pyrealsense2
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev


'''


offset = np.tile(50, (1080,1920))
T265_MANDATORY = False


def check_folder(relative_path):
    """
    check_folder : check  the existence and if not, create the path for the results folder

    :param relative_path:path to be checked


    :return nothing:
    """

    workingDir = os.getcwd()
    path = workingDir + relative_path

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

        print("The new directory is created!", path)
    else:
        print('directory ok:', path)


check_folder("/data/")
now = datetime.now()
hourstr = now.strftime("%Y-%m-%d %H:%M:%S")
#acquisition_today =  "aquisition_" + str(now)
#save_location = "/data/"+acquisition_today
#check_folder(save_location)

path_here = os.getcwd()
SAVE_VIDEO_TIME = 10 # 0 per non salvare
FPS_DISPLAY = True

def writeCSVdata(time,data):
    """
    write data 2 CSV
    :param data: write to a csv file input data (append to the end)
    :return: nothing
    """
    # scrive su un file csv i dati estratti dalla rete Neurale
    name = "localization"

    file = open('./data/' + name + '_'+ time +'.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(data)
    file.close()

def search_device(ctx):
    enable_D435i = False
    enable_T265 = False
    device_aviable = {}

    print("ctx.devices  = ",ctx.devices)
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print(d)
            device = d.get_info(rs.camera_info.name)
            serial = d.get_info(rs.camera_info.serial_number)
            model = str(device.split(' ')[-1])

            device_aviable[model] = [serial,device]


            print('Found device: ', device_aviable)

    else:
        print("No Intel Device connected")
    keys = list(device_aviable.keys())
    for i in range(len(keys)):
        if keys[i] == "D435I":
            enable_D435i = True
        elif keys[i] == "T265":
            enable_T265 = True
        else:
            print("camera not recognized")

    print("D435i ok ? => ", enable_D435i, "___|||____  T265 ok ? => ", enable_T265)

    return enable_D435i, enable_T265, device_aviable











##config.enable_device('947122110515')

ctx = rs.context()
enable_D435i, enable_T265, device_aviable = search_device(ctx)



time.sleep(1)

#D435____________________________________________
if enable_D435i:
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    seriald435 = str(device_aviable['D435I'][0])
    print("serial : ", type(seriald435))
    config.enable_device(seriald435)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280 , 720 , rs.format.z16, 30)

    #config.enable_stream(rs.stream.accel,rs.format.motion_xyz32f,200)
    #config.enable_stream(rs.stream.gyro,rs.format.motion_xyz32f,200)
    saver = rs.save_single_frameset()
    align_to = rs.stream.color
    align = rs.align(align_to)
    try:
    # Start streaming
        pipeline.start(config)
        print("D435I started")
    except Exception as e:
        print("error pipeline D435 starting:||||:: %s", str(e))
    #_________________________________________________

if enable_T265:
    #T265_________________________________________________
    pipelineT265 = rs.pipeline(ctx)
    configT265 = rs.config()
    serialt265 = str(device_aviable['T265'][0])
    print(serialt265)
    configT265.enable_device(serialt265)
    configT265.enable_stream(rs.stream.pose)

    #saver.set_option()


    try:
    # Start streaming
        pipelineT265.start(configT265)
        print("T265 started")
    except Exception as e:
        print("error pipeline T265 starting:||||:: %s", str(e))
    #_______________________________________________________

if SAVE_VIDEO_TIME != 0:
    now = datetime.now()
    hourstr = now.strftime("%Y-%m-%d %H:%M:%S")

    gst_out = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=RGB.mkv "
    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER,  20.0, (1920, 1080))

    try:
        #gst_out_depth   = "appsrc ! video/x-raw, format=GRAY ! queue ! videoconvert ! video/x-raw,format=GRAY ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=DEPTH.mkv "
        gst_out_depth = "appsrc caps=video/x-raw,format=GRAY8 ! videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! h265parse ! filesink location=DEPTH.mkv "
        #gst_out_depth = ("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=test.mkv" )
        #gst_out_depth = ('appsrc caps=video/x-raw,format=GRAY8,width=1920,height=1080,framerate=30/1 ! '' videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! ''h265parse ! filesink location=test.h265 ')
        out_depth = cv2.VideoWriter(gst_out_depth, cv2.CAP_GSTREAMER,  20.0, (1920, 1080),0)


    except Exception as e:
        print("error save 1ch depth:||||:: %s", str(e))


frame = 0
now = datetime.now()
time1 = now.strftime("%d-%m-%Y|%H:%M:%S")

while True:
    frame += 1

    # T265
    start = time.time()

    if enable_T265:
        tframes = pipelineT265.wait_for_frames()
        pose = tframes.get_pose_frame()
        if pose:
            data = pose.get_pose_data()

            #print("Frame #{}".format(pose.frame_number))
            #print("Position: {}".format(data.translation))
            #print("Velocity: {}".format(data.velocity))
            #print("Acceleration: {}\n".format(data.acceleration))
            writeCSVdata(time1,[frame,data.translation,data.velocity])

    if enable_D435i:
        # Wait for a coherent pair of frames: depth and color

        try:
            frames = pipeline.wait_for_frames()

        except Exception as e:
            print("PIPELINE error:||||:: %s", str(e))
            sys.exit()

        #frames.as_motion_frame().get_motion_data()


        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()



        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())



        width = int(1920)
        height = int(1080)
        dim = (width, height)

        # resize image depth to fit rgb
        resized = cv2.resize(depth_image, dim, interpolation=cv2.INTER_AREA)

        #convert u16 mm bw image to u16 cm bw
        resized = resized/10
        #rescale without first 50 cm of offset unwanted
        resized = resized - offset
        #tolgo tutto sotto i 30 cm


        #stretchin all in the 0-255 cm interval
        maxi = np.clip(resized,0,255)
        #convert to 8 bit
        intcm = maxi.astype('uint8')



        if SAVE_VIDEO_TIME != 0:
            try:
                out.write(color_image)
                try:
                    #save here depth map??
                    out_depth.write(intcm)
                    #np.savetxt("image.txt", depth_image,fmt='%i')

                except Exception as e:
                    print("error saving depth 1 ch:||||:: %s", str(e))
                #cv2.imwrite('im.jpg', color_image)
                #frames = pipeline.wait_for_frames()
                #saver.process(frames)
                pass


            except Exception as e:
                print("error save video:||||:: %s", str(e))
            #cv2.imwrite('im.jpg', color_image)

            #result.write(color_image)


        #print("size", depth_image.shape,  color_image.shape)
        #images = np.hstack((color_image, depth_colormap))
        #cv2.imshow('Color Stream', depth_image)

        color_image = resize_image(color_image,50)
        depth_image = resize_image(depth_image, 50)
        #cv2.imshow('depth Stream', color_image)
        #cv2.imshow('dept!!!h Stream', intcm)

        if FPS_DISPLAY:
            end = time.time()
            seconds = end - start
            fps = 1 / seconds
            print(fps)
        key = cv2.waitKey(1)
        if key == 27:
            #result.release()
            #cv2.destroyAllWindows()
            break


    if enable_T265 == False and enable_D435i == False:
        print("no device, termination...")
        sys.exit()



if enable_D435i:
    pipeline.stop()
    out.release()
    out_depth.release()
    cv2.destroyAllWindows()
if enable_T265:
    pipelineT265.stop()


