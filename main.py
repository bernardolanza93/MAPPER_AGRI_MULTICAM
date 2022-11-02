#!/usr/bin/python
import sys
import time

import pyrealsense2 as rs
import cv2
import numpy as np
import os


'''
pip install pyrealsense2
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev


'''


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


SAVE_VIDEO_TIME = 10 # 0 per non salvare


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
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    #config.enable_stream(rs.stream.accel,rs.format.motion_xyz32f,200)
    #config.enable_stream(rs.stream.gyro,rs.format.motion_xyz32f,200)

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


    try:
    # Start streaming
        pipelineT265.start(configT265)
        print("T265 started")
    except Exception as e:
        print("error pipeline T265 starting:||||:: %s", str(e))
    #_______________________________________________________

if SAVE_VIDEO_TIME != 0:

    result = cv2.VideoWriter('/data/filename.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             20.0, (1080, 720),1)



while True:

    # T265
    if enable_T265:
        tframes = pipelineT265.wait_for_frames()
        pose = tframes.get_pose_frame()
        if pose:
            data = pose.get_pose_data()

            #print("Frame #{}".format(pose.frame_number))
            #print("Position: {}".format(data.translation))
            #print("Velocity: {}".format(data.velocity))
            #print("Acceleration: {}\n".format(data.acceleration))

    if enable_D435i:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        #frames.as_motion_frame().get_motion_data()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()


        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        #print(depth_image.shape) 720*1080
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        if SAVE_VIDEO_TIME != 0:
            result.write(color_image)
            #cv2.imwrite('im.jpg', color_image)

            result.write(color_image)
        #print("size", depth_image.shape,  color_image.shape)
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('Color Stream', images)


       #cv2.imshow('depth Stream', depth_image)
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
    result.release()
    cv2.destroyAllWindows()
if enable_T265:
    pipelineT265.stop()


