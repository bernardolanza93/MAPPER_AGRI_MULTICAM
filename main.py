#!/usr/bin/python
import sys
import time

import pyrealsense2 as rs
import cv2
import numpy as np
import os
import shutil
from datetime import datetime
from evaluator_utils import *
import math as m
import os.path
from pypylon import pylon
import multiprocessing
import os

#png uint 16#

'''
pip install pyrealsense2
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
install realsense viewer to use t265 old firmware


install pylon viewer from
https://www.forecr.io/blogs/connectivity/pylon-installation-for-basler-camera
version linux arm 64 bit 6.3/6.2
sudo pip3 install pypylon


/usr/bin/python3 -m pip install --upgrade pip


#_____________DISABLE GUI: 
If you want to disable the Desktop GUI only temporarily run the following command.
sudo init 3 

To enable the desktop when you finish, run the following command.
sudo init 5 

If you wish to stop Desktop GUI on every reboot, run the following command.
sudo systemctl set-default multi-user.target

To enable GUI again, run the following command.
sudo systemctl set-default graphical.target



'''

#
offset = np.tile(50, (1080,1920))
T265_MANDATORY = False
SEARCH_USB_CAMERAS = False
USE_PYLON_CAMERA = True
now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%H_%M_%_SUM")
SAVE_VIDEO_TIME = 1  # 0 per non salvareTrue
FPS_DISPLAY = True
DISPLAY_RGB = 1
FRAMES_TO_ACQUIRE = 30






def returnCameraIndexes():
    """
    checks the first 10 indexes of cameras usb connected


    :return: an array of the opened camera index
    """
    # checks the first 10 indexes of cameras usb connected
    index = 0
    arr = []
    i = 10
    while i > 0:
        # print("retry cap : ", index)
        try:
            cap = cv2.VideoCapture(index)
        except:
            print("camera index %s not aviable",index)
        # print("cap status :" ,cap.isOpened())

        if cap.isOpened():
            print("is open! index = %s", index)
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    print(arr)
    return arr

def writeCSVdata_generic(name, data):
    """
    write data 2 CSV
    :param data: write to a csv file input data (append to the end)
    :return: nothing
    """
    # scrive su un file csv i dati estratti dalla rete Neurale


    file = open(name, 'a')
    writer = csv.writer(file)
    writer.writerow(data)
    file.close()

def calculate_and_save_intrinsics(intrinsics):

    #print("intrinsics create...", intrinsics, type(intrinsics))

    title = "intrinsics.csv"

    if not os.path.exists(title):
        int = [intrinsics.width, intrinsics.height, intrinsics.ppx, intrinsics.ppy, intrinsics.fx, intrinsics.fy, intrinsics.model, intrinsics.coeffs]
        writeCSVdata_generic(title, int)
        print("new file intrinsics written")
        print(int)


def organize_video_from_last_acquisition():
    try:
        path_dir = "aquisition_raw/"
        if not os.path.exists(path_dir):
            # Create the folder if it doesn't exist
            print("path not present:",path_dir, ", created!")
            os.makedirs(path_dir)

        #create directory to contain file

        name1 = "aquisition_"


        # convert to string

        folder_name = path_dir  + name1 + date_time

        print("folder dest:", folder_name)


        current_directory = os.getcwd()

        file_found = []
        for file in os.listdir(current_directory):

            if file.endswith(".mkv"):

                print("file found:",os.path.join(current_directory, file))
                file_found.append(file)

        os.makedirs(folder_name)

        for f in file_found:

            source = os.path.join(current_directory, f)
            destination = os.path.join(current_directory,folder_name)
            shutil.move(source, destination)
            print(source," moved to : ",destination)


    except Exception as e:
        print("error saving files in folders...",e)



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
        if keys[i] == "D435I" or keys[i] == "D435":
            print("found: ", keys[i])
            enable_D435i = True
        elif keys[i] == "T265":
            enable_T265 = True
        else:
            print("camera not recognized")

    print("D435i ok ? => ", enable_D435i, "___|||____  T265 ok ? => ", enable_T265)

    return enable_D435i, enable_T265, device_aviable



def RS_capture(queue,status):

    check_folder("/data/")




    print("FOLDER ORGANIZED COMPLETED!")
    ##config.enable_device('947122110515')

    ctx = rs.context()
    enable_D435i, enable_T265, device_aviable = search_device(ctx)

    print("D435:", enable_D435i, " | T265:", enable_T265)

    # D435____________________________________________
    if enable_D435i:
        """
        # Declare pointcloud object, for calculating pointclouds and texture mappings
        pc = rs.pointcloud()
        # We want the points object to be persistent so we can display the last cloud when a frame drops
        points = rs.points()
        """
        pipeline = rs.pipeline(ctx)
        config = rs.config()
        try:
            seriald435 = str(device_aviable['D435I'][0])
        except:
            print("no d435i try classic model d435")
            seriald435 = str(device_aviable['D435'][0])

        print("serial : ", type(seriald435))
        config.enable_device(seriald435)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        # We'll use the colorizer to generate texture for our PLY

        # config.enable_stream(rs.stream.accel,rs.format.motion_xyz32f,200)
        # config.enable_stream(rs.stream.gyro,rs.format.motion_xyz32f,200)
        saver = rs.save_single_frameset()
        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            # Start streaming
            pipeline.start(config)
            # colorizer = rs.colorizer()
            print("D435I START OK ___________")
        except Exception as e:
            print("ERROR ON START D435:||||:: %s", str(e))
        # _________________________________________________

    if enable_T265:
        # T265_________________________________________________

        pipelineT265 = rs.pipeline(ctx)
        configT265 = rs.config()
        serialt265 = str(device_aviable['T265'][0])
        print(serialt265)
        configT265.enable_device(serialt265)
        configT265.enable_stream(rs.stream.pose)
        configT265.enable_stream(rs.stream.gyro)
        print("configured succesfully T265...")

        # saver.set_option()

        try:
            # Start streaming
            started = pipelineT265.start(configT265)
            print("T265 started OK",started)
        except Exception as e:
            print("error pipeline T265 starting:||||:: %s", str(e))
        # _______________________________________________________
    else:
        print("no T265 MODE")

    if SAVE_VIDEO_TIME != 0:
        if enable_D435i:
            gst_out = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=RGB.mkv "
            out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER,  20.0, (1900, 1080))

            #gst_out_depth   = "appsrc ! video/x-raw, format=GRAY ! queue ! videoconvert ! video/x-raw,format=GRAY ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=DEPTH.mkv "
            gst_out_depth = "appsrc caps=video/x-raw,format=GRAY8 ! videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! h265parse ! filesink location=DEPTH.mkv "
            #gst_out_depth = ("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=test.mkv" )
            #gst_out_depth = ('appsrc caps=video/x-raw,format=GRAY8,width=1920,height=1080,framerate=30/1 ! '' videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! ''h265parse ! filesink location=test.h265 ')
            out_depth = cv2.VideoWriter(gst_out_depth, cv2.CAP_GSTREAMER,  20.0, (1920, 1080), 0)




    else:
        print("NO SAVE VIDEO D435 MODE")
    frame_c = 0

    print("START LOOP")

    for i in range(FRAMES_TO_ACQUIRE):
        if enable_T265 or enable_D435i:
            start = time.time()
            frame_c += 1

            if enable_T265:
                try:
                    tframes = pipelineT265.wait_for_frames()
                except Exception as e:
                    print("ERROR T265 wait4fr: %s", e, "object ideally not present",started)
                    #started = pipelineT265.start(configT265)
                    #tframes = pipelineT265.wait_for_frames()



                    pose = 0
                try:
                    pose = tframes.get_pose_frame()

                except Exception as e:
                    print("ERROR T265 getFr: %s", e)
                    pose = 0

                if pose:
                    data = pose.get_pose_data()
                    w = data.rotation.w
                    x = -data.rotation.z
                    y = data.rotation.x
                    z = -data.rotation.y

                    pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi;
                    roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z) * 180.0 / m.pi;
                    yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) * 180.0 / m.pi;
                    anglePRY = [pitch, roll, yaw]

                    # print("Frame #{}".format(pose.frame_number))
                    # print("Position: {}".format(data.translation))
                    # print("Velocity: {}".format(data.velocity))
                    # print("Acceleration: {}\n".format(data.acceleration))
                    now = datetime.now()
                    time_st = now.strftime("%d-%m-%Y|%H:%M:%S")
                    writeCSVdata(date_time, [frame_c, time_st, data.translation, data.velocity, anglePRY])
                    if not enable_D435i:
                        #converte la velocita di salvataggio dai 1500 FPS (T265 standalone)  ad un acquisizione piu realistica (15 FPS della D435)
                        time.sleep(0.067)

            if enable_D435i:
                # Wait for a coherent pair of frames: depth and color

                try:
                    frames = pipeline.wait_for_frames()


                except Exception as e:
                    print("PIPELINE error:||||:: %s", str(e))
                    sys.exit()

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if frame_c < 3:
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                    calculate_and_save_intrinsics(depth_intrin)
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                print("DEPTH before", depth_image.shape)



                # width = int(1920)
                # height = int(1080)
                # dim = (width, height)
                #
                # # resize image depth to fit rgb
                # print("DEPTH before", depth_image.shape)
                # resized = cv2.resize(depth_image, dim, interpolation=cv2.INTER_AREA)
                # print("DEPTH after", resized.shape)
                # convert u16 mm bw image to u16 cm bw
                resized = depth_image / 10
                # rescale without first 50 cm of offset unwanted
                resized = resized - offset

                # stretchin all in the 0-255 cm interval
                maxi = np.clip(resized, 0, 255)
                # convert to 8 bit
                intcm = maxi.astype('uint8')

                # print("RGB", color_image.shape)
                print("DEPTH", intcm.shape)

                if SAVE_VIDEO_TIME != 0:
                    try:
                        out.write(color_image)


                    except Exception as e:
                        print("error save video:||||:: %s", str(e))

                    try:
                        # save here depth map
                        out_depth.write(intcm)

                    except Exception as e:
                        print("error saving depth 1 ch:||||:: %s", str(e))

            if FPS_DISPLAY:
                end = time.time()
                seconds = end - start
                fps = round(1 / seconds, 3)
                print(fps)

            if DISPLAY_RGB:
                # cv2.imshow('depth Stream', color_image)
                cv2.imshow('dept!!!h Stream', intcm)

                key = cv2.waitKey(1)
                if key == 27:
                    # result.release()
                    # cv2.destroyAllWindows()
                    break
        else:
            print("no realsense error!!!!")
            time.sleep(5)

        if enable_T265 == False and enable_D435i == False :
            print("no device, termination...")
            sys.exit()

    if enable_D435i:
        pipeline.stop()
        try:
            out.release()
        except Exception as e:
            print("ERROR RELEASE RGB: %s", str(e))
        try:
            out_depth.release()
        except Exception as e:
            print(out_depth)
            print(intcm.shape)
            print("ERROR RELEASE DEPTH: %s", str(e))

        cv2.destroyAllWindows()
        print("closing object...")
        time.sleep(2)
    if enable_T265:
        pipelineT265.stop()
    print("TERMINATING RS CAPTURE")
    status.value = 0
    time.sleep(2)
    sys.exit()



def main(q,status):

    """
    functioon to acquire images from basler dart camera with pylon

    :param q: queue where to put the images extracted
    :param status: status to control the events of the process
    :return: nothing
    """
    config_file = "cfg_file.txt"


    if USE_PYLON_CAMERA:
        # conecting to the first available camera
        try:

            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            #lo usa la cri vediamo a che serve
            camera.Open()

            print('Using device: ', camera.GetDeviceInfo().GetModelName())
            try:
                pylon.FeaturePersistence.Load(config_file, camera.GetNodeMap(), True)
                #pylon.FeaturePersistence.Save(config_file, camera.GetNodeMap())
            except Exception as e:

                print("basler failed load config", e)
                print("basler failed", config_file)
                status.value = 0

            #

            # Grabing Continusely (video) with minimal delay
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            converter = pylon.ImageFormatConverter()


            # converting to opencv bgr format
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            # Set video resolution
            frame_width = 2592
            frame_height = 1944
            size = (frame_width, frame_height)

            # result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, size)

            basler_presence = True
            status.value = 1
            print("basler configured")
        except Exception as e:
            basler_presence = False
            status.value = 0
            print("basler failed", e)


        print("BASLER P:",basler_presence," | BASLER S:",status.value)

    else:
        print("NO BASLER MODE")




    frame_c = 0

    print("START LOOP")

    for i in range(FRAMES_TO_ACQUIRE):
        start = time.time()
        if USE_PYLON_CAMERA:
            if  status.value == 0:
                break
            else:
                frame_c += 1

                # T265

                try:
                    if basler_presence:
                        if camera.IsGrabbing():

                            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                            if grabResult.GrabSucceeded():
                                # Access the image data
                                image = converter.Convert(grabResult)
                                img_basler = image.GetArray()

                                if SAVE_VIDEO_TIME != 0:
                                    try:
                                        q.put(img_basler)

                                    except:
                                        print("error save basler")


                            else:
                                print("camera not succeded, no image")
                                status.value = 0
                        else:
                            print("camera is not grabbing")
                            status.value = 0
                except Exception as e:
                    print("ERROR basler in loop wait4fr: %s", e)
                    basler_presence = False
                    status.value = 0


            key = cv2.waitKey(1)
            if key == 27:
                #result.release()
                #cv2.destroyAllWindows()
                break


        if basler_presence == False:
            print("no device, termination...")
            sys.exit()

    if basler_presence:

        cv2.destroyAllWindows()

def image_saver(q,basler_status):
    """

    :param q: multiprocessing queue, here there are all the image , one by one, aquired by the basler dart pylon process
    :param basler_status: status associated with the process event
    :return: nothing

    """


    time.sleep(1)
    if USE_PYLON_CAMERA:
        while True:

            if basler_status.value == 1:

                print("saving:", basler_status.value)
                frame_width = 2592
                frame_height = 1944




                gst_out_BASLER = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=RGB_BAS.mkv "
                out_BASLER = cv2.VideoWriter(gst_out_BASLER, cv2.CAP_GSTREAMER, 10, (frame_width, frame_height))
                while True:
                    qsize = q.qsize()
                    print("size: ", qsize)
                    img_basler = q.get()
                    out_BASLER.write(img_basler)


                out_BASLER.release()
            else:
                print("NO saving:", basler_status.value)
                time.sleep(1)




def observer(status):

    try:
        while True:
            time.sleep(0.001)

    except KeyboardInterrupt:
        print(' KeyboardInterrupt- AB_main_PC Killed by user, exiting...{} '.format(datetime.now()))
        print("OBSERVER ZERO")
        status.value = 0




def processor():
    try:

        organize_video_from_last_acquisition()

        status = multiprocessing.Value("i", 0)
        q = multiprocessing.Queue(maxsize=1000)
        q_reals = multiprocessing.Queue(maxsize=1000)
        status_realsense = multiprocessing.Value("i", 0)

        p1 = multiprocessing.Process(target=main, args=(q,status))
        p2 = multiprocessing.Process(target=image_saver, args=(q,status))
        #p3 = multiprocessing.Process(target=observer, args=(status,))
        p4=  multiprocessing.Process(target=RS_capture, args=(q_reals,status_realsense))

        p1.start()
        print("main started")
        p2.start()
        #p3.start()
        p4.start()

        p1.join()
        p2.join()
        #p3.join()
        p4.join()


        # both processes finished
        print("Both processes finished execution!")

        # check if processes are alive
        # controllo se sono ancora vivi o se sono terminati e ne printo lo status
        print("MAIN is alive? -> {}".format(p1.is_alive()))
        print("SAVER is alive?    -> {}".format(p2.is_alive()))
        print("REALSENSE is alive?    -> {}".format(p4.is_alive()))
    except KeyboardInterrupt:
        print(' KILLED ..{} '.format(datetime.now()))
        print("STATUS PROCESSOR ZERO")
        time.sleep(1)
        status.value = 0
        sys.exit()






processor()


