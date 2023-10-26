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
import Jetson.GPIO as GPIO
import time




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
config_file = "cfg_file.txt"


#install GPIO library:
# sudo pip install Jetson.GPIO
# sudo groupadd -f -r gpio
#
# sudo usermod -a -G gpio your_user_name


# Set the GPIO pin numbers
button_pin = 31  # Replace with the actual pin number
led_green_pin = 33  # Replace with the actual pin number
led_red_pin = 35  # Replace with the actual pin number
status = 0

# Initial state and LED mapping
led_state = 0  # 0 for led1, 1 for led2
led_pins = [led_green_pin, led_red_pin]



def process_1_GPIO(status):
    print("start")
    status.value = 0

    # Configure the GPIO pins
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    for pin in led_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    try:

        while True:




            if status.value == 0:
                GPIO.output(led_red_pin, GPIO.HIGH)
                GPIO.output(led_green_pin, GPIO.LOW)
            if status.value == 1:
                GPIO.output(led_red_pin, GPIO.LOW)
                GPIO.output(led_green_pin, GPIO.HIGH)
            if status.value == 2:
                for i in range(1, 6):
                    GPIO.output(led_red_pin, GPIO.HIGH)
                    time.sleep(0.2)
                    GPIO.output(led_red_pin, GPIO.LOW)
                    time.sleep(0.2)
                    print("error comeback")




            button_state = GPIO.input(button_pin)
            if button_state == GPIO.LOW:
                print("button premuto!!!")
                # Toggle the value
                if status.value == 0:
                    print("TO GREEN")
                    status.value = 1
                    time.sleep(0.2)
                elif status.value == 1:
                    print("TO RED")
                    status.value = 0
                    time.sleep(0.2)
                else:
                    print("ERROR")
                    sys.exit()



            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    GPIO.cleanup()

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
    name = "odometry"

    file = open('./data/' + name + '_'+ time +'.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(data)
    file.close()
