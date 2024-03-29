

# MAPPER_AGRI_MULTICAM

acquisition and mapping system for agricoltural infield analisys.
Programme for assessing vine growth index using a vision system.

## Description

This repository contains both acquisition and analysis software for optical measurements in agriculture. 
The main.py file runs a program capable of acquiring and saving images from an INTEL Realsense D435 camera, via an NVIDIA Jetson Nano card.
The media_mapper_evaluator file and the associated evaluator_utils file (containing the grafted functions) allow the analysis of the media produced by the main, which is carried out on a PC. We are therefore able to produce geometric and volumetric measurements from the acquisitions and estimate the woody biomass within our acquisitions.

## TODO COMPLETE SOON : GSTREAMER SUPPORT ON OPENCV INSTALLATION
### now discovering

## Executable:

### browse directory to execute bash files
```
cd MAPPER_AGRI_MULTICAM/
```

### before (if not performed) give the writing reading permission to the executable file:

```
chmod +x run_test_all_cam.sh
```
### Execute files to run specific programs and update software

```
./run_test_all_cam.sh
./run_calibration_camera.sh
./run_calibration_no_capture.sh
./run_odometry.sh
./run_NO_ARUCO_odometry.sh
./run_vision.sh
```

## To close all process in backgroud digit:
```
pkill python
```

## Generate requirements.txt:
```
cd MAPPER_AGRI_MULTICAM/
pip freeze > requirements.txt
```

## Install requirements.txt:
```
cd MAPPER_AGRI_MULTICAM/
pip install -r requirements.txt
```
## Firstly, review what packages are installed on your Jetson Nano.
```
pip3 list
```

## Configuring sensors in the field  

The camera should be positioned at a fixed distance from the row so that it completely frames the pruning area. It is necessary to estimate:

-the pruning belt (z-axis from the end of the trunk to the maximum height of branch development)

-VerticalFOV of the D435i camera 

The distance from the row will depend on these two parameters ( Df ~ Hfov , Lfp)

## Data saving protocol

Row data will be saved by means of two video streaming matrices, one RGB (3-channel) and one DEPTH (3-channel to be converted back to monochannel depth). The video labed will be 
-timestamped (it will give the reading order [e.g. min_sec_millsec]) and

-with the x,y,z coordinates from the T265 camera (x_y_z) saved to a separate csv file.

-to use the data from the T265, an absolute reference system must be created using an ARUCO MARKER to know the starting position with very good accuracy

## Segmentation processing

The data acquired in the laboratory on a fixed background (without T265) are segmented.
A single-channel B&W colour mask containing wood volume pixels is extracted.



Translated with www.DeepL.com/Translator (free version)

## Data analysis

Volume measurements will be taken (starting with static pixel/volume calibration with stationary samples equidistant from the optics). Volumetric measurements will be taken using a graduated reservoir to find out the true volume of the samples. RMS values will be evaluated from a static acquisition and comparison with the reference. We will also evaluate the STD-DEV by assessing how the measurement is affected by light effects in the static field. 
These tests will be extended to more critical conditions, such as:

-Decreasing resolution of the measurand (moving the sample away from the chamber) Finding the optimal chamber measuring distance

-increased measurand depth variability (sample oriented in complex positions at different depths)

## Data visualisation protocol

DATASET AUGMENTATION
Using fixed bg lab row acquisitions of one or more shoots, the resulting mask will be used to generate the DNN training label. Post-pruning acquisitions with the lab acquisition (photomontage style) of only the segmented rgb pixels will be used as input.
DATASET CREATION 
we will use the row acquisitions as input, the difference with the segmented post pruning images will give us the label for the DNN
We will make many acquisitions, with a fixed camera, in which we will first put a white background cloth, and then reacquire without the cloth so that we have automatic segmentation info (the video can keep going to avoid moving the camera)
DATASET FROM DARK
Acquisition at night with illuminator, controlled bg and possibility to effectively segment as many real row data. The network will be trained in night conditions 

## Management of depth data.
Once a mask of pixels has been obtained (in the lab without bg, or in the vineyard with DNN), the depth data for those pixels will be extracted and
average (n pixels at an average distance of d)
you will convert each pixel already in mm^2 according to its distance from the chamber and immediately calculate the actual volume 
Or without segmentation you will estimate the distance to the vineyard row with (RANSAC/ARUCO[depth technique or RGB])

Translated with www.DeepL.com/Translator (free version)



## Getting Started

### Dependencies

* Librealsense is the foundamental library for acquisition, to acquire also the T265 camera realsese viewer is needed to fullfill all the requisites of the old T265 firmware
* Nvidia Jetpack, python 3.9, Opencv >4.6

### Installing

* We can provide you a requirementes.txt file if needed
* matlab script are customized to fit my laptop, other script are relative to the working directory, automatically filling the absolute path with your WD.

```

sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
```

## INSTALL PYREALSENSE FOLLOWING THIS REPO
### install SDK to utilize T265, and instal pyrealsense2 to import RS libraries
```
https://github.com/35selim/RealSense-Jetson/tree/main
```

### On python to see the opencv version installed:
```
print (cv2.__version__)
print cv2.getBuildInformation()
```

### Before installing opencv you need some requirements and purge old version:
```
echo "** Remove other OpenCV first"
sudo sudo apt-get purge *libopencv*


echo "** Install requirement"
sudo apt-get update
sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y python2.7-dev python3.6-dev python-dev python-numpy python3-numpy
sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp
sudo apt-get install -y curl
```

### INSTALL OPENCV NOT OPTIMIZED (NO CUDA NO GSTREAM):
```
sudo apt-get install python3-opencv
```
### Use 
```
sudo apt-get remove -y
```
### Or for specific:
```
pip uninstall opencv-python==4.5.1
```
### to uninstall and 
```
sudo apt list --installed
```
### to list installed packages.

### Other way to uninstall:
```
sudo sudo apt-get purge *libopencv*
```
### To uninstall manually (not suggested)
```
sudo find / -name " *opencv* " -exec rm -i {} \;
```
### with above command, I find the opencv related folders and files, and remove them.


### After removing process, I checked the version using following command if opencv still exist.
```
pkg-config --modversion opencv
```

## INSTALL OPENCV FROM SOURCE:
### GUIDE:

https://qengineering.eu/install-opencv-on-jetson-nano.html

### REPO FOR AUTOMATED SOURCE INSTALLATION:

https://github.com/Qengineering/Install-OpenCV-Jetson-Nano

### Here you can see another guide to install opencv on jetson by source (backup guide jut for help)

https://github.com/AastaNV/JEP/tree/master/script

## INSTALL OPEN GOPRO (Ubuntu 64bit pc X86):
### ARM 64 linux todo
```
pip install open-gopro
```
### ToDo: pylon viewewr and pypylon library installation. SDK software to display and python library to acquire. 
### search for ARM 64 linuz tar.gz pylon software
```
https://www2.baslerweb.com/en/downloads/software-downloads/
```



## Executing program

* How to run the program
* Step-by-step bullets
```
python3.9 media_mapper_evaluator.py
python3.9 main.py
```

## Improve Performance:
* improve power mode
```
sudo nvpmodel -m 0
sudo jetson_clocks
```
## To be Tested
If you want to disable the Desktop GUI only temporarily run the following command.
```
sudo init 3 
```
To enable the desktop when you finish, run the following command.
```
sudo init 5
```
If you wish to stop Desktop GUI on every reboot, run the following command.
```
sudo systemctl set-default multi-user.target
```
To enable GUI again, run the following command. 
```
sudo systemctl set-default graphical.target
```

to start Gui session on a system without a current GUI just execute:
```
sudo systemctl start gdm3.service
```



You can test the device with:
```
tegrastats
```
or 
```
jtop
```
to see if the GPU resources are fully utilized

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. [@BernardoLanza]([https://www.linkedin.com/in/bernardo-lanza-554064163/])

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release



