



import numpy as np
import matplotlib.pyplot as plt

import time

from numpy.linalg import norm
import os

import os
import sys
import math
import cv2







PATH_HERE = os.getcwd()
PATH_2_FILE = "/data/"
PATH_2_AQUIS = "/aquisition/"
SAVE_VIDEO = False
TRACKBAR = False
THRESHOLD = True
OPENING = True
PIXEL_COUNTING = True
THRES_VALUE = 75
MASK_DEPTH = False
CONVERT_DEPTH_TO_1CH = False
CROPPING = True
MEDIUM_DEPTH_DISPLAY = True
BLOB_DETECTOR = True


BOT = (0, 8, 11)
TOP = (180, 218, 126)

print("thres_value = ",THRES_VALUE)



def RGBtoD(r, g, b):

    if r >= g and r >= b:
        if g >= b:
            return g - b
        else:
            return (g - b) + 1529
    elif g >= r and g >= b:
        return b - r + 510
    elif b >= g and b >= r:
        return r - g + 1020


def ColorToD(frame):
    min_depth = 0
    max_depth = 10
    depth_frame = np.zeros((frame.shape[0], frame.shape[1]))
    div = max_depth - min_depth
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            r, g, b = frame[i][j][0], frame[i][j][1], frame[i][j][2]
            hue_value = RGBtoD(r, g, b)

            z_value = (min_depth + ((div) * hue_value) / 1529)
            depth_frame[i][j] = z_value/255
            #if z_value != 0:

            #print(z_value/10)

    return depth_frame



def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)



def resize_image(img, percentage):
    # Display the frame
    # saved in the file
    scale_percent = 70  # percent of original size
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2



def mask_generation(frame,bottom_color, top_color):

    #frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv = frame

    ## mask of green (36,25,25) ~ (86, 255,255)
    #mask = cv2.inRange(hsv, (40, 40,40) , (70, 255,255))
    mask = cv2.inRange(hsv, bottom_color, top_color)

    #mask = cv2.inRange(hsv, (25, 15, 15), (100, 255, 255))
    #attive prime
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #mask = cv2.GaussianBlur(mask, (19, 19), 0)
    #mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]

    ## slice the green
    imask = mask > 0
    imagem = (255 - mask)


    #blob_detector(imagem)


    green =  255 * np.ones_like(frame, np.uint8)
    green[imask] = frame[imask]  # dentro i mask metto frame

    return green, imagem

def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)


def blob_detector(im,frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    im = cv2.erode(im, kernel, iterations=1)

    start_time3 = time.time()



    edge = cv2.Canny(im, 175, 175)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    # Find the index of the largest contour
    valid = []
    color = 0
    i = 0
    print("|||||||||||||||||||||||||||||||||||||||||||")
    for cnt in contours:
        i += 1
        print("_________________________")
        #area
        area = cv2.contourArea(cnt)
        #perimeter
        perimeter = cv2.arcLength(cnt, True)
        #fitting a line

        if perimeter != 0 and area != 0:
            circularity = (4 * math.pi * area ) / (pow(perimeter,2))

            print("area and perim and circularity", i, int(area), int(perimeter), circularity)


            if perimeter > 1000:
                if circularity < 0.1:
                    if area > 50  :





                        cv2.drawContours(frame, [cnt], 0, (0 , 255 , 0), 3)

                        rows, cols = frame.shape[:2]
                        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                        lef = int((-x * vy / vx) + y)
                        ri = int(((cols - x) * vy / vx) + y)
                        cv2.line(frame, (cols - 1, ri), (0, lef), (0, 255, 0), 2)
                        #x, y, w, h = cv2.boundingRect(cnt)
                        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        epsilon = 0.1 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                        #print(box)

                        org = (100, 100)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        frame = cv2.putText(frame, str(i), (box[0][0], box[0][1] - 100), font,
                                            fontScale, color, thickness, cv2.LINE_AA)


                        if False:
                            up = 0 + 50
                            down = frame_height - 50
                            left = 0 + 300
                            right = frame_width - 300


                            frame = frame[minx:maxx, miny:maxy]


                            frame = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]

    return im,frame,edge
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                         # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)





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



check_folder("/aquisition/")

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

for file in os.listdir(PATH_HERE + PATH_2_AQUIS):
    print("files:", os.listdir(PATH_HERE + PATH_2_AQUIS))

    if file.endswith(".mkv"):
        print(file.split(".")[0])

        if file.split(".")[0] == "RGB":

            path_rgb= os.path.join(PATH_HERE + PATH_2_AQUIS, file)
            video1 = cv2.VideoCapture(path_rgb)
        elif file.split(".")[0] == "DEPTH":

            path_depth= os.path.join(PATH_HERE + PATH_2_AQUIS, file)
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
print("height : ", frame_height)
print("width : ", frame_width)

size = (frame_width, frame_height)
size2 = (frame_width2, frame_height2)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
if SAVE_VIDEO:
    result = cv2.VideoWriter('filename.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             20, size)
if TRACKBAR:
    cv2.namedWindow(window_capture_name)
    cv2.namedWindow(window_detection_name)

    cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)



while (True):
    ret, frame = video1.read()
    ret2, frame2 = video2.read()

    if ret == True and ret2 == True:

        # Write the frame into the
        # file 'filename.avi'
        if SAVE_VIDEO:

            result.write(frame)

        if CROPPING:
            up = 0 + 50
            down = frame_height -50
            left = 0 +300
            right = frame_width -300
            frame = frame[up:down, left:right]
            frame2 = frame2[up:down, left:right]


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        #th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        #th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        #frame_HSV = frame

        #sistemi di mask generation
        if TRACKBAR:
            mask = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        elif THRESHOLD:
            ret, mask = cv2.threshold(gray, THRES_VALUE, 255, cv2.THRESH_BINARY)
        #frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #mask = cv2.inRange(frame_HSV, BOT, TOP)

        if OPENING:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)




        imask = mask < 255
        imagem = (255 - mask)
        green = 255 * np.ones_like(frame, np.uint8)
        green[imask] = frame[imask]  # dentro i mask metto frame
        if MASK_DEPTH:
            frame22 = 255 * np.ones_like(frame2, np.uint8)
            frame22[imask] = frame2[imask]

        else:
            frame22 = frame2
        if CONVERT_DEPTH_TO_1CH:
            frame22 = ColorToD(frame22)


        if PIXEL_COUNTING:
            org = (100, 100)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2
            number_of_black_pix = np.sum(mask == 0)
            image = cv2.putText(green, 'pix : ' + str(number_of_black_pix), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        #Green,Imagen = mask_generation(frame,BOT,TOP)
        #Imagen = cv2.bitwise_not(Imagen)
        #Imagen = undesired_objects(Imagen)

        # Display the frame
        # saved in the file
        #maskedcolor = resize_image(Green,50)
        #mask = resize_image(Imagen,50)
        #cv2.imshow('Frame', maskedcolor)
        if CONVERT_DEPTH_TO_1CH:
            if MEDIUM_DEPTH_DISPLAY:
                print(frame22)
                #medium = frame22()
        if BLOB_DETECTOR:
            mask,frame,edge = blob_detector(mask, frame)

        dimension = 50
        edge = resize_image(edge,dimension)
        frame = resize_image(frame,dimension)
        green = resize_image(green, dimension)
        mask = resize_image(mask, dimension)
        frame22 = resize_image(frame22, dimension)


        #cv2.imshow('fff', frame)
        #cv2.imshow('ff', mask)
        #fig,lineR,lineG,lineB = make_hinstogram_base_plt()
        #animate_plot_data(frame, fig, lineR,lineG,lineB)

        # Press S on keyboard
        # to stop the process
        #                                         time.sleep(0.1)
        #if cv2.waitKey(1) & 0xFF == ord('s'):
            #break
        cv2.imshow("or", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("green", green)
        cv2.imshow("bella", frame22)
        cv2.imshow("belhla", edge)

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
video.release()
if SAVE_VIDEO:
    result.release()

# Closes all the frames
cv2.destroyAllWindows()


sys.exit()
