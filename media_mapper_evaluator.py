



import numpy as np
import matplotlib.pyplot as plt

import time

from numpy.linalg import norm
import os

import os
import sys

import cv2


PATH_HERE = os.getcwd()
PATH_2_FILE = "/data/"
PATH_2_AQUIS = "/aquisition/"
SAVE_VIDEO = False


BOT = (0, 8, 11)
TOP = (180, 218, 126)




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



def make_hinstogram_base_plt():

    bins = 16
    resizeWidth = 0

    # Initialize plot.
    fig, ax = plt.subplots()

    ax.set_title('Histogram (HSV)')

    ax.set_xlabel('Bin')
    ax.set_ylabel('Frequency')

    # Initialize plot line object(s). Turn on interactive plotting and show plot.
    lw = 3
    alpha = 0.5

    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='h')
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='s')
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='v')

    ax.set_xlim(0, bins - 1)
    ax.set_ylim(0, 0.08)
    ax.legend()
    plt.ion()
    plt.show()

    return fig,lineR,lineG,lineB

    # Grab, process, and display video frames. Update plot line object(s).






def animate_plot_data(frame, fig,lineR,lineG,lineB):
    bins = 16
    resizeWidth = 0

    #result.write(frame)

    numPixels = np.prod(frame.shape[:2])


    (b, g, r) = cv2.split(frame)
    histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
    histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
    histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
    lineR.set_ydata(histogramR)
    lineG.set_ydata(histogramG)
    lineB.set_ydata(histogramB)

    fig.canvas.draw()














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


def blob_detector(im,min_area_blob):
    start_time3 = time.time()

    im = cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255,255,255])








    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    #params.minThreshold = 10;
    #params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = 0
    params.minArea = 300

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.001

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.001

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.001



    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(im)
    total_count = 0

    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    if len(keypoints)>1:

        for keyPoint in keypoints:
            total_count = total_count + 1
            x = keyPoint.pt[0]
            y = keyPoint.pt[1]
            s = keyPoint.size

            #print("size :",s)
            if s > min_area_blob:
                cv2.circle(im, (int(x), int(y)), int(s), (0,255,0), 2)

            else:
                cv2.circle(im, (int(x), int(y)), int(s), (0, 0, 255), 2)


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
        path= os.path.join(PATH_HERE + PATH_2_AQUIS, file)
        video = cv2.VideoCapture(path)


        # We need to check if camera
        # is opened previously or not
        if (video.isOpened() == False):
            print("Error reading video file")

        # We need to set resolutions.
        # so, convert them from float to integer.
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))

        size = (frame_width, frame_height)

        # Below VideoWriter object will create
        # a frame of above defined The output
        # is stored in 'filename.avi' file.
        if SAVE_VIDEO:
            result = cv2.VideoWriter('filename.avi',
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     20, size)

        cv2.namedWindow(window_capture_name)
        cv2.namedWindow(window_detection_name)

        cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
        cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
        cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
        cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
        cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
        cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)



        while (True):
            ret, frame = video.read()

            if ret == True:

                # Write the frame into the
                # file 'filename.avi'
                if SAVE_VIDEO:

                    result.write(frame)


                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                ret, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
                #th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                #th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                #frame_HSV = frame
                #mask = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
                #frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                #mask = cv2.inRange(frame_HSV, BOT, TOP)



                imask = mask < 255
                imagem = (255 - mask)

                # blob_detector(imagem)

                green = 255 * np.ones_like(frame, np.uint8)
                green[imask] = frame[imask]  # dentro i mask metto frame
                #Green,Imagen = mask_generation(frame,BOT,TOP)
                #Imagen = cv2.bitwise_not(Imagen)
                #Imagen = undesired_objects(Imagen)

                # Display the frame
                # saved in the file
                #maskedcolor = resize_image(Green,50)
                #mask = resize_image(Imagen,50)
                #cv2.imshow('Frame', maskedcolor)
                dimension = 100
                frame = resize_image(frame,dimension)
                green = resize_image(green, dimension)

                #cv2.imshow('fff', frame)
                #cv2.imshow('ff', mask)
                #fig,lineR,lineG,lineB = make_hinstogram_base_plt()
                #animate_plot_data(frame, fig, lineR,lineG,lineB)

                # Press S on keyboard
                # to stop the process
                #                                         time.sleep(0.1)
                #if cv2.waitKey(1) & 0xFF == ord('s'):
                    #break
                cv2.imshow(window_capture_name, frame)
                cv2.imshow(window_detection_name, green)

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
