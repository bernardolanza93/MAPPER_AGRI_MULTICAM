import numpy as np
import cv2 as cv
from additional_functions import *
import sys



filename = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition_raw/GX010054.MP4"

save_folder = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/example_photogrammetry'
cap = cv.VideoCapture(filename)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here


    # Display the resulting frame
    cv.imshow('frame', resize_image(frame,30))
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    if key == ord('s'):
        a = count_files_in_folder(save_folder)
        print(frame.shape)
        cv2.imwrite(save_folder + '/GOPRO_' + str(a) + '.jpg', frame)
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()