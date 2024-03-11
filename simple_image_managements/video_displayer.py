import numpy as np
import cv2 as cv
from additional_functions import *
import sys


def segment_white_area(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get binary mask of white areas
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # Apply morphology closing to close small gaps or points in the mask

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply dilation to further enhance and enlarge white regions
    kernel_dilate = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, kernel_dilate, iterations=2)





    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)




    # Create a white image
    white = np.full_like(frame, (255, 255, 255), dtype=np.uint8)

    # Fill the white areas with solid white
    white_area = cv2.bitwise_and(white, white, mask=mask)

    # Keep the non-white areas of the original frame
    non_white_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine the white area and non-white area images
    result = cv2.add(white_area, non_white_area)

    return result


filename = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition_raw/GX010095.MP4"

save_folder = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/example_photogrammetry'
cap = cv.VideoCapture(filename)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0
while True:
    i = i+1
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    frame = segment_white_area(frame)


    # Display the resulting frame

    a = count_files_in_folder(save_folder)
    print(frame.shape)
    if i % 3 == 0:
        cv2.imwrite(save_folder + '/GOPRO_' + str(a) + '.jpg', frame)
    #cv.imshow('frame', resize_image(frame,30))
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    if key == ord('s'):
        print("s")

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()