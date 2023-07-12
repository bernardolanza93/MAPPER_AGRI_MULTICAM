

import cv2
import numpy as np
import os


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


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video1_path = 'video_dave/F3_6-FN3_ev0.h264'
cap = cv2.VideoCapture(video1_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fps = float(cap.get(cv2.CAP_PROP_FPS))
video_name = os.path.basename(video1_path)  # Get the file name with extension
video_name_without_ext = os.path.splitext(video_name)[0]  # Remove the extension
print(video_name_without_ext)
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(str(video_name_without_ext) + "_converted.avi", fourcc, fps,
                      size)  # setta la giusta risoluzionw e fps

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        print(frame.shape)
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        out.write(frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
out.release()
cap.release()
cv2.destroyAllWindows()

# Closes all the frames