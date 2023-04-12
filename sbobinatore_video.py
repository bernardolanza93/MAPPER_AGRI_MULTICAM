import cv2
import os

# specify the path to the global folder containing the video folders
input_folder_path = "/home/mmt-ben/Downloads/video-20230403T100325Z-001/video/"

# specify the path to the folder where output images will be saved
output_folder_path = "/home/mmt-ben/Documents/dataset_cell/"

# create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    print("created missing output folder")

# loop through each folder in the input folder
for folder_name in os.listdir(input_folder_path):
    print(folder_name)
    folder_path = os.path.join(input_folder_path, folder_name)

    # check if the folder name starts with "acquisition_time_"
    if folder_name.startswith("VID"):
        print(folder_path)

        # loop through each video file in the folder
        for filename in os.listdir(folder_path):

            # check if the file is a video file in .mkv format
            if filename.endswith(".mp4"):
                filepath = os.path.join(folder_path, filename)

                # open the video file using OpenCV
                cap = cv2.VideoCapture(filepath)
                frame_count = 0

                # loop through each frame in the video
                while cap.isOpened():
                    ret, frame = cap.read()

                    # check if there are still frames left in the video
                    if ret:
                        # construct the filename for the output image
                        output_filename = "frame_" + str(folder_name) +"_" + str(frame_count).zfill(6) + ".jpg"

                        output_filepath = os.path.join(output_folder_path, output_filename)

                        # save the current frame as a JPEG image
                        if (frame_count % 12 == 0):
                            print(output_filename)
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                            cv2.imwrite(output_filepath, frame)

                        # increment the frame count
                        frame_count += 1
                    else:
                        break

                # release the video capture object
                cap.release()

print("Done!")