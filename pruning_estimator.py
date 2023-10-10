
from scipy.spatial import distance as dist
import os
import cv2
import math
import time
import numpy as np
import statistics
import pyrealsense2 as rs
import csv
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import fnmatch
import shutil
from misc_utility_library import *


#SETTARE QUI I PARAMETRI PER LA TUA MACCHINA
#________________________________________
#FOLDER DOVE VENGONO SALVATI I DATI DENTRO LA DIRECTORY DEL CODICE
folder_data = "tesi"
# DEFINIRE LA POSIZIONE DELLA CARTELLA CONTENETE I FILE DELLE ACQUISIZIONI (mettere solo la cartella padre omnicomprensiva)
root_folder = '/home/mmt-ben/Documents/MAIN'
#DATI REALI DEL CARTONCINO IN MM
L_real = 315.0
D_real = 72.2

print(sys.version)



THRES_VALUE = 70
PATH_2_AQUIS = "/home/mmt-ben/Documents/calibration_video/"
PATH_HERE = os.getcwd()
OFFSET_CM_COMPRESSION = 50


def save_video_geometrical_data(csv_file,data_frame):
    header = ["nrfr","Z_val","cx","cy","rl","rd","ar","alpha"]
    # Specify the CSV file name


    # Write data to the CSV file row by row
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the custom header to the CSV file
        writer.writerow(header)

        # Write the rows to the CSV file
        for row in data_frame:
            writer.writerow(row)
    print(f"Data has been saved to {csv_file}.")

def geometry_evaluator(mask,frame):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure at least one contour was found
    if len(contours) > 0:
        # Find the largest contour (the rectangular shape)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the rotated rectangle that bounds the contour
        rect = cv2.minAreaRect(largest_contour)

        box = cv2.boxPoints(rect)  # cv2.cv.BoxPoints(rect) for OpenCV <3.x
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        # Extract dimensions, center, and angle of rotation
        (center_x, center_y), (width, height), angle = rect

        # Calculate the aspect ratio


        # Calculate the orientation angle in degrees with respect to the x-axis
        if width > height:
            orientation_degrees = angle
            aspect_ratio = height/width

        else:
            orientation_degrees = 90 + angle
            aspect_ratio = width / height

        if orientation_degrees >= 180:
            orientation_degrees = orientation_degrees-180

        # Print the results
        cx = center_x
        cy = center_y
        l = max(width, height)
        d = min(width, height)
        ar = aspect_ratio
        alpha = orientation_degrees

        ratio_l = L_real / l
        ratio_d = D_real / d

        #print("dim:", l,d,"real", L_real,D_real, "ratio", ratio_l,ratio_d)


        # print(f"Center (X, Y): ({center_x}, {center_y})")
        # print(f"Major Dimension: {max(width, height)}")
        # print(f"Minor Dimension: {min(width, height)}")
        # print(f"Aspect Ratio: {aspect_ratio}")
        # print(f"Orientation (degrees): {orientation_degrees}")
        return int(cx),int(cy),ratio_l,ratio_d,ar,alpha

    else:

        print("No contours found in the mask.")




def strech_hist_of_gray_image_4_contrast(gray_raw):
    # Define the desired minimum and maximum intensities for the output
    desired_min = 0  # The minimum intensity after adjustment
    desired_max = 255  # The maximum intensity after adjustment

    # Calculate the current minimum and maximum intensities
    current_min = np.min(gray_raw)
    current_max = np.max(gray_raw)

    # Perform the linear contrast adjustment
    gray = (gray_raw - current_min) * (
            (desired_max - desired_min) / (current_max - current_min)) + desired_min

    # Convert the adjusted image to 8-bit unsigned integer format
    gray = np.uint8(gray)
    return gray



def crop_with_rect_rot(frame,rect, display = False):
    inw = frame.shape[1]
    inh = frame.shape[0]
    if inw > inh:
        widthside = True
    else:
        widthside = False




    # # margin augmented
    # frame = cv2.copyMakeBorder(src=frame, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT,
    #                           value=(255))



    box = cv2.boxPoints(rect)
    box = np.int0(box)


    # crop rot box + margin white
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)




    if display:
        cv2.imshow("frameAFT", warped)

    succesful = True

    inw = warped.shape[1]
    inh = warped.shape[0]
    if widthside:
        if inw < inh:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    else:
        if inh < inw:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)


    return warped





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

def obtain_intrinsics():
    intrinsics = rs.intrinsics()
    with open("intrinsics.csv", "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                intrinsics.width = int(line[0])
                intrinsics.height = int(line[1])
                intrinsics.ppx = float(line[2])
                intrinsics.ppy = float(line[3])
                intrinsics.fx = float(line[4])
                intrinsics.fy = float(line[5])

                if str(line[6]) == "distortion.inverse_brown_conrady":
                    intrinsics.model = rs.distortion.inverse_brown_conrady
                else:
                    print("not rec ognized this string for model: ", str(line[6]))
                    intrinsics.model = rs.distortion.inverse_brown_conrady

                listm = line[7].split(",")

                new_list = []
                for i in range(len(listm)):
                    element = listm[i]
                    element = element.replace("[", "")
                    element = element.replace(" ", "")
                    element = element.replace("]", "")
                    element = float(element)
                    new_list.append(element)

                intrinsics.coeffs = new_list

    return intrinsics




def second_layer_accurate_cnt_estimator_and_draw(mask_bu, frame):
    imagem1 = (255 - mask_bu)

    contours1, hierarchy1 = cv2.findContours(imagem1, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    i = 0
    for cnt1 in contours1:
        # calcolo area e perimetro
        area1 = cv2.contourArea(cnt1)
        h = mask_bu.shape[0]
        w = mask_bu.shape[1]

        area_box = h * w
        ratio = area1 / area_box

        # perimeter
        perimeter1 = cv2.arcLength(cnt1, True)
        i += 1

        if perimeter1 > 200 and area1 > 200:
            # calcolo circolarità
            circularity1 = (4 * math.pi * area1) / (pow(perimeter1, 2))
            M = cv2.moments(cnt1)
            # centroids
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            M0 = M['m00'] / 1000
            M01 = M['m01'] / 1000000
            M10 = M['m10'] / 1000000
            M02 = M['m02'] / 1000000000
            M20 = M['m20'] / 1000000000

            hull = cv2.convexHull(cnt1)
            hull_area = cv2.contourArea(hull)
            solidity = float(area1) / hull_area

            if perimeter1 > 200 and perimeter1 < 7000:  # 1200
                if circularity1 > 0.20 and circularity1 < 0.6:  # 0.05 / 0.1, 0.02
                    if area1 > 800 and area1 < 150000:  # 2200
                        if M0 > 1.15 and M0 < 130:  # 2200
                            if M01 > 0.40 and M01 < 70:  # 2200
                                if M10 > 0.5 and M10 < 140:  #
                                    if M02 > 0.25 and M02 < 40:
                                        if M20 > 0.002 and M20 < 150:
                                            if solidity > 0.95 and solidity < 0.999:
                                                if ratio > 0.0005 and ratio < 0.8:  # rapporto pixel contour e bounding box

                                                    # print("|____________________________________|")
                                                    # print("__|RATIO-CORRECT|__:", ratio)
                                                    print("|____________________|2nd LAYER CHOSEN", i, " area:",
                                                          int(area1), " perim:", int(perimeter1), " circul:",
                                                          round(circularity1, 5))
                                                    print(" M0:", round(M0, 3), " M01:", round(M01, 3), " M10:",
                                                          round(M10, 3), " M02:", round(M02, 3), " M20:", round(M20, 5),
                                                          "ratio", round(ratio, 5), " solidity", round(solidity, 4))
                                                    cv2.drawContours(frame, [cnt1], 0, (200, 200, 50), 1)

                                                    return cnt1, frame, True
    print("________!!!!____________advanced shoots not detected")
    print("len contours", len(contours1))

    for cnt1 in contours1:
        # calcolo area e perimetro
        area1 = cv2.contourArea(cnt1)
        # perimeter
        perimeter1 = cv2.arcLength(cnt1, True)
        if perimeter1 > 200 and area1 > 200:
            # calcolo circolarità
            circularity1 = (4 * math.pi * area1) / (pow(perimeter1, 2))
            M = cv2.moments(cnt1)
            # centroids
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            M0 = M['m00'] / 1000
            M01 = M['m01'] / 1000000
            M10 = M['m10'] / 1000000
            M02 = M['m02'] / 1000000000
            M20 = M['m20'] / 1000000000
            print("|!||!|!|!|!|!|!|!|!|!|!|! 2nd LAYER CANDIDATE", i, " area:", int(area1), " perim:", int(perimeter1),
                  " circul:", round(circularity1, 5))
            print(" M0:", round(M0, 3), " M01:", round(M01, 3), " M10:", round(M10, 3), " M02:", round(M02, 3), " M20:",
                  round(M20, 5), "ratio", round(ratio, 5), " solidity", round(solidity, 4))
            h = mask_bu.shape[0]
            w = mask_bu.shape[1]

            area_box = h * w
            ratio = area1 / area_box
            print("__|RATIO|__:", ratio)

    # time.sleep(1000)
    return 0, frame, False





#testing system




# Function to find .mkv files in a folder
def find_mkv_files(folder):
    mkv_files = []
    for root, dirs, files in os.walk(folder):
        for filename in fnmatch.filter(files, '*.mkv'):
            mkv_files.append(os.path.join(root, filename))
    return mkv_files





# Specify the folder name you want to create


# Check if the folder doesn't exist, then create it
if not os.path.exists(folder_data):
    os.makedirs(folder_data)
    print(f"Folder '{folder_data}' has been created.")
else:
    print(f"Folder '{folder_data}' already exists.")

intrinsics = obtain_intrinsics()

# Loop through folders and subfolders
for root, dirs, files in os.walk(root_folder):
    if 'RGB.mkv' in files and 'DEPTH.mkv' in files:
        folder_ext = os.path.basename(root_folder)  # Get the external folder name
        folder_int = os.path.relpath(root, root_folder)  # Get the internal folder name
        folder_int = folder_int.replace(os.path.sep, '_')  # Replace path separators with underscores

        # Combine external and internal folder names to create the final folder name
        folder_name = f'{folder_ext}_{folder_int}'


        # Retrieve the .mkv files
        rgb_mkv_file = os.path.join(root, 'RGB.mkv')
        depth_mkv_file = os.path.join(root, 'DEPTH.mkv')
        #print(rgb_mkv_file)

        video1 = cv2.VideoCapture(rgb_mkv_file)

        video2 = cv2.VideoCapture(depth_mkv_file)

        # We need to check if camera
        # is opened previously or not
        if (video1.isOpened() == False):
            print("Error reading video rgb")
        if (video2.isOpened() == False):
            print("Error reading video depth")

        frame_width = int(video1.get(3))
        frame_height = int(video1.get(4))
        frame_width2 = int(video2.get(3))
        frame_height2 = int(video2.get(4))
        size = (frame_width, frame_height)
        size2 = (frame_width2, frame_height2)
        nrfr = 0

        dataframe = []

        while (video1.isOpened() and video2.isOpened()):

            ret, frame = video1.read()
            ret2, frame2 = video2.read()
            nrfr = nrfr + 1


            if ret == True and ret2 == True:



                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = strech_hist_of_gray_image_4_contrast(gray_raw)
                ret, mask = cv2.threshold(gray, THRES_VALUE, 255, cv2.THRESH_BINARY)
                # try:
                cnt1, frame, completion = second_layer_accurate_cnt_estimator_and_draw(mask, frame)

                inverted_mask = 255 - mask
                cx,cy,ratio_l,ratio_d,ar,alpha = geometry_evaluator(inverted_mask,frame)


                #3 spazi immagine
                # frame = crop_with_rect_rot(frame, rect)
                # gray = crop_with_rect_rot(gray, rect)
                # frame2 = crop_with_rect_rot(frame2, rect)
                # mask = crop_with_rect_rot(mask, rect)

                # Ensure that both images have the same dimensions


                # Create a mask where black pixels in the binary mask are False
                mask_proc = inverted_mask != 0

                # Use NumPy to apply the mask to the grayscale image
                masked_image = np.where(mask_proc, frame2, 0)

                # Calculate the mean value of the masked grayscale image
                Z_raw = np.mean(masked_image[masked_image != 0])
                Z_val = OFFSET_CM_COMPRESSION + Z_raw

                data = [nrfr,Z_val,cx,cy,ratio_l,ratio_d,ar,alpha]
                dataframe.append(data)
                print(data)



                if frame.shape[0] == frame2.shape[0] == mask.shape[0]:
                    concatenated_image = np.hstack([frame, cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
                    cv2.imshow("or", resize_image(concatenated_image, 50))
                    #cv2.imshow("fr",resize_image(frame,50))

                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    sys.exit()
                    break
            else:
                break

        video1.release()
        video2.release()
        cv2.destroyAllWindows()




        folder_name = folder_name.replace(" ", "_")
        csv_file = folder_data + "/" +  folder_name + ".csv"
        delete_csv_file(csv_file)
        save_video_geometrical_data(csv_file, dataframe)


        print(f'Saved data for folder: {folder_name}')
        #delete_csv_file(filenam_path)

























