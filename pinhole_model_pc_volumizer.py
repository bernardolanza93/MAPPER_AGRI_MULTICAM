from scipy.spatial import distance as dist
import os
import cv2
import math
import time
import numpy as np
import statistics
import pyrealsense2 as rs
import csv
from scipy.spatial import cKDTree
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from CONFIGURATION_VISION import *
from pointcloud_utility_library import *
import open3d as o3d
from sklearn.preprocessing import RobustScaler
import pyntcloud
import pandas as pd
import random
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from misc_utility_library import *



print(sys.version)

#allora invece che stimare d e l stiamo stimando le componenti verticali e orizz di un ramo, che se inclinato a 45 gradi fanno risultare il diametro  = L e quindi sballano di brutto! (


print("PATH! ", PATH_EXAMPLE)


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

def connect_thin_wood(gray_image):

    # Threshold the image to create binary regions
    # Threshold the image to create binary regions
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image to identify blobs
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to store the result
    result_image = gray_image.copy()

    # Initialize a list to store the oriented bounding boxes of blobs
    blob_oriented_bboxes = []

    # Initialize a list to store the mean gray values for each blob
    mean_gray_values = []

    # Iterate through the contours (blobs) and find the oriented bounding box for each blob
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        blob_oriented_bboxes.append(box)

        # Extract and average gray values from inside the detected blob (contour)
        mask = np.zeros_like(gray_image)
        cv2.fillPoly(mask, [contour], 255)
        masked_gray = gray_image * (mask == 255)
        non_zero_values = masked_gray[mask == 255]
        mean_gray = int(np.mean(non_zero_values))
        mean_gray_values.append(mean_gray)


    # Sort the blob_oriented_bboxes by their X coordinate
    blob_oriented_bboxes.sort(key=lambda x: x[0][0])

    # Create polygons from the four points of each pair of bounding boxes
    polygons = []
    for i in range(len(blob_oriented_bboxes) - 1):
        bbox1 = blob_oriented_bboxes[i]
        bbox2 = blob_oriented_bboxes[i + 1]

        # Calculate pairwise distances between vertices
        distances = np.zeros((4, 4))
        for j in range(4):
            for k in range(4):
                distances[j, k] = np.linalg.norm(bbox1[j] - bbox2[k])

        # Find the indices of the nearest vertices for the first pair
        min_indices_first = np.unravel_index(np.argmin(distances), distances.shape)

        # Temporarily set the corresponding distances to a large value
        distances[min_indices_first[0], :] = np.inf
        distances[:, min_indices_first[1]] = np.inf

        # Find the indices of the nearest vertices for the second pair
        min_indices_second = np.unravel_index(np.argmin(distances), distances.shape)

        # Create a polygon from the four points
        polygon = np.array([bbox1[min_indices_first[0]], bbox2[min_indices_first[1]],
                            bbox2[min_indices_second[1]], bbox1[min_indices_second[0]]], dtype=np.int32)
        polygons.append(polygon)

    # Fill the polygons with the mean gray values
    for i, polygon in enumerate(polygons):
        mean_gray = mean_gray_values[i]

        # Fill the polygon with the mean gray value
        cv2.fillPoly(result_image, [polygon], mean_gray)

    # Display the final result
    return result_image
def connect_thin_wood_v1(gray_image):
    # Threshold the image to create binary regions
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image to identify blobs
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to store the result
    result_image = gray_image.copy()

    # Initialize a list to store the bounding boxes of blobs
    blob_bboxes = []

    # Iterate through the contours (blobs) and find the bounding box for each blob
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        blob_bboxes.append((x, y, x + w, y + h))

    # Sort the blob_bboxes by their X coordinate
    blob_bboxes.sort(key=lambda x: x[0])

    # Draw bounding boxes of each blob
    for bbox in blob_bboxes:
        x1, y1, x1_right, y1_bottom = bbox
        cv2.rectangle(result_image, (x1, y1), (x1_right, y1_bottom), 255, 2)

    # Draw bounding boxes between adjacent blobs
    for i in range(len(blob_bboxes) - 1):
        x1, y1, x1_right, y1_bottom = blob_bboxes[i]
        x2, y2, x2_right, y2_bottom = blob_bboxes[i + 1]

        # Calculate the dimensions of the new bounding box
        new_x = min(x1_right, x2)
        new_y = min(y1, y2)
        new_width = max(x1_right, x2) - new_x
        new_height = max(y1_bottom, y2_bottom) - new_y

        # Draw the new bounding box
        cv2.rectangle(result_image, (new_x, new_y), (new_x + new_width, new_y + new_height), 255, 2)

    # Display the final result
    cv2.imshow('Final Result', result_image)
    cv2.waitKey(0)



def simple_geometrical_analisys(allx, ally, allz):
    # Calculate percentiles
    x_01 = np.percentile(allx, 1)
    x_99 = np.percentile(allx, 99)
    y_01 = np.percentile(ally, 1)
    y_99 = np.percentile(ally, 99)
    z_01 = np.percentile(allz, 1)
    z_99 = np.percentile(allz, 99)
    z_50 = np.percentile(allz, 50)

    # Calculate differences
    # print("x1 ", x_01, ", x2 ", x_99)
    diff_x = x_99 - x_01
    diff_y = y_99 - y_01
    diff_z = z_99 - z_01

    # Calculate the sum vector using the Pythagorean theorem
    sum_vector = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)

    # # Print the results
    print("LENGTH EXTERNAL:", diff_x)
    print("DIAMETER EXTERNAL:", diff_y)



def detect_rotated_box(mask, frame):
    cnt1, completion = second_layer_accurate_cnt_estimator_and_draw(mask)
    rect = cv2.minAreaRect(cnt1)

    return rect, cnt1


def volume(L, d):
    if L is not None and d is not None:
        vol = (math.pi * pow(d, 2) * L) / 4
        return vol
    else:
        return 0


def crop_with_rect_rot(frame, rect, display=False):
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
    # print(M)

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

@timeit
def convert_depth_image_to_pointcloud(depth_image, intrinsics, frame, x, y, w_r, h_r):
    h, w = depth_image.shape

    enlargement_factor = 1.2
    enlarged_w = int(w_r * enlargement_factor)
    enlarged_h = int(h_r * enlargement_factor)

    # Calculate the new top-left corner of the enlarged rectangle
    new_x = int(x - (enlarged_w - w_r) / 2)
    new_y = int(y - (enlarged_h - h_r) / 2)

    # Calculate the new bottom-right corner of the enlarged rectangle
    new_x2 = new_x + enlarged_w
    new_y2 = new_y + enlarged_h

    # # Draw the enlarged rectangle on the image
    # color = (0, 255, 0)  # Green color (in BGR format)
    # thickness = 2  # Thickness of the rectangle's border
    # cv2.rectangle(frame, (new_x, new_y), (new_x2, new_y2), color, thickness)
    #
    # cv2.imshow("ds", resize_image(frame,80))

    end_point = (new_x + enlarged_w, new_y + enlarged_h)

    pointcloud = np.zeros((h, w, 3), np.float32)
    allz = []
    allx = []
    ally = []


    for r in range(new_y, end_point[1]):  # y
        for c in range(new_x, end_point[0]):  # x
            distance = float((depth_image[r, c] + 50) * 10)  # [y, x]

            if distance > MIN_DEPTH:
                # if distance > 1110:
                # else:
                #     cv2.circle(frame, (c, r), 1, (0, 0, 255), 1)

                result = rs.rs2_deproject_pixel_to_point(intrinsics, [c, r], distance)  # [c,r] = [x,y]
                # result[0]: right, result[1]: down, result[2]: forward

                # if abs(result[0]) > 1000.0 or255 abs(result[1]) > 1000.0 or abs(result[2]) > 1000.0:
                # print(result)
                # z,x,y
                pointcloud[r, c] = [int(result[2]), int(-result[0]), int(-result[1])]  # z,x,y
                allz.append(int(result[2]))
                allx.append(int(-result[0]))
                ally.append(int(-result[1]))






                # z,x,y
                # pointcloud[r,c] = [0,0,0]

    #visualize_3D_list_pointcloud(allx,ally, allz)




    return pointcloud, allz, allx, ally

def second_layer_accurate_cnt_estimator_and_draw(mask_bu):

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

            if perimeter1 > 200 and perimeter1 < 15000:  # 1200
                if circularity1 > 0.001 and circularity1 < 0.08:  # 0.05 / 0.1, 0.02
                    if area1 > 700 and area1 < 850000:  # 2200
                        if M0 > 0.30 and M0 < 630:  # 2200
                            if M01 > 0.01 and M01 < 370:  # 2200
                                if M10 > 0.01 and M10 < 1040:  #
                                    if M02 > 0.25 and M02 < 240:
                                        if M20 > 0.0002 and M20 < 2150:
                                            if solidity > 0.1 and solidity < 0.5:
                                                if ratio > 0.00005 and ratio < 0.8:  # rapporto pixel contour e bounding box

                                                    # print("|____________________________________|")
                                                    # print("__|RATIO-CORRECT|__:", ratio)



                                                    # print("|____________________|2nd LAYER CHOSEN", i, " area:",
                                                    #       int(area1), " perim:", int(perimeter1), " circul:",
                                                    #       round(circularity1, 5))
                                                    # print(" M0:", round(M0, 3), " M01:", round(M01, 3), " M10:",
                                                    #       round(M10, 3), " M02:", round(M02, 3), " M20:", round(M20, 5),
                                                    #       "ratio", round(ratio, 5), " solidity", round(solidity, 4))



                                                    # cv2.drawContours(frame, [cnt1], 0, (200, 200, 50), 1)

                                                    return cnt1, True
    print("________!!!!____________advanced shoots not detected")
    print("len contours", len(contours1))
    cv2.imshow("eee", mask_bu)
    time.sleep(3)
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
    return 0, False



intrinsics = obtain_intrinsics()
L_ = []
D_ = []
RRLLL = []
RRDDD = []
RPCLLL = []
RPCDDD = []
Z_all = []

print("CONFIG: iteration:", iteration)


delete_csv_file(csv_file_path)

for folders in os.listdir(PATH_HERE + PATH_2_AQUIS):
    #print("files:", os.listdir(PATH_HERE + PATH_2_AQUIS))
    folder_name = folders
    # videos = os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name)
    # writeCSVdata(folder_name, ["frame", "pixel", "volume", "distance_med", "volumes", "distances"])

    for videos in os.listdir(PATH_HERE + PATH_2_AQUIS + "/" + folder_name):

        # print(videos)
        print("FILENAME:", folder_name)

        if videos.endswith(".mkv"):
            #print(videos.split(".")[0])

            if videos.split(".")[0] == "RGB":

                path_rgb = os.path.join(PATH_HERE + PATH_2_AQUIS + "/" + folder_name, videos)
                # creo l oggetto per lo streaming
                video1 = cv2.VideoCapture(path_rgb)
            elif videos.split(".")[0] == "DEPTH":

                path_depth = os.path.join(PATH_HERE + PATH_2_AQUIS + "/" + folder_name, videos)
                video2 = cv2.VideoCapture(path_depth)

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
    # print("height : ", frame_height)
    # print("width : ", frame_width)
    size = (frame_width, frame_height)
    size2 = (frame_width2, frame_height2)

    nrfr = 0
    time.sleep(1)

    volumes_extracted = []

    while (video1.isOpened() and video2.isOpened()):


        ret, frame = video1.read()
        ret2, frame2 = video2.read()
        nrfr = nrfr + 1
        #print(nrfr)


        if ret and ret2:
            if nrfr > 2 and nrfr < 29:

                # gestionn depthimg
                frame2of = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((5, 5), np.float32) / 25
                ###!!!! BLURRA SOLO I VALORI INTERNI ALLA MASCHERA:
                # prendi i valori dentro la maschera (media o simile) ed estendili a tutta l immagine

                gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray = strech_hist_of_gray_image_4_contrast(gray_raw)


                ret, mask = cv2.threshold(gray, THRES_VALUE, 255, cv2.THRESH_BINARY)

                # cv2.imshow("raw", resize_image(gray_raw, 70))
                # cv2.imshow("equalized", resize_image(gray, 70))
                # #
                # # # cv2.imshow("or", resize_image(cropped_frame,300))
                # cv2.imshow("pc mask", resize_image(mask,70))
                # #
                # key = cv2.waitKey(CONTINOUS_STREAM)
                # if key == ord('q') or key == 27:
                #     break


                rect_point_of_interest, cnt = detect_rotated_box(mask, frame)
                # make white out of the bounfing refct
                x, y, w, h = cv2.boundingRect(cnt)
                white_bg = 255 * np.ones_like(mask)
                roi = mask[y:y + h, x:x + w]
                white_bg[y:y + h, x:x + w] = roi
                mask = white_bg
                mask_inverted = cv2.bitwise_not(mask)

                #rimuovo puntini neri rumore maschera: CLOSING
                kernel_noise_black_dot_mask = np.ones((5, 5), np.uint8)
                mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_CLOSE, kernel_noise_black_dot_mask)

                # rimuovo puntini neri rumore depth: CLOSING
                kernel_noise = np.ones((1, 1), np.uint8)
                frame2of_no_noise = cv2.morphologyEx(frame2of, cv2.MORPH_OPEN, kernel_noise)
                # punti chiari sui neri spariscono

                #uso una maschera piu stretta per eliminare gli effetti di bordo, poi comunque rimaschererò con la maschera corretta
                kernel_mask_provv = np.ones((3, 3), np.uint8)
                mask_inverted_provv = cv2.erode(mask_inverted, kernel_mask_provv, iterations=1)

                # maschero il depth con la maschera rgb sottile
                black_out_depth_in = cv2.bitwise_and(frame2of_no_noise, frame2of_no_noise, mask=mask_inverted_provv)
                # rimuovo i puntini bianchi di background:
                kernel_white_bg_dot = np.ones((3, 3), np.uint8)
                black_out_depth_in = cv2.morphologyEx(black_out_depth_in, cv2.MORPH_OPEN, kernel)

                black_out_depth_in = connect_thin_wood(black_out_depth_in)
                # try:
                #     connect_thin_wood(black_out_depth_in)
                #
                # except Exception as e:
                #     print("ERROR CONNECTION: ",e)

                #icostruisco con una dilation i punti di noise all interno del depth
                kernel_denoising = np.ones((KERNEL_VALUE_DILATION, KERNEL_VALUE_DILATION), np.uint8)

                black_out_depth_in_denoised = cv2.dilate(black_out_depth_in, kernel_denoising, iterations=DILATION_ITERATION)

                # rendo la maschera più sottile per non includere gli effetti di birdo nella computazione della media
                kernel_mask_erosion = np.ones((5, 5), np.uint8)
                eroded_mask_inverted = cv2.erode(mask_inverted, kernel_mask_erosion, iterations=1)

                # calcolo valore medio interno alla maschera
                # estraggo valori interni alla maschera
                pixels_OI = frame2of_no_noise[eroded_mask_inverted == 255]
                # li converto in lista monodim
                pixels_OI_l = pixels_OI.tolist()
                # elimino i valori ugiuali a zero
                pixels_OI_l = [i for i in pixels_OI_l if i != 0]
                # estraggo ora il valore medio
                avg_masked_value = int(np.mean(pixels_OI_l))
                print("MEAN DEPTH", avg_masked_value + 50)
                # riempi l esterno con il valore medio
                black_out_depth_in_AVARAGED = black_out_depth_in_denoised.copy()
                # copio l immagine per editarla
                black_out_depth_in_AVARAGED[np.where((black_out_depth_in_AVARAGED == [0]))] = [avg_masked_value]
                length_branch = np.sqrt(w ** 2 + h ** 2)
                # arrotondo al dispari piu vicino per definire il kernel in base alla lunghezza
                ker_size = int(np.ceil(length_branch / 70))
                # faccio un closing
                kernel1 = np.ones((ker_size, ker_size), np.uint8)
                black_out_depth_in_blurred = cv2.morphologyEx(black_out_depth_in_AVARAGED, cv2.MORPH_CLOSE, kernel1)
                # black_out_depth_in_blurred = cv2.dilate(black_out_depth_in_blurred, kernel1, iterations=1)
                # riapplico la maschera sull immagine blurrata di depth con background stabile
                black_out_depth_in_remasked = cv2.bitwise_and(black_out_depth_in_blurred, black_out_depth_in_blurred,
                                                              mask=mask_inverted)


                # qui stiamo cercando di visualizzare quanti pixel neri ci sono all interno della maschera che corrispondono al rumore TO DO
                pixels = frame2of[[mask_inverted]]

                # black_out_dapth_in = frame2of * mask_inverted
                if POINTCLOUD_EVALUATION_VOLUME:
                    try:

                        # maschero anche la pointcloud prendendo in considerazione solo la mask
                        pc, allz, allx, ally = convert_depth_image_to_pointcloud(black_out_depth_in_remasked, intrinsics, frame,
                                                                                 x, y, w, h)

                        #visualize_3D_list_pointcloud(allx, ally, allz)
                        simple_geometrical_analisys(allx, ally, allz)

                        pointc = np.column_stack((allx, ally, allz))
                        #visualize_3D_list_pointcloud(pointc)
                        global_pc = z_score_normalization(pointc, 0.01)
                        #visualize_3D_list_pointcloud(global_pc)
                    except:
                        print("error vis cyl")


                    collector = []
                    collector.append(global_pc)
                    next_iteration_collector = []
                    for i in range(iteration):

                        #print("iteration:", i, " n PC:", len(collector))
                        for p in range(len(collector)):
                            #print("splittin image n", p+1 , " /", len(collector))

                            half1, half2 = split_PC(collector[p])
                            next_iteration_collector.append(half1)
                            next_iteration_collector.append(half2)




                        collector = next_iteration_collector
                        #svuoto il contenitore temporaneo da portare all iterazione dopo
                        next_iteration_collector = []

                    zzs = []
                    lls = []
                    dds = []
                    pcds = []
                    obds = []
                    #print("images:", len(collector))
                    for mp in range(len(collector)):
                        #visualize_3D_list_pointcloud(collector[mp])
                        try:
                            length,width, z_mean ,pcd, obb = dimension_evaluator(collector[mp])
                        except:
                            print("error! dimension null")
                            length,width,z_mean, pcd, obb = 0,0,0,0,0


                        lls.append(length)  # lungheezaa
                        dds.append(width)  # diametro
                        zzs.append(z_mean)
                        pcds.append(pcd)
                        obds.append(obb)

                    w_clean, l_clean = clean_measurements(dds, lls)






                    # Initialize the total volume
                    volumes_all = []

                    # Calculate sub-volumes iteratively
                    for i in range(len(w_clean)):
                        sub_volume = volume(l_clean[i], w_clean[i])
                        volumes_all.append(sub_volume)

                    volume_tot = sum(volumes_all)
                    length = sum(l_clean)
                    diam_med = np.mean(w_clean)
                    z_med = np.mean(zzs)
                    std_len = np.std(l_clean)
                    std_dia = np.std(w_clean)
                    std_vol = np.std(volumes_all)



                    save_to_csv_recursive([volume_tot, z_med], csv_file_path)
                    print("length: ", length, "STDlen:", std_len, "diam med", diam_med, "STDdiam", std_dia)
                    # print("diam: ", dyy)
                    print("volume: ", volume_tot, " STD:", std_vol, "real", volume(320, 8))

                    if SHOW_VOLUME_TOO_BIG:
                        if volume_tot > OUTLIER_VOLUME:
                            opt = o3d.visualization.VisualizerWithKeyCallback().get_render_option()

                            o3d.visualization.draw_geometries(pcds + obds,point_show_normal=False)

                    if VISUALIZE_CYLINDRIFICATED_WOOD:
                    #if VISUALIZE_CYLINDRIFICATED_WOOD:
                        opt = o3d.visualization.VisualizerWithKeyCallback().get_render_option()

                        o3d.visualization.draw_geometries(pcds + obds,point_show_normal=False)
                        # Create a visualization option

                        # Create a visualization window using the standard OpenGL approach
                        # Create a visualization object

                #pc = cv2.bitwise_and(pc, pc, mask=mask_inverted)

                #pc_cropped = crop_with_rect_rot(pc, rect_point_of_interest)


                # visible_pointcloud = ((pc_cropped - pc_cropped.min()) * (1 / (pc_cropped.max() - pc_cropped.min()) * 255)).astype('uint8')

                # END _ ONLY DISPLAY


                # crop for imaging
                black_out_depth_in_denoised_cropped = crop_with_rect_rot(black_out_depth_in_denoised, rect_point_of_interest)
                black_out_depth_in_cropped = crop_with_rect_rot(black_out_depth_in, rect_point_of_interest)
                black_out_depth_in_AVARAGED_cropped = crop_with_rect_rot(black_out_depth_in_AVARAGED,
                                                                         rect_point_of_interest)
                cropped_frame = crop_with_rect_rot(frame, rect_point_of_interest)
                mask_inverted_crop = crop_with_rect_rot(mask_inverted, rect_point_of_interest)
                black_out_depth_in_blurred_cropped = crop_with_rect_rot(black_out_depth_in_blurred,
                                                                        rect_point_of_interest)
                black_out_depth_in_remasked_cropped = crop_with_rect_rot(black_out_depth_in_remasked,
                                                                         rect_point_of_interest)
                frame2of_no_noise_cropped = crop_with_rect_rot(frame2of_no_noise, rect_point_of_interest)

                frame2of_cropped = crop_with_rect_rot(frame2of, rect_point_of_interest)

                # plt.hist(mask_inverted.tolist(),bins=25)
                # plt.show()

                #volume_tot_cylinders = advanced_cylindricator_v2(mask_inverted_crop, pc_cropped, cropped_frame,black_out_depth_in_remasked_cropped)

                # plt.hist(allz, bins=300)
                # plt.show()
                if SHOW_FILTERING_PROCESS:
                    processing = cv2.vconcat([frame2of_cropped, frame2of_no_noise_cropped, black_out_depth_in_cropped,black_out_depth_in_denoised_cropped,
                                              black_out_depth_in_AVARAGED_cropped, black_out_depth_in_blurred_cropped,
                                              black_out_depth_in_remasked_cropped, mask_inverted_crop])
                    # print("frame2of_cropped, frame2of_no_noise_cropped,black_out_depth_in_cropped, black_out_depth_in_AVARAGED_cropped, black_out_depth_in_blurred_cropped,black_out_depth_in_remasked_cropped")

                    cv2.imshow("processing", resize_image(processing, 200))

                if SHOW_MASK_ONLY or volume_tot > OUTLIER_VOLUME:
                    cv2.imshow("mask", resize_image(mask_inverted_crop, 150))

                # cv2.imshow("or", resize_image(cropped_frame,300))
                # cv2.imshow("pc mask", resize_image(visible_pointcloud,300))

                key = cv2.waitKey(CONTINOUS_STREAM)
                if SHOW_DEPTH_ONLY:
                    cv2.imshow("depth", resize_image(frame2of, 100))

                # cv2.imshow("or", resize_image(cropped_frame,300))
                # cv2.imshow("pc mask", resize_image(visible_pointcloud,300))



                key = cv2.waitKey(CONTINOUS_STREAM)
                if key == ord('q') or key == 27:
                    break



                # if volume_tot > OUTLIER_VOLUME:
                #         time.sleep(5)



        else:
            break


    video1.release()
    video2.release()
    cv2.destroyAllWindows()

    print("COMPLETED ", nrfr, " frames")

plt.close('all')


