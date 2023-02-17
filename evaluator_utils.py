



import numpy as np


import time

from numpy.linalg import norm
import os

import os
import sys
import math
import cv2
import csv
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from datetime import datetime
import pyrealsense2 as rs





THRES_VALUE = 65
CROPPING_ADVANCED = True
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
OFFSET_CM_COMPRESSION = 50
CYLINDER_SHOW = True



def calculate_pointcloud(color_intrin,color_to_depth_extrin,depth_intrin,color_image,depth_image):
    # funzionerà se riporto l immagine depth in uint16 in mm senza offset
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    aligned_color = np.zeros((height, width, 3))
    height_color = color_image.shape[0]
    width_color = color_image.shape[1]
    for v in range(width_color):
        for u in range(height_color):

            color_pixel = [v, u]
            color_point = rs.rs2_deproject_pixel_to_point(color_intrin, color_pixel, 1)
            depth_point = rs.rs2_transform_point_to_point(color_to_depth_extrin, color_point)
            depth_pixel = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)
            # depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), depth_scale,
            #     0.11, 1.0, depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, color_pixel)
            u_depth = int(round(depth_pixel[1]))
            v_depth = int(round(depth_pixel[0]))
            if u_depth < 0 or u_depth > height - 1 or v_depth < 0 or v_depth > width - 1:
                pass
            else:
                aligned_color[u_depth][v_depth][0] = color_image[u][v][0]
                aligned_color[u_depth][v_depth][1] = color_image[u][v][1]
                aligned_color[u_depth][v_depth][2] = color_image[u][v][2]


def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    _intrinsics.model = cameraInfo.distortion_model
    #_intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.D]
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
    #result[0]: right, result[1]: down, result[2]: forward
    return result[2], -result[0], -result[1]

def distance_med_from_masked_depth(mask,depth):
    #diam
    imask = mask < 255
    frame22 = 255 * np.ones_like(depth, np.uint8)  # all white
    frame22[imask] = depth[imask]
    distance_med = extract_medium_from_depth_segmented((depth)[imask])
    #print("ala", (depth)[imask])
    #print("DISTCYL", distance_med)

    return distance_med, frame22

def volume_from_mask_cylinder(mask, frame):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # if multiple cnt take only the bigger
    # print("cnt ")
    h = mask.shape[0]
    w = mask.shape[1]
    cnt = []
    for cnt_i in contours:
        area = cv2.contourArea(cnt_i)
        area_image = h * w
        ratio = area / area_image
        #print("ratio", ratio)
        # print("ratio", ratio)
        if ratio > 0.001:
            cnt = cnt_i
    if cnt == [] and len(contours) != 0:

        #print("contours", contours)
        print("contours l", len(contours))

        cnt = contours[-1]



    rect = cv2.minAreaRect(cnt)
    """
    except:
        cv2.imshow("error rect",mask)
        print(mask)
        print("contour",cnt)
        print(mask.shape)
        cv2.waitKey(0)
        time.sleep(100)
    """


    box = cv2.boxPoints(rect)
    box = np.int0(box)
    draw = False
    dA, dB, mask = calc_box_legth(box, mask, draw)
    # volume
    # implementa questo con la box e non con tutta l immagine
    h = mask.shape[0]
    w = mask.shape[1]
    lenght_shoot = max(dA, dB)
    number_of_black_pix = np.sum(mask == 0)
    if lenght_shoot != 0:
        diam_med = number_of_black_pix / lenght_shoot
        volume = math.pi * pow((diam_med / 2), 2) * lenght_shoot

    else :
        volume = 0
    #print("vol",lenght_shoot)
    return volume, (dA,dB), frame

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def calc_box_legth(box, orig, draw):

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint bet ween bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    # compute the Euclidean distance between the midpoints

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # draw the midpoints on the imageà
    if draw:
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)
        #print("drawd :", (tlblX, tlblY) )
        #print("dimensions", dB, dA)
    # draw the object sizes on the image



    """
    cv2.putText(orig, "{:.1f}in".format(dA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    """

    return dA,dB , orig


def crop_image_with_box_and_margin(frame, box):

    h, w, c = frame.shape

    P1 = box[0]
    P2 = box[1]
    P3 = box[2]
    P4 = box[3]
    margine = int(h/10)
    xmax = max(P1[0], P2[0], P3[0], P4[0]) + margine
    xmin = min(P1[0], P2[0], P3[0], P4[0]) - margine
    ymax = max(P1[1], P2[1], P3[1], P4[1]) + margine
    ymin = min(P1[1], P2[1], P3[1], P4[1]) - margine
    #print(box)


    if xmin > 0 and ymin > 0:
        if xmax <= w and ymax <= h:
            print("cropped")

            frame = frame[ymin:ymax, xmin:xmax]

            return  frame

    print("impossible crop entire image",w, xmin, xmax, h,ymin,ymax)
    return frame

def rotated_box_cropper(mask, depth):
    #margin augmented
    mask = cv2.copyMakeBorder(src=mask, top=15, bottom=15, left=15, right=15,borderType=cv2.BORDER_CONSTANT, value=(255))
    depth = cv2.copyMakeBorder(src=depth, top=15, bottom=15, left=15, right=15,borderType=cv2.BORDER_CONSTANT, value=(255))

    #calc cnt
    imagem1 = (255 - mask)

    contours, hierarchy = cv2.findContours(imagem1, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0 :
        #if multiple cnt take only the bigger
        #print("cnt ")


        h = mask.shape[0]
        w = mask.shape[1]
        cnt = []
        for cnt_i in contours:
            area = cv2.contourArea(cnt_i)
            area_box = h*w
            ratio = area/area_box
            if ratio > 0.001 and area > 10:
                #print("ratio, area", ratio, area)

                if ratio > 0.027 :
                    cnt = cnt_i
                elif ratio > 0.015 and ratio < 0.027:
                    if area > 400:
                        cnt = cnt_i

        if cnt == []:
            print("not found a good ")
            print("len cont", len(contours))
            print("image dimension:",mask.shape)

            for cnt_i in contours:
                area = cv2.contourArea(cnt_i)
                area_box = h * w
                ratio = area / area_box
                print("ratio, area", ratio, area)

            cnt = contours[-1]
        #calc box

        rect = cv2.minAreaRect(cnt)




        box = cv2.boxPoints(rect)
        box = np.int0(box)

        mask = (255 - mask)

        depth = cv2.bitwise_not(depth)
        #crop rot box + margin white
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")




        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(mask, M, (width, height),borderMode=cv2.BORDER_REPLICATE)
        warped_depth = cv2.warpPerspective(depth, M, (width, height),borderMode=cv2.BORDER_REPLICATE)

        warped = (255 - warped)
        warped_depth = cv2.bitwise_not(warped_depth)
        succesful = True
        return warped, warped_depth, box, succesful
    else:
        succesful = False
        return mask, depth, [], succesful




def image_splitter(frame):

    h = frame.shape[0]
    w = frame.shape[1]
    #channels = frame.shape[2]

    # decido se tagliare in altezza o larghezza
    if h > w:
        #top bottom
        half2 = h // 2

        img1 = frame[:half2, :]
        img2 = frame[half2:, :]

    else:
        #left right
        half = w // 2

        img1 = frame[:, :half]
        img2 = frame[:, half:]


    return img1, img2


def sub_box_iteration_cylindrificator(box1,frame, mask, depth):
    imagem1 = (255 - mask)
    contours, hierarchy = cv2.findContours(imagem1, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # if multiple cnt take only the bigger
        # print("cnt ")

        h = mask.shape[0]
        w = mask.shape[1]
        cnt = []
        for cnt_i in contours:
            area = cv2.contourArea(cnt_i)
            area_box = h * w
            ratio = area / area_box
            if ratio > 0.001 and area > 10:
                # print("ratio, area", ratio, area)

                if ratio > 0.027:
                    cnt = cnt_i
                elif ratio > 0.015 and ratio < 0.027:
                    if area > 400:
                        cnt = cnt_i

        if cnt == []:
            print("not found a good ")

            for cnt_i in contours:
                area = cv2.contourArea(cnt_i)
                area_box = h * w
                ratio = area / area_box
                print("ratio, area", ratio, area)

            cnt = contours[-1]
        # calc box

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

    draw = True
    dA,dB, frame = calc_box_legth(box, frame, draw)
    #print("dim : ", frame.shape, box)
    # area = calc area of black in image
    number_of_black_pix = np.sum(mask == 0)
    # approx diameter = wood area  / lenght of box=max(dA,dB=
    diameter_medium = number_of_black_pix / (max(dA,dB))
    # iteration needed = maxL of box / approximate diameter
    iteration_needed = int(((max(dA,dB)) / diameter_medium) /2)
    iteration = 4

    images_collector = []
    depth_images_collector = []
    images_collector.append(mask)
    depth_images_collector.append(depth)
    all_images_in_loop = []
    all_images_in_loop.append(images_collector)



    for it in range(iteration):
        try:

            #prima ruoto e taglio
            new_collector_images = []
            new_collector_depth_images = []
            for im_num in range(len(images_collector)):

                rot_mask, rot_mask_depth, box, succesful = rotated_box_cropper(images_collector[im_num], depth_images_collector[im_num])
                if succesful != False:

                    #poi splitto
                    img1,img2 = image_splitter(rot_mask)
                    img_d1, img_d2 = image_splitter(rot_mask_depth)

                    new_collector_images.append(img1)
                    new_collector_images.append(img2)
                    new_collector_depth_images.append(img_d1)
                    new_collector_depth_images.append(img_d2)
                else:
                    breaking_point = True
                    print(" not succesful cylkindricization, termiate frame")
                    return 0,0
            images_collector = new_collector_images
            depth_images_collector = new_collector_depth_images
            all_images_in_loop.append(images_collector)


        #print("images : ",len(images_collector))
            if CYLINDER_SHOW:
                volumes = []
                distances = []
                for x in range(len(images_collector)):


                    distance_cylinder,masked_depth = distance_med_from_masked_depth(images_collector[x],depth_images_collector[x])
                    #print("distancce")

                    distances.append(distance_cylinder)
                    vis = np.concatenate((depth_images_collector[x], images_collector[x],masked_depth), axis=1)
                    cv2.imshow("im"+str(x),vis)
                    cv2.moveWindow("im"+str(x), 150*x, 150*x);

                    volume, dimensions, frame = volume_from_mask_cylinder(images_collector[x], frame)
                    #print("volu")






                    volumes.append(volume)


        except Exception as e:
            print("error volume, %s", str(e))
            cv2.imshow("MASK!!", mask)
            cv2.imshow("FRAME!!", frame)

            #for y in range(len(images_collector)):
                #cv2.imshow("error" + str(y), images_collector[y])
            for k in range(len(all_images_in_loop)):

                for j in range (len(all_images_in_loop[k])):

                    cv2.imshow("iteration_" + str(k)  + "_image_"+str(j), all_images_in_loop[k][j])
            cv2.imshow("ERROR specific" + str(x), images_collector[x])
            cv2.waitKey(0)
            time.sleep(1000)



    #for image_in in images_collector:




    return volumes,distances, dA, dB

def optional_closing(frame):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    #frame = cv2.erode(frame, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    print("opened")


    return frame

def extract_medium_from_depth_segmented(depth):

    distance_medium = np.mean(depth) + OFFSET_CM_COMPRESSION

    return distance_medium
def set_white_extreme_depth_area(frame, depth, max_depth, min_depth):
    #print("depth = ", depth.shape)
    #depth = cv2.inRange(depth, min_depth-50, max_depth-50)
    depth = cv2.inRange(depth, min_depth-50, max_depth-50)


    #depth = cv2.inRange(depth, max_depth - 50, min_depth - 50)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #depth = cv2.morphologyEx(depth, cv2.MORPH_OPEN, kernel)
    #depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)

    result = cv2.bitwise_and(frame, frame, mask=depth)
    result[depth == 0] = [255, 255, 255]  # Turn background white

    #depth = cv2.inRange(depth, 60, 200)
    return result


"""
def skeletonize_mask(mask_bu):

    # skeleton
    immi = (255 - mask_bu)

    size = np.size(immi)
    skel = np.zeros(immi.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv2.morphologyEx(immi, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(immi, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(immi, element)
        skel = cv2.bitwise_or(skel, temp)
        immi = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(immi) == 0:
            break


    return skel
"""

def volume_from_lenght_and_diam_med(box, frame, mask_bu):
    #lenght of the box, aka minimum rectangular distance
    # |__->   L2
    #area : pixel
    # L1(diam_med) = A / L2
    lenght_shoot = distanceCalculate(box[3], box[1])


    pixel = count_and_display_pixel(frame, mask_bu)

    diam_med = pixel / lenght_shoot
    cv2.line(frame, box[3], box[1], (255, 255, 0), 4) #int(diam_med)

    volume = math.pi * pow((diam_med / 2), 2) * lenght_shoot
    #print("volume: ", volume)
    return volume,frame


def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def second_layer_accurate_cnt_estimator_and_draw(mask_bu,frame):

    imagem1 = (255 - mask_bu)

    contours1, hierarchy1 = cv2.findContours(imagem1, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    i = 0
    for cnt1 in contours1:
        #calcolo area e perimetro
        area1 = cv2.contourArea(cnt1)
        h = mask_bu.shape[0]
        w = mask_bu.shape[1]

        area_box = h * w
        ratio = area1 / area_box

        # perimeter
        perimeter1 = cv2.arcLength(cnt1, True)
        i += 1

        if perimeter1 > 200  and area1 > 200:
            #calcolo circolarità
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


            if perimeter1 > 200 and perimeter1 < 7000: #1200
                if circularity1 > 0.01 and circularity1 < 0.3: #0.05 / 0.1, 0.02
                    if area1 > 800 and area1 < 150000: #2200
                        if M0 > 1.15 and M0 < 40:  # 2200
                            if M01 > 0.40 and  M01 < 40:  # 2200
                                if M10 > 0.5 and M10  < 40:            #
                                    if M02 > 0.25 and M02 < 40:
                                        if M20 > 0.002 and M20 < 95:
                                            if solidity > 0.01 and solidity < 0.5:
                                                if ratio > 0.0005 and ratio < 0.8: #rapporto pixel contour e bounding box


                                                    #print("|____________________________________|")
                                                    #print("__|RATIO-CORRECT|__:", ratio)
                                                    print("|____________________|2nd LAYER CHOSEN", i," area:" ,int(area1)," perim:" , int(perimeter1)," circul:" , round(circularity1,5))
                                                    print(" M0:", round(M0,3)," M01:",round(M01,3)," M10:", round(M10,3), " M02:", round(M02,3)," M20:",  round(M20,5), "ratio", round(ratio,5), " solidity", round(solidity, 4))
                                                    #cv2.drawContours(frame, [cnt1], 0, (0, 200, 50), 1)

                                                    #moments


                                                    Ix = M['m20']
                                                    Iy = M['m02']
                                                    b = int(pow((Iy * pow(area1,2))/Ix, 1/4))
                                                    h = int(pow((Ix * pow(area1,2))/Iy, 1/4))
                                                    x1 = cx - b/2
                                                    x2 = cx + b/2
                                                    y1 = cy - h / 2
                                                    y2 = cy + h / 2
                                                    #print("dim", b,h)
                                                    #print("c", cx, cy)


                                                    frame = cv2.circle(frame, (cx,cy), 2, (255,0,0), 2)

                                                    return cnt1,frame, True
    print("________!!!!____________advanced shoots not detected")
    print("len contours", len(contours1))
    for cnt1 in contours1:
        #calcolo area e perimetro
        area1 = cv2.contourArea(cnt1)
        # perimeter
        perimeter1 = cv2.arcLength(cnt1, True)
        if perimeter1 > 200  and area1 > 200:
            #calcolo circolarità
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


    #time.sleep(1000)
    return 0,frame,False

def crop_with_box_one_shoot(box, mask_bu, frame,depth):

    P1 = box[0]
    P2 = box[1]
    P3 = box[2]
    P4 = box[3]
    margine = 10
    xmax = max(P1[0], P2[0], P3[0], P4[0]) + margine
    xmin = min(P1[0], P2[0], P3[0], P4[0]) - margine
    ymax = max(P1[1], P2[1], P3[1], P4[1]) + margine
    ymin = min(P1[1], P2[1], P3[1], P4[1]) - margine
    #print(box)

    h, w, c = frame.shape


    if xmin > 0 and ymin > 0:
        if xmax <= w and ymax <= h:

            mask_bu = mask_bu[ymin:ymax, xmin:xmax]
            frame = frame[ymin:ymax, xmin:xmax]
            depth = depth[ymin:ymax, xmin:xmax]

            return mask_bu, frame, depth

    print("impossible crop entire image, width:",w,"x min e max:", xmin, xmax,"height: ", h,ymin,ymax)
    return mask_bu, frame, depth


def draw_and_identify_current_cnt(frame, i,box):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    frame = cv2.putText(frame, str(i), (box[0][0], box[0][1] - 100), font,
                        fontScale, color, thickness, cv2.LINE_AA)

def draw_and_calculate_poligonal_max_diameter(cnt, frame):
    epsilon = 0.003 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    #cv2.drawContours(frame, [approx], -1, (255, 255, 0), 1)


def draw_and_calculate_rotated_box(cnt, frame):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    return box

def fit_and_draw_line_cnt(cnt, frame):

    rows, cols = frame.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lef = int((-x * vy / vx) + y)
    ri = int(((cols - x) * vy / vx) + y)
    cv2.line(frame, (cols - 1, ri), (0, lef), (0, 255, 0), 2)



def first_layer_detect_raw_shoots(im,frame):
    im = (255 - im)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))


    im = cv2.dilate(im, kernel, iterations=1)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(im, 175, 175)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge = cv2.dilate(edge, kernel1, iterations=1)


    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the index of the largest contour
    valid = []
    color = 0
    i = 0
    #print("|||||||||||||||||||||||||||||||||||||||||||")

    for cnt in contours:
        i += 1

        # area
        area = cv2.contourArea(cnt)
        # perimeter
        perimeter = cv2.arcLength(cnt, True)
        # fitting a line

        if perimeter != 0 and area != 0:
            circularity = (4 * math.pi * area) / (pow(perimeter, 2))

            #print("AVIABLE:  i A,P,C | ", i," | ", int(area)," | ", int(perimeter), " | ",circularity)
            '''
            if perimeter > 100 and perimeter < 100000:  # 1200  
                if circularity > 0.0005 and circularity < 0.9:  # 0.05 / 0.1, 0.02
                    if area > 300 and area < 120000:  # 2200
            '''
            if area > 1000:

                if circularity < 0.4:
                    if perimeter > 500:
                        #print("first layer CHOSEN! i A,P,C", i, int(area), int(perimeter), circularity)


                        return i
    print("no one aviable")
    return 0



def count_and_display_pixel(green,mask):
    height = green.shape[0]
    width = green.shape[1]
    #org = (int(height/2), int(width/2))
    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    color = (255, 0, 0)
    thickness = 1
    number_of_black_pix = np.sum(mask == 0)
    green = cv2.putText(green, 'pix : ' + str(number_of_black_pix), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return number_of_black_pix


def writeCSVdata(time,data):
    """
    write data 2 CSV
    :param data: write to a csv file input data (append to the end)
    :return: nothing
    """
    # scrive su un file csv i dati estratti dalla rete Neurale
    name = ""

    file = open('./data/' + time +'.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(data)
    file.close()

def writeCSVdata(time, data):
    """
    write data 2 CSV
    :param data: write to a csv file input data (append to the end)
    :return: nothing
    """
    # scrive su un file csv i dati estratti dalla rete Neurale
    name = ""

    file = open('./data/' + time + '.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(data)
    file.close()


"""
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
"""


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

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, bottom_color, top_color)

    imask = mask > 0
    imagem = (255 - mask)


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


def blob_detector(frame,depth):
    #print("start blob")
    pixel  = 0
    volume = 0
    frame_BU = frame.copy()
    gray_bu = cv2.cvtColor(frame_BU, cv2.COLOR_BGR2GRAY)
    ret, mask_bu = cv2.threshold(gray_bu, THRES_VALUE, 255, cv2.THRESH_BINARY)
    mask_bu = optional_closing(mask_bu)


    try:
        i = first_layer_detect_raw_shoots(mask_bu, frame)
    except:
        print("error first layer, skipped analysis")
        i = 0
    #print("end first lay blob")
    if i != 0:
        #every video should be analized from folder with specific names: A_1 ramo A condizione 1
        #save csv all in same folder in same method  A_1
    #need to implement a continous imges analyzer, to do so, use all the momentum and area perim and circularity
        # if those data are not enought use also color differenziation hsv
        #if this system is not enough use pixel


        cnt1, frame , completion = second_layer_accurate_cnt_estimator_and_draw(mask_bu, frame)

        if completion:
            #print("complietion ok,", completion)
            fit_and_draw_line_cnt(cnt1, frame)
            #draw_and_calculate_poligonal_max_diameter(cnt1, frame)
            box1 = draw_and_calculate_rotated_box(cnt1, frame)
            #print("box: ",box1)


            mask_bu, frame, depth = crop_with_box_one_shoot(box1, mask_bu, frame, depth)








            #total_volume,frame = sub_box_iteration_cylindrificator(iteration, box1,frame, mask_bu)


            pixel = count_and_display_pixel(frame,mask_bu)
            #print("counted pixel and displayed")

            volume, frame = volume_from_lenght_and_diam_med(box1, frame, mask_bu)

            volumes, distances, dA, dB = sub_box_iteration_cylindrificator(box1, frame, mask_bu,depth)
            cylindrification_results = [volumes, distances]
            return frame, mask_bu, depth, pixel, volume, cylindrification_results, True , dA, dB
        else:
            print("second layer failed, returning empty")
            return frame,mask_bu, depth, 0, 0, [[0], [0]], False ,0,0

    else:
        return frame,mask_bu, depth, 0, 0, [[0], [0]], False,0,0




        #skel = skeletonize_mask(mask_bu)

















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



