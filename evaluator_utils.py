import numpy as np

import time
import struct
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
from additional_functions import *

THRES_VALUE = 30
CROPPING_ADVANCED = True
max_value = 50
max_value_H = 360 // 2
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

def rotate_image_width_horizontal_max(image):
    h = image.shape[0]
    w = image.shape[1]

    if h > w:
        #rotate cc
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)




    return image


def medium_points_of_box_for_dimension_extraction(box, orig):
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

    #  tl *--------------tltr------------------* tr
    #     |                                    |
    #     |-  tlbl                             |-  tlbr
    #     |                                    |
    #  bl *----------------blbr----------------* br
    # compute the Euclidean distance between the midpoints


    # draw the midpoints on the imageà
    draw = True
    divider_diam = 3


    # draw lines between the midpoints

        #ora metto i punti


        # print("drawd :", (tlblX, tlblY) )
        # print("dimensions", dB, dA)
    # draw the object sizes on the image

    #  tl *--------------tltr------------------* tr
    #     |                                    |
    #     |-  tlbl                             |-  tlbr
    #     |                                    |
    #  bl *----------------blbr----------------* br
    #

    return (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)


def calc_box_for_subcylinder_recognition(mask):
    mask = (255 - mask)
    #cv2.imshow("mcdo", imagem1)
    h = mask.shape[0]
    w = mask.shape[1]
    cnt = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt_i in contours:
        area = cv2.contourArea(cnt_i)
        area_image = h * w
        ratio = area / area_image

        # print("ratio", ratio)
        # print("ratio", ratio)
        if ratio > 0.001:
            cnt = cnt_i
    if cnt == [] and len(contours) != 0:
        # print("contours", contours)
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
    return box


def real_volume_from_pointcloud(depth_frame, intrinsics, box, rgbframe, mask):
    frame2_u16 = convert_u8_img_to_u16_d435_depth_image(depth_frame)
    pointcloud = convert_depth_image_to_pointcloud(frame2_u16, intrinsics)
    #print("shapes:", rgbframe.shape, frame2_u16.shape, pointcloud.shape)

    # array = create_flatten_array_for_ply_save(pointcloud)

    # write_pointcloud('pointcloud.ply', array)

    h, w, c = pointcloud.shape

    #  tl *--------------tltr------------------* tr
    #     |                                    |
    #     |-  tlbl                             |-  tlbr
    #     |                                    |
    #  bl *----------------blbr----------------* br
    #

    (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY) = medium_points_of_box_for_dimension_extraction(box,
                                                                                                                   rgbframe)
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    orig = rgbframe
    divider_diam = 5

    (tl, tr, br, bl) = box

    diameter = dA
    length = dB

    h, w, c = orig.shape

    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (0, 255, 0), -1)
    secure_pointL = (int(tlblX + diameter / divider_diam), int(tlblY))
    secure_pointR = (int(trbrX - diameter / divider_diam), int(trbrY))




    mask_copy_rect = mask.copy()
    cv2.rectangle(rgbframe, tl, br, (255, 0, 65), 2)
    kernel = np.ones((9, 9), np.uint8)
    mask_copy_rect = cv2.dilate(mask_copy_rect, kernel, iterations=1)
    mask_inv_rect = 255 - mask_copy_rect
    cnts,hrc = cv2.findContours(mask_inv_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # loop over the contours
    cs =[]
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        #cv2.drawContours(orig, [c], -1, (0, 255, 0), 2)
        #cv2.circle(orig, (cX, cY), 2, (255, 255, 255), -1)
        cs.append(cX)
        cs.append(cY)



    l_image  = mask_inv_rect[0: h, 0:int(w/2)]
    r_image  = mask_inv_rect[0: h, int(w/2):w]
    pointcloud_l = pointcloud[0: h, 0:int(w/2)]
    pointcloud_r  = pointcloud[0: h, int(w/2):w]

    #cv2.imshow("mass",mask)


#pointcloud visibile
    visible_pointcloud = ((pointcloud - pointcloud.min()) * (1/(pointcloud.max() - pointcloud.min()) * 255)).astype('uint8')
    #cv2.imshow("mass_pt", visible_pointcloud)




    zvec = []
    yvec = []
    xvec = []
    for x in range(pointcloud.shape[0]):
        for y in range(pointcloud.shape[1]):
            point = pointcloud[x, y]
            point_mask = mask[x, y]
            if point_mask == 0:
                #cv2.circle(rgbframe, (y + int(w/2), x), 1, (255, 0, 255), -1)
                zzz = int(point[2])
                xxx = int(point[1]) #ATTENZIONE CHE SONO INVERTITI!!!!!!!!
                yyy = int(point[0])
                zvec.append(zzz)
                xvec.append(xxx)
                yvec.append(yyy)




    deltax = max(xvec) - min(xvec)
    deltay = max(yvec) - min(yvec)
    deltaz = max(zvec) - min(zvec)
    meanz = statistics.mean(list(zvec))
    stdz = statistics.stdev(list(zvec))

    print("Right styatistcal", meanz, stdz)


    print("delta PRIMA x y z", deltax,deltay,deltaz )


    zvec = []
    yvec = []
    xvec = []
    filter_alpha = 0.5
    for x in range(pointcloud.shape[0]):
        for y in range(pointcloud.shape[1]):
            point = pointcloud[x, y]
            point_mask = mask[x, y]
            if point_mask == 0:

                #cv2.circle(rgbframe, (y + int(w/2), x), 1, (255, 0, 255), -1)
                zzz = int(point[2])
                xxx = int(point[1])
                yyy = int(point[0])
                if zzz < meanz + stdz * filter_alpha and zzz > meanz - stdz * filter_alpha:
                    cv2.circle(rgbframe, (y , x), 1, (255, 0, 255), -1)
                    zvec.append(zzz)
                    xvec.append(xxx)
                    yvec.append(yyy)

    deltax = max(xvec) - min(xvec)
    deltay = max(yvec) - min(yvec)
    deltaz = max(zvec) - min(zvec)
    perc_max95_x =  np.percentile(xvec, 99)
    perc_max05_x = np.percentile(xvec, 1)
    delta_percx = perc_max95_x-perc_max05_x
    print("len point", len(xvec),len(yvec),len(zvec))
    print("delta DOPO x y z", deltax, deltay, deltaz, "delta percentile 95-0", delta_percx)
    ratiommpx =  delta_percx/length
    print("RATIO", delta_percx/length)
    diam_cm = ratiommpx*dA
    print("DIAMETER: ", diam_cm)
    print("LENGHT: ", delta_percx)
    string1 = str("l_mm = " + str(int(delta_percx)) +" r:" + str(round(ratiommpx,3)) +" dmm:"  + str(int(diam_cm)))
    cv2.putText(orig ,string1, (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)


    pdt = pointcloud[int(tltrY), int(tltrX)]
    pdb = pointcloud[int(blbrY), int(blbrX)]
    pl_left = pointcloud[int(tlblY), int(tlblX)]
    pl_right = pointcloud[int(trbrY), int(trbrX)]
    diametro = two_points_euc_distance(pdt, pdb)  # pointcloud(y,x)
    lunghezza = two_points_euc_distance(pl_left, pl_right)  # pointcloud(y,x)

    b, g, r = cv2.split(pointcloud)  # r sembrerebbe orizzonte, x
    b = ((b - b.min()) * (1 / (b.max() - b.min()) * 255)).astype('uint8')
    g = ((g - g.min()) * (1 / (g.max() - g.min()) * 255)).astype('uint8')
    r = ((r - r.min()) * (1 / (r.max() - r.min()) * 255)).astype('uint8')


    # cv2.imshow("ora rgb", mask)
    #
    # cv2.waitKey(0)

    volume = 0

    return diametro, lunghezza


def convert_depth_image_to_pointcloud(depth_image, intrinsics):
    h, w = depth_image.shape

    pointcloud = np.zeros((h, w, 3), np.float32)

    for r in range(h):
        for c in range(w):
            distance = float(depth_image[r, c])
            result = rs.rs2_deproject_pixel_to_point(intrinsics, [c, r], distance)  # [c,r] = [x,y]
            # result[0]: right, result[1]: down, result[2]: forward

            # if abs(result[0]) > 1000.0 or abs(result[1]) > 1000.0 or abs(result[2]) > 1000.0:
            # print(result)

            pointcloud[r, c] = [int(result[2]), int(-result[0]), int(-result[1])]
            #x,y,z
    return pointcloud


def distance_med_from_masked_depth(mask, depth):
    # diam
    imask = mask < 255
    frame22 = 255 * np.ones_like(depth, np.uint8)  # all white
    frame22[imask] = depth[imask]
    distance_med = extract_medium_from_depth_segmented((depth)[imask])
    # print("ala", (depth)[imask])
    # print("DISTCYL", distance_med)

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
        # print("ratio", ratio)
        # print("ratio", ratio)
        if ratio > 0.001:
            cnt = cnt_i
    if cnt == [] and len(contours) != 0:
        # print("contours", contours)
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

    else:
        volume = 0
    # print("vol",lenght_shoot)
    return volume, (dA, dB), frame


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
    # if draw:
    #     cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    #     cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    #     cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    #     cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    #     # draw lines between the midpoints
    #     cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    #              (255, 0, 255), 2)
    #     cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
    #              (255, 0, 255), 2)
    #     # print("drawd :", (tlblX, tlblY) )
    #     # print("dimensions", dB, dA)
    # # draw the object sizes on the image

    """
    cv2.putText(orig, "{:.1f}in".format(dA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    """

    return dA, dB, orig


def crop_image_with_box_and_margin(frame, box):
    h, w, c = frame.shape

    P1 = box[0]
    P2 = box[1]
    P3 = box[2]
    P4 = box[3]
    margine = int(h / 10)
    xmax = max(P1[0], P2[0], P3[0], P4[0]) + margine
    xmin = min(P1[0], P2[0], P3[0], P4[0]) - margine
    ymax = max(P1[1], P2[1], P3[1], P4[1]) + margine
    ymin = min(P1[1], P2[1], P3[1], P4[1]) - margine
    # print(box)

    if xmin > 0 and ymin > 0:
        if xmax <= w and ymax <= h:
            #print("cropped")

            frame = frame[ymin:ymax, xmin:xmax]

            return frame

    print("impossible crop entire image", w, xmin, xmax, h, ymin, ymax)
    return frame


def rotated_box_cropper(mask, depth, rgb):
    # margin augmented
    mask = cv2.copyMakeBorder(src=mask, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT,
                              value=(255))
    depth = cv2.copyMakeBorder(src=depth, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT,
                               value=(255))
    rgb = cv2.copyMakeBorder(src=rgb, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT, value=(255))

    # calc cnt
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
            print("len cont", len(contours))
            print("image dimension:", mask.shape)

            for cnt_i in contours:
                area = cv2.contourArea(cnt_i)
                area_box = h * w
                ratio = area / area_box
                #print("ratio, area", ratio, area)

            cnt = contours[-1]
        # calc box

        rect = cv2.minAreaRect(cnt)

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        mask = (255 - mask)

        depth = cv2.bitwise_not(depth)
        # crop rot box + margin white
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(mask, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        warped_depth = cv2.warpPerspective(depth, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        warped_rgb = cv2.warpPerspective(rgb, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

        warped = (255 - warped)
        warped_depth = cv2.bitwise_not(warped_depth)
        succesful = True
        return warped, warped_depth, warped_rgb, box, succesful
    else:
        succesful = False
        return mask, depth, rgb, [], succesful


def image_splitter(frame):
    h = frame.shape[0]
    w = frame.shape[1]
    # channels = frame.shape[2]

    # decido se tagliare in altezza o larghezza
    if h > w:
        # top bottom
        half2 = h // 2

        img1 = frame[:half2, :]
        img2 = frame[half2:, :]

    else:
        # left right
        half = w // 2

        img1 = frame[:, :half]
        img2 = frame[:, half:]

    return img1, img2


def sub_box_iteration_cylindrificator(box1, frame, mask, depth, intrinsics):
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
                #print("ratio, area", ratio, area)

            cnt = contours[-1]
        # calc box

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        # cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

    draw = False
    dA, dB, frame = calc_box_legth(box, frame, draw)
    # print("dim : ", frame.shape, box)
    # area = calc area of black in image
    number_of_black_pix = np.sum(mask == 0)
    # approx diameter = wood area  / lenght of box=max(dA,dB=
    diameter_medium = number_of_black_pix / (max(dA, dB))
    # iteration needed = maxL of box / approximate diameter
    iteration_needed = int(((max(dA, dB)) / diameter_medium) / 2)


    iteration = 0



    rgb_images_collector = []
    images_collector = []
    depth_images_collector = []

    rgb_images_collector.append(frame)
    images_collector.append(mask)
    depth_images_collector.append(depth)

    all_images_in_loop = []
    all_images_in_loop.append(images_collector)

    for it in range(iteration):

        # prima ruoto e taglio
        new_collector_images = []
        new_collector_depth_images = []
        new_collector_RGB_images = []

        for im_num in range(len(images_collector)):

            rot_mask, rot_mask_depth, rot_rgb, box, succesful = rotated_box_cropper(images_collector[im_num],
                                                                                    depth_images_collector[im_num],
                                                                                    rgb_images_collector[im_num])
            if succesful != False:

                # poi splitto
                img1, img2 = image_splitter(rot_mask)
                img_d1, img_d2 = image_splitter(rot_mask_depth)
                img_rgb1, img_rgb2 = image_splitter(rot_rgb)

                new_collector_images.append(img1)
                new_collector_images.append(img2)

                new_collector_depth_images.append(img_d1)
                new_collector_depth_images.append(img_d2)

                new_collector_RGB_images.append(img_rgb1)
                new_collector_RGB_images.append(img_rgb2)


            else:
                breaking_point = True
                print(" not succesful cylkindricization, termiate frame")
                return 0, 0
        images_collector = new_collector_images
        depth_images_collector = new_collector_depth_images
        rgb_images_collector = new_collector_RGB_images

        all_images_in_loop.append(images_collector)

    # print("images : ",len(images_collector))
    if CYLINDER_SHOW:
        volumes = []
        distances = []
        #print(len(images_collector))
        for x in range(len(images_collector)):
            distance_cylinder, masked_depth = distance_med_from_masked_depth(images_collector[x],
                                                                             depth_images_collector[x])
            rgb = rotate_image_width_horizontal_max(rgb_images_collector[x])
            depthr = rotate_image_width_horizontal_max(depth_images_collector[x])
            depth_mask = rotate_image_width_horizontal_max(masked_depth)
            maskr = rotate_image_width_horizontal_max(images_collector[x])


            # print("distancce")
            try:
                distances.append(distance_cylinder)

                volume, dimensions, frame = volume_from_mask_cylinder(maskr, frame)
                boxc = calc_box_for_subcylinder_recognition(maskr)

                diametro, lunghezza = real_volume_from_pointcloud(depthr, intrinsics, boxc,
                                                                  rgb, maskr)
                #print("dimensions image: ", x, " cylinder: diam: ", diametro, "lunghezza: ", lunghezza)
                # print("volu")
            except Exception as e:
                print("e", e)
            # depth #mask #dpth mASKED #RGB

            vis = np.concatenate((rgb,cv2.cvtColor(depthr, cv2.COLOR_GRAY2BGR),cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR),cv2.cvtColor(maskr, cv2.COLOR_GRAY2BGR)), axis=0)
            cv2.imshow("im" + str(x), vis)
            cv2.moveWindow("im" + str(x), 150 * x, 150 * x)

            volumes.append(volume)



    return volumes, distances, dA, dB



def extract_medium_from_depth_segmented(depth):
    distance_medium = np.mean(depth) + OFFSET_CM_COMPRESSION

    return distance_medium


def set_white_extreme_depth_area(frame, depth, max_depth, min_depth):
    # print("depth = ", depth.shape)
    # depth = cv2.inRange(depth, min_depth-50, max_depth-50)
    depth = cv2.inRange(depth, min_depth - 50, max_depth - 50)

    # depth = cv2.inRange(depth, max_depth - 50, min_depth - 50)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # depth = cv2.morphologyEx(depth, cv2.MORPH_OPEN, kernel)
    # depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)

    result = cv2.bitwise_and(frame, frame, mask=depth)
    result[depth == 0] = [255, 255, 255]  # Turn background white

    # depth = cv2.inRange(depth, 60, 200)
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
                if circularity1 > 0.01 and circularity1 < 0.5:  # 0.05 / 0.1, 0.02
                    if area1 > 800 and area1 < 150000:  # 2200
                        if M0 > 1.15 and M0 < 100:  # 2200
                            if M01 > 0.40 and M01 < 70:  # 2200
                                if M10 > 0.5 and M10 < 70:  #
                                    if M02 > 0.25 and M02 < 40:
                                        if M20 > 0.002 and M20 < 95:
                                            if solidity > 0.01 and solidity < 1:
                                                if ratio > 0.0005 and ratio < 0.8:  # rapporto pixel contour e bounding box

                                                    # print("|____________________________________|")
                                                    # print("__|RATIO-CORRECT|__:", ratio)
                                                    print("|____________________|2nd LAYER CHOSEN", i, " area:",
                                                          int(area1), " perim:", int(perimeter1), " circul:",
                                                          round(circularity1, 5))
                                                    print(" M0:", round(M0, 3), " M01:", round(M01, 3), " M10:",
                                                          round(M10, 3), " M02:", round(M02, 3), " M20:", round(M20, 5),
                                                          "ratio", round(ratio, 5), " solidity", round(solidity, 4))
                                                    # cv2.drawContours(frame, [cnt1], 0, (0, 200, 50), 1)

                                                    # moments

                                                    Ix = M['m20']
                                                    Iy = M['m02']
                                                    b = int(pow((Iy * pow(area1, 2)) / Ix, 1 / 4))
                                                    h = int(pow((Ix * pow(area1, 2)) / Iy, 1 / 4))
                                                    x1 = cx - b / 2
                                                    x2 = cx + b / 2
                                                    y1 = cy - h / 2
                                                    y2 = cy + h / 2
                                                    # print("dim", b,h)
                                                    # print("c", cx, cy)

                                                    # frame = cv2.circle(frame, (cx,cy), 2, (255,0,0), 2)

                                                    return cnt1, frame, True
    print("________!!!!____________advanced shoots not detected")
    print("len contours", len(contours1))
    cv2.imshow("eee", mask_bu)
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






def blob_detector(frame, depth, intrinsics):
    # print("start blob")
    pixel = 0
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
    # print("end first lay blob")
    if i != 0:
        # every video should be analized from folder with specific names: A_1 ramo A condizione 1
        # save csv all in same folder in same method  A_1
        # need to implement a continous imges analyzer, to do so, use all the momentum and area perim and circularity
        # if those data are not enought use also color differenziation hsv
        # if this system is not enough use pixel

        cnt1, frame, completion = second_layer_accurate_cnt_estimator_and_draw(mask_bu, frame)

        if completion:
            # print("complietion ok,", completion)
            fit_and_draw_line_cnt(cnt1, frame)
            # draw_and_calculate_poligonal_max_diameter(cnt1, frame)
            box1 = draw_and_calculate_rotated_box(cnt1, frame)
            mask_bu, frame, depth = crop_with_box_one_shoot(box1, mask_bu, frame, depth)
            pixel = count_and_display_pixel(frame, mask_bu)
            volume, frame = volume_from_lenght_and_diam_med(box1, frame, mask_bu)
            volumes, distances, dA, dB = sub_box_iteration_cylindrificator(box1, frame, mask_bu, depth, intrinsics)
            cylindrification_results = [volumes, distances]
            return frame, mask_bu, depth, pixel, volume, cylindrification_results, True, dA, dB
        else:
            print("second layer failed, returning empty")
            return frame, mask_bu, depth, 0, 0, [[0], [0]], False, 0, 0

    else:
        return frame, mask_bu, depth, 0, 0, [[0], [0]], False, 0, 0

        # skel = skeletonize_mask(mask_bu)


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
