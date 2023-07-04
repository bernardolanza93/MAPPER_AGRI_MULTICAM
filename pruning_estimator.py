
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



print(sys.version)

ZOOM = 100
iteration = 0
THRES_VALUE = 40
ALPHA_FILTER = 1
#modifica con il tuo percorso dove ci sono i video salvati
#con slash finale, questo path deve contenere le cartelle delle singole acquisizioni
PATH_2_AQUIS = "/home/mmt-ben/Documents/calibration_video/"
PATH_HERE = os.getcwd()
OFFSET_CM_COMPRESSION = 50
KERNEL = 5
POINT_CLOUD_GRAPH = False
L_real = 337
D_real = 73


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


    return (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)


def convert_depth_image_to_pointcloud(depth_image, intrinsics):
    h, w = depth_image.shape

    pointcloud = np.zeros((h, w, 3), np.float32)

    for r in range(h): #y
        for c in range(w): #x
            distance = float(depth_image[r, c]) # [y, x]

            result = rs.rs2_deproject_pixel_to_point(intrinsics, [c, r], distance)  # [c,r] = [x,y]
            # result[0]: right, result[1]: down, result[2]: forward

            # if abs(result[0]) > 1000.0 or abs(result[1]) > 1000.0 or abs(result[2]) > 1000.0:
            # print(result)
            # z,x,y
            pointcloud[r, c] = [int(result[2]), int(-result[0]), int(-result[1])] #z,x,y
            #z,x,y
    return pointcloud

def convert_u8_img_to_u16_d435_depth_image(u8_image):

    u8_image = u8_image + OFFSET_CM_COMPRESSION
    u16_image = u8_image.astype('uint16')
    u16_image_off = u16_image
    u16_image_off_mm = u16_image_off * 10
    return u16_image_off_mm


def real_volume_from_pointcloud(pointcloud,depth_frame, intrinsics, box, rgbframe, mask):

    #  tl *--------------tltr------------------* tr
    #     |                                    |
    #     |-  tlbl                             |-  tlbr
    #     |                                    |
    #  bl *----------------blbr----------------* br
    #
    (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY) = medium_points_of_box_for_dimension_extraction(box,
                                                                                                                   rgbframe)
    A = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    B = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    (tl, tr, br, bl) = box

    if A > B:
        dB = A
        dA = B
        (tlblX, tlblY), (trbrX, trbrY), (tltrX, tltrY), (blbrX, blbrY) = medium_points_of_box_for_dimension_extraction(
            box,
            rgbframe)
        (tl, bl,br, tr) = box
        print("rotated")
    else:
        dA = A
        dB = B

    orig = rgbframe
    length = dB

    mask_copy_rect = mask.copy()
    cv2.line(orig, tl, tr, (0, 255, 0), 1)
    cv2.line(orig, bl, br, (0, 255, 0), 1)

    #erodiamo un pelo la maschera per tralasciare i valori estremi
    # Creating kernel
    kernel = np.ones((KERNEL, KERNEL), np.uint8)

    # Using cv2.erode() method

    mask = cv2.dilate(mask, kernel)
    cv2.imshow("post proc mask 4 depth", mask)
    zvec = []
    yvec = []
    xvec = []

    for x in range(pointcloud.shape[0]):
        for y in range(pointcloud.shape[1]):
            point = pointcloud[x, y]
            point_mask = mask[x, y]
            if point_mask == 0:
                if point[0] > 501:
                    #cv2.circle(rgbframe, (y + int(w/2), x), 1, (255, 0, 255), -1)
                    yyy = int(point[2])
                    xxx = int(point[1]) #ATTENZIONE CHE SONO INVERTITI!!!!!!!!
                    zzz = int(point[0])
                    zvec.append(zzz)
                    xvec.append(xxx)
                    yvec.append(yyy)

    deltax = max(xvec) - min(xvec)
    deltay = max(yvec) - min(yvec)
    deltaz = max(zvec) - min(zvec)


    meanz = round(statistics.mean(list(zvec)),2)
    stdz = round(statistics.stdev(list(zvec)),2)


    zvec = []
    yvec = []
    xvec = []
    filter_alpha = 1
    for x in range(pointcloud.shape[0]):
        for y in range(pointcloud.shape[1]):
            point = pointcloud[x, y]
            point_mask = mask[x, y]

            if point_mask == 0:
                if int(point[0]) > 501:

                    #cv2.circle(rgbframe, (y + int(w/2), x), 1, (255, 0, 255), -1)
                    yyy = int(point[2])
                    xxx = int(point[1])
                    zzz = int(point[0])
                    if zzz < meanz + stdz * filter_alpha and zzz > meanz - stdz * filter_alpha:
                        cv2.circle(rgbframe, (y , x), 1, (255, 0, 255), -1)
                        zvec.append(zzz)
                        xvec.append(xxx)
                        yvec.append(yyy)
    meanz = round(statistics.mean(list(zvec)),2)
    stdz = round(statistics.stdev(list(zvec)),2)

    #ORA CALCOLIAMO SOLO IL DELTA X MA DOVRAI FARE PITAGORA E CALCOLARE LA DIAGONALE (I TRALCI SONO STRETTI QUINDI LA DIAGONALE VERA DOVREBBE CONTARE POCO, PENSALA
    # SE IL PEZZO E INCLINATO 45 LA POINTCLOUD DEVE PRESENTARE I VALORI X E Y

    perc_max95_x = np.percentile(xvec, 99)
    perc_max05_x = np.percentile(xvec, 1)
    perc_max95_y = np.percentile(yvec, 99)
    perc_max05_y = np.percentile(yvec, 1)

    delta_percx = perc_max95_x - perc_max05_x
    delta_percy = perc_max95_y - perc_max05_y


    ratio_mm_px_L = delta_percx / length
    ratio_mm_px_D = delta_percy / dA

    ratio_real_L = L_real / length
    ratio_real_D = D_real / dA



    perc_tot = round(math.sqrt(delta_percx ** 2 + delta_percy ** 2),2)
    #print(" DELTA X", delta_percx, "DELTA y", delta_percy, "HYP", perc_tot)
    #print("Z-mean:", meanz," Z-std:", stdz)
    ratiommpx = perc_tot / length
    #print("RATIO", ratiommpx)
    diam_cm = ratiommpx * dA
    #print("DIAMETER: ", diam_cm)
    #print("LENGHT: ", perc_tot)
    string1 = str("l_mm = " + str(int(perc_tot)) + " r:" + str(round(ratiommpx, 3)) + " dmm:" + str(int(diam_cm)))
    cv2.putText(orig, string1, (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 0), 1)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 255, 0), 1)
    cv2.circle(orig, (int(tltrX), int(tltrY)), 6, (0, 255, 0), 1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 6, (0, 255, 0), 1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 6, (0, 255, 0), 1)
    cv2.circle(orig,  (int(trbrX), int(trbrY)), 6, (0, 255, 0), 1)
    #cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 255, 0), 1)
    # cv2.circle(orig, (int(tlblX), int(tlblY)), 2, (255, 0, 0), 1)
    # cv2.circle(orig, (int(trbrX), int(trbrY)), 2, (0, 255, 0), 1)
    #cv2.circle(orig, (int(tltrX), int(tltrY)), 2, (255, 0, 0), 1)
    #cv2.circle(orig, (int(blbrX), int(blbrY)), 2, (0, 255, 0), 1)


    if POINT_CLOUD_GRAPH:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.hist(zvec, bins=50)
        ax1.set_title('Z value')
        ax2.hist(xvec, bins=50)
        ax2.axvline(x=perc_max95_x, color='r', linestyle='-')
        ax2.axvline(x=perc_max05_x, color='r', linestyle='-')
        ax2.set_title('X, dim:' + str(delta_percx))
        ax3.hist(yvec, bins=50)
        ax3.axvline(x=perc_max95_y, color='r', linestyle='-')
        ax3.axvline(x=perc_max05_y, color='r', linestyle='-')
        ax3.set_title('Y, dim:' + str(delta_percy))
        fig.suptitle("len:" + str(len(zvec))  + " mean:" +  str(meanz) + " sdt:" + str(stdz) + " dim" + str(perc_tot) + "mm")

    return delta_percx, delta_percy , ratio_mm_px_L , ratio_mm_px_D , ratio_real_L , ratio_real_D , meanz



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


    rect = cv2.minAreaRect(cnt)


    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def rotate_image_width_horizontal_max(image):
    h = image.shape[0]
    w = image.shape[1]

    if h > w:
        #rotate cc
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image

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

def rotated_box_cropper(mask, depth, rgb, pointcloud):

    inh, inw, c = rgb.shape
    if inw > inh:
        widthside = True
    else:
        widthside = False

    # margin augmented
    mask = cv2.copyMakeBorder(src=mask, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT,
                              value=(255))
    depth = cv2.copyMakeBorder(src=depth, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT,
                               value=(255))
    rgb = cv2.copyMakeBorder(src=rgb, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT, value=(255))

    pointcloud = cv2.copyMakeBorder(src=pointcloud, top=15, bottom=15, left=15, right=15, borderType=cv2.BORDER_CONSTANT, value=(255))

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
            if len(contours) == 1:
                cnt = cnt_i
            else:
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
        warped_pointcloud = cv2.warpPerspective(pointcloud, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

        warped = (255 - warped)
        warped_depth = cv2.bitwise_not(warped_depth)
        succesful = True

        inh, inw, c = warped_rgb.shape
        if widthside:
            if inw < inh:
                warped_rgb = cv2.rotate(warped_rgb, cv2.ROTATE_90_CLOCKWISE)
                warped_depth = cv2.rotate(warped_depth, cv2.ROTATE_90_CLOCKWISE)
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                warped_pointcloud = cv2.rotate(warped_pointcloud, cv2.ROTATE_90_CLOCKWISE)
        else:
            if inh < inw:
                warped_rgb = cv2.rotate(warped_rgb, cv2.ROTATE_90_CLOCKWISE)
                warped_depth = cv2.rotate(warped_depth, cv2.ROTATE_90_CLOCKWISE)
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                warped_pointcloud = cv2.rotate(warped_pointcloud, cv2.ROTATE_90_CLOCKWISE)




        return warped_pointcloud,warped, warped_depth, warped_rgb, box, succesful
    else:
        succesful = False
        return pointcloud,mask, depth, rgb, [], succesful

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def calc_box_legth(box, orig, draw):


    #  tl *--------------tltr------------------* tr
    #     |                                    |
    #     |-  tlbl                             |-  tlbr
    #     |                                    |
    #  bl *----------------blbr----------------* br
    #

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

    A = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    B = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    if A < B:
        dA = A
        dB = B
    else:
        dB = A
        dA = B

    return dA, dB, orig

def sub_box_iteration_cylindrificator(box1, frame, mask, depth, intrinsics,pointcloud):
    imagem1 = (255 - mask)
    diametro, lunghezza = 0, 0


    draw = False
    dA, dB, frame = calc_box_legth(box1, frame, draw)

    rgb_images_collector = []
    images_collector = []
    depth_images_collector = []
    pointcloud_collector = []


    if iteration == 0:
        rot_pointcloud, rot_mask, rot_mask_depth, rot_rgb, box, succesful = rotated_box_cropper(mask,depth,frame, pointcloud)
        rgb_images_collector.append(rot_rgb)
        images_collector.append(rot_mask)
        depth_images_collector.append(rot_mask_depth)
        pointcloud_collector.append(rot_pointcloud)

    else:

        rgb_images_collector.append(frame)
        images_collector.append(mask)
        depth_images_collector.append(depth)
        pointcloud_collector.append(pointcloud)

        all_images_in_loop = []
        all_images_in_loop.append(images_collector)


        for it in range(iteration):

            # prima ruoto e taglio
            new_collector_images = []
            new_collector_depth_images = []
            new_collector_RGB_images = []
            new_collector_pointcloud =[]

            for im_num in range(len(images_collector)):

                rot_pointcloud, rot_mask, rot_mask_depth, rot_rgb, box, succesful = rotated_box_cropper(images_collector[im_num],
                                                                                        depth_images_collector[im_num],
                                                                                       rgb_images_collector[im_num],pointcloud_collector[im_num])

                if succesful != False:

                    # poi splitto

                    img1, img2 = image_splitter(rot_mask)
                    img_d1, img_d2 = image_splitter(rot_mask_depth)
                    img_rgb1, img_rgb2 = image_splitter(rot_rgb)
                    img_pc1, img_pc2 = image_splitter(rot_pointcloud)

                    new_collector_images.append(img1)
                    new_collector_images.append(img2)

                    new_collector_depth_images.append(img_d1)
                    new_collector_depth_images.append(img_d2)

                    new_collector_RGB_images.append(img_rgb1)
                    new_collector_RGB_images.append(img_rgb2)

                    new_collector_pointcloud.append(img_pc1)
                    new_collector_pointcloud.append(img_pc2)

                else:
                    breaking_point = True
                    print(" not succesful cylkindricization, termiate frame")
                    return 0, 0
            images_collector = new_collector_images
            depth_images_collector = new_collector_depth_images
            rgb_images_collector = new_collector_RGB_images
            pointcloud_collector = new_collector_pointcloud

            all_images_in_loop.append(images_collector)

        # print("images : ",len(images_collector))

#_________________fine iterazioni 1 step

    LLL = []
    DDD = []
    RRL = []
    RRD = []
    RPCL = []
    RPCD = []
    ZZZ = []

    for x in range(len(images_collector)):

        rgb = rgb_images_collector[x]
        depthr = depth_images_collector[x]
        maskr = images_collector[x]
        pc = pointcloud_collector[x]


        # print("distancce")
        try:

            boxc = calc_box_for_subcylinder_recognition(maskr)


            delta_percx, delta_percy , ratio_mm_px_L , ratio_mm_px_D , ratio_real_L , ratio_real_D, Z_d = real_volume_from_pointcloud(pc,depthr, intrinsics, boxc,
                                                              rgb, maskr)

            LLL.append(delta_percx)
            DDD.append(delta_percy)
            RRL.append(ratio_real_L)
            RRD.append(ratio_real_D)
            RPCL.append(ratio_mm_px_L)
            RPCD.append(ratio_mm_px_D)
            ZZZ.append(Z_d)

        except Exception as e:
            print("e", e)
        # depth #mask #dpth mASKED #RGB
        h, w, c = rgb.shape
        ax = 0
        if w < h:
            ax = 1

        visible_pointcloud = (
                    (pc - pc.min()) * (1 / (pc.max() - pc.min()) * 255)).astype('uint8')


        vis = np.concatenate((rgb,cv2.cvtColor(depthr, cv2.COLOR_GRAY2BGR),cv2.cvtColor(maskr, cv2.COLOR_GRAY2BGR),visible_pointcloud), axis=ax)
        cv2.imshow("im" + str(x), resize_image(vis,ZOOM))
        cv2.moveWindow("im" + str(x), 150 * x, 150 * x)

    if len(LLL) > 0:
        l_tot = round(sum(LLL),2)
        d_med = round(statistics.mean(DDD),2)
        r_L_real_med = round(statistics.mean(RRL),4)
        r_D_real_med = round(statistics.mean(RRD), 4)
        r_L_pc_med = round(statistics.mean(RPCL), 4)
        r_D_pc_med = round(statistics.mean(RPCD), 4)
        dep_med = round(statistics.mean(ZZZ),2)
    else:
        l_tot = 0
        d_med = 0
        r_med = 0
        dep_med = 0




    return  l_tot, d_med, r_L_real_med, r_D_real_med, r_L_pc_med, r_D_pc_med, dep_med

def draw_and_calculate_rotated_box(cnt, frame):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    return box


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
                if circularity1 > 0.44 and circularity1 < 0.48:  # 0.05 / 0.1, 0.02
                    if area1 > 800 and area1 < 150000:  # 2200
                        if M0 > 1.15 and M0 < 130:  # 2200
                            if M01 > 0.40 and M01 < 70:  # 2200
                                if M10 > 0.5 and M10 < 140:  #
                                    if M02 > 0.25 and M02 < 40:
                                        if M20 > 0.002 and M20 < 150:
                                            if solidity > 0.96 and solidity < 0.999:
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



intrinsics = obtain_intrinsics()
L_ = []
D_ = []
RRLLL = []
RRDDD = []
RPCLLL = []
RPCDDD = []
Z_all = []

ii = 0
print("total file: ", len(os.listdir(PATH_2_AQUIS)))

for folders in os.listdir(PATH_2_AQUIS):
    ii = ii + 1

    print("files number:", ii, " name:", os.listdir(PATH_2_AQUIS))
    folder_name = folders
    #videos = os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name)
    #writeCSVdata(folder_name, ["frame", "pixel", "volume", "distance_med", "volumes", "distances"])

    for videos in os.listdir(PATH_2_AQUIS + folder_name):

        #print(videos)
        print("ITERATION:", folder_name)

        if videos.endswith(".mkv"):
            print(videos.split(".")[0])

            if videos.split(".")[0] == "RGB":

                path_rgb= os.path.join(PATH_2_AQUIS + folder_name, videos)
                #creo l oggetto per lo streaming
                video1 = cv2.VideoCapture(path_rgb)
            elif videos.split(".")[0] == "DEPTH":

                path_depth= os.path.join(PATH_2_AQUIS + folder_name, videos)
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




    while (video1.isOpened() and video2.isOpened()):


        plt.close('all')
        ret, frame = video1.read()
        ret2, frame2 = video2.read()
        nrfr = nrfr + 1
        if nrfr > 5:

            if ret == True and ret2 == True:

                try:
                    #gestionn depth
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(gray, THRES_VALUE, 255, cv2.THRESH_BINARY)
                    cv2.imshow("initial mask", resize_image(mask,50))



                    #try:




                    cnt1, frame, completion = second_layer_accurate_cnt_estimator_and_draw(mask, frame)

                    rect = cv2.minAreaRect(cnt1)

                    frame = crop_with_rect_rot(frame,rect)
                    frame2 = crop_with_rect_rot(frame2,rect)
                    mask = crop_with_rect_rot(mask,rect)







                    frame2_u16 = convert_u8_img_to_u16_d435_depth_image(frame2)
                    pointcloud = convert_depth_image_to_pointcloud(frame2_u16, intrinsics)


                    box1 = draw_and_calculate_rotated_box(cnt1, frame)
                    l_tot, d_med, r_L_real_med, r_D_real_med, r_L_pc_med, r_D_pc_med, dep_med  = sub_box_iteration_cylindrificator(box1, frame, mask, frame2, intrinsics,pointcloud)
                    # L e D dovrebbero essere costanti
                    L_.append(l_tot)
                    D_.append(d_med)
                    RRLLL.append(r_L_real_med)
                    RRDDD.append(r_D_real_med)
                    RPCLLL.append(r_L_pc_med)
                    RPCDDD.append(r_D_pc_med)
                    Z_all.append(dep_med)

                except Exception as e:
                     print("processing error:", e)


                if POINT_CLOUD_GRAPH:
                    plt.draw()
                    plt.pause(0.001)
                #cv2.imshow("cdscsdc", mask)
                cv2.imshow("or", resize_image(frame,50))

                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    sys.exit()
                    break
            else:
                break
        else:
            continue

    video1.release()
    video2.release()
    cv2.destroyAllWindows()

    print("COMPLETED ", nrfr," frames")

plt.close('all')

L_.append(l_tot)
D_.append(d_med)
RRLLL.append(r_L_real_med)
RRDDD.append(r_D_real_med)
RPCLLL.append(r_L_pc_med)
RPCDDD.append(r_D_pc_med)
Z_all.append(dep_med)



fig,  ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(L_)
ax1.set_title('lenght medium =' + str(round(statistics.mean(L_),2)))
ax1.axhline(y= statistics.mean(L_), color='r', linestyle='-')
ax1.axhline(y= L_real, color='g', linestyle='--')

ax2.plot(D_)
ax2.set_title('diameter medium =' + str(round(statistics.mean(D_),2)))
ax2.axhline(y=statistics.mean(D_), color='r', linestyle='-')
ax2.axhline(y= D_real, color='g', linestyle='--')


ax3.scatter(Z_all,RRLLL, color='b')
#ax3.scatter(Z_all,RPCLLL, color='b')
ax3.grid(True)
#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(Z_all, RPCLLL, 1)


mrr, brr = np.polyfit(Z_all, RRLLL, 1)

y_estimated = [(x * mrr) + brr for x in Z_all]
y_residuals = [a_i - b_i for a_i, b_i in zip(y_estimated, RRLLL)]
squared_res = [i ** 2 for i in y_residuals]
MSE = statistics.mean(squared_res)
RMSE = math.sqrt(MSE)
print("mse rmse", MSE, RMSE)


r_squared =  r2_score(RRLLL, y_estimated)

#view R-squared value
print("R squared = ", r_squared)





mr, br = np.polyfit(Z_all, RRLLL, 1, cov=True)
dm = np.sqrt(br[0][0])
db = np.sqrt(br[1][1])

print("m: {} +/- {}".format(mr[0], np.sqrt(br[0][0])))
print("b: {} +/- {}".format(mr[1], np.sqrt(br[1][1])))


ax3.set_title("R = m*Z + b : RMSE=" + str(round(RMSE,4))+ " \n m="  + str(round(mrr,7)) + "+/-" +str(round(dm,7)) + "; b="+ str(round(brr,3))+ "+/-" +str(round(db,3)))

#add linear regression line to scatterplot
Z_all = [float(i) for i in Z_all]
#ax3.plot(Z_all,[x * m for x in Z_all]+b,color='g', linestyle='--')
ax3.plot(Z_all,[x * mrr for x in Z_all]+brr, color='r', linestyle='-.')
ax3.set_xlabel('Z-depth [mm]')
ax3.set_ylabel('ratio [mm/px]')

print("m, b", mrr , brr)



ax4.scatter(Z_all,RRDDD, color='r')
ax4.set_title('D) RRmean:' + str(round(statistics.mean(RRDDD),2)) + 'RPC:' + str(round(statistics.mean(RPCDDD),2)) + "P mean" + str(round(statistics.mean(Z_all),2)))
ax4.scatter(Z_all,RPCDDD, color='b')
ax4.legend()
ax4.grid(True)





#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(Z_all, RPCDDD, 1)
mr, br = np.polyfit(Z_all, RRDDD, 1)
#add linear regression line to scatterplot
ax4.plot(Z_all,[x * m for x in Z_all]+b,color='g', linestyle='--')
ax4.plot(Z_all,[x * mr for x in Z_all]+br,color='r', linestyle='--')


folder_path = 'results'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created folder: {folder_path}")
else:
    print(f"Folder already exists: {folder_path}")


#save data

header = ['length_px', 'diameter_px', 'depth', 'ratio_pixel_length','ratio_pixel_diameter']

file_path = 'results/calibration_data.csv'

with open(file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row
    writer.writerow(header)

    # Write data columns
    for data in zip(L_, D_, Z_all,RRLLL,RRDDD ):
        writer.writerow(data)

print(f"CSV file '{file_path}' created successfully.")

#save results

header = ['m', 'm_uncert', 'b', 'b_uncert','RMSE','R_squared' ]

file_path = 'results/calibration_results.csv'

with open(file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row
    writer.writerow(header)

    # Write data columns
    data = [mrr, dm, brr, db,RMSE, r_squared]
    writer.writerow(data)

print(f"CSV file '{file_path}' created successfully.")



print("m, b", mr , br)



fig.suptitle("analysis of results")

plt.show()














