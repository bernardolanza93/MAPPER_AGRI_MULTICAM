
from scipy.spatial import distance as dist
import os
import cv2
import math
import time
import numpy as np
import statistics
import pyrealsense2 as rs
import csv
import sys


print(sys.version)



THRES_VALUE = 30
PATH_2_AQUIS = "/aquisition/"
PATH_HERE = os.getcwd()
OFFSET_CM_COMPRESSION = 50



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




def convert_u8_img_to_u16_d435_depth_image(u8_image):
    # print(u8_image)
    u16_image = u8_image.astype('uint16')
    u16_image_off = u16_image + OFFSET_CM_COMPRESSION
    u16_image_off_mm = u16_image_off * 10
    # print(u16_image_off_mm)
    # print(u16_image_off_mm.shape, type(u16_image_off_mm),u16_image_off_mm.dtype)

    return u16_image_off_mm




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





    mask_copy_rect = mask.copy()
    cv2.rectangle(rgbframe, tl, br, (255, 0, 65), 2)



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



    #ORA CALCOLIAMO SOLO IL DELTA X MA DOVRAI FARE PITAGORA E CALCOLARE LA DIAGONALE (I TRALCI SONO STRETTI QUINDI LA DIAGONALE VERA DOVREBBE CONTARE POCO, PENSALA
    # SE IL PEZZO E INCLINATO 45 LA POINTCLOUD DEVE PRESENTARE I VALORI X E Y
    deltax = max(xvec) - min(xvec)
    deltay = max(yvec) - min(yvec)
    deltaz = max(zvec) - min(zvec)
    perc_max95_x =  np.percentile(xvec, 99)
    perc_max05_x = np.percentile(xvec, 1)
    delta_percx = perc_max95_x-perc_max05_x

    print("delta DOPO x y z", deltax, deltay, deltaz, "delta percentile 95-0", delta_percx)
    ratiommpx =  delta_percx/length
    print("RATIO", delta_percx/length)
    diam_cm = ratiommpx*dA
    print("DIAMETER: ", diam_cm)
    print("LENGHT: ", delta_percx)
    string1 = str("l_mm = " + str(int(delta_percx)) +" r:" + str(round(ratiommpx,3)) +" dmm:"  + str(int(diam_cm)))
    cv2.putText(orig ,string1, (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)



    return diam_cm, delta_percx



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

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # draw the midpoints on the imageà
    # if draw:

    return dA, dB, orig







def sub_box_iteration_cylindrificator(box1, frame, mask, depth, intrinsics):
    imagem1 = (255 - mask)


    draw = False
    dA, dB, frame = calc_box_legth(box1, frame, draw)
    # print("dim : ", frame.shape, box)
    # area = calc area of black in image

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



    #print(len(images_collector))
    for x in range(len(images_collector)):

        rgb = rotate_image_width_horizontal_max(rgb_images_collector[x])
        depthr = rotate_image_width_horizontal_max(depth_images_collector[x])
        maskr = rotate_image_width_horizontal_max(images_collector[x])


        # print("distancce")
        try:



            boxc = calc_box_for_subcylinder_recognition(maskr)

            diametro, lunghezza = real_volume_from_pointcloud(depthr, intrinsics, boxc,
                                                              rgb, maskr)

        except Exception as e:
            print("e", e)
        # depth #mask #dpth mASKED #RGB

        vis = np.concatenate((rgb,cv2.cvtColor(depthr, cv2.COLOR_GRAY2BGR),cv2.cvtColor(maskr, cv2.COLOR_GRAY2BGR)), axis=0)
        cv2.imshow("im" + str(x), vis)
        cv2.moveWindow("im" + str(x), 150 * x, 150 * x)





    return  diametro, lunghezza




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
                                                    cv2.drawContours(frame, [cnt1], 0, (0, 200, 50), 1)

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


for folders in os.listdir(PATH_HERE + PATH_2_AQUIS):
    print("files:", os.listdir(PATH_HERE + PATH_2_AQUIS))
    folder_name = folders
    #videos = os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name)
    #writeCSVdata(folder_name, ["frame", "pixel", "volume", "distance_med", "volumes", "distances"])

    for videos in os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name):

        #print(videos)
        print("ITERATION:", folder_name)




        if videos.endswith(".mkv"):
            print(videos.split(".")[0])

            if videos.split(".")[0] == "RGB":

                path_rgb= os.path.join(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name, videos)
                #creo l oggetto per lo streaming
                video1 = cv2.VideoCapture(path_rgb)
            elif videos.split(".")[0] == "DEPTH":

                path_depth= os.path.join(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name, videos)
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
        ret, frame = video1.read()
        ret2, frame2 = video2.read()

        if ret == True and ret2 == True:
            #gestionn depth
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)



            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray, THRES_VALUE, 255, cv2.THRESH_BINARY)

            cnt1, frame, completion = second_layer_accurate_cnt_estimator_and_draw(mask, frame)
            box1 = draw_and_calculate_rotated_box(cnt1, frame)
            dA, dB = sub_box_iteration_cylindrificator(box1, frame, mask, frame2, intrinsics)



            cv2.imshow("cdscsdc", mask)
            cv2.imshow("or", frame)

            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                sys.exit()
                break
        else:
            break

    video1.release()
    video2.release()
    cv2.destroyAllWindows()





