
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



print(sys.version)

ZOOM = 100
iteration = 2
THRES_VALUE = 70
ALPHA_FILTER = 1
PATH_2_AQUIS = "/aquisition/"
PATH_HERE = os.getcwd()
OFFSET_CM_COMPRESSION = 50
KERNEL = 5
POINT_CLOUD_GRAPH = False
L_real = 315
D_real = 73
CONTINOUS_STREAM = 0

# ML = 0.00075867178
# BL = 0.0333853017
# MD = 0.000788289686
# BD = -0.010275683330683


def simple_geometrical_analisys(allx,ally,allz):
    # Calculate percentiles
    x_01 = np.percentile(allx, 1)
    x_99 = np.percentile(allx, 99)
    y_01 = np.percentile(ally, 1)
    y_99 = np.percentile(ally, 99)
    z_01 = np.percentile(allz, 1)
    z_99 = np.percentile(allz, 99)
    z_50 = np.percentile(allz, 50)

    # Calculate differences
    print("x1 ", x_01, ", x2 ", x_99)
    diff_x = x_99 - x_01
    diff_y = y_99 - y_01
    diff_z = z_99 - z_01

    # plt.hist(allx, bins=50)
    # plt.show()

    # Calculate the sum vector using the Pythagorean theorem
    sum_vector = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)

    # Print the results
    print("Difference for X:", diff_x)
    print("Difference for Y:", diff_y)
    print("Difference for Z:", diff_z)
    print("Sum Vector:", sum_vector)
    print("perc1z", z_01)
    print("perc99z", z_99)
    print("perc50z", z_50)




def clear_image_outer_box(box,mask):
    # Get image dimensions
    height, width = mask.shape[:2]

    # Create a mask with the same dimensions as the image
    backgorund = np.zeros((height, width), dtype=np.uint8)
    # Fill the mask with white pixels inside the box region
    cv2.rectangle(backgorund, (box[0], box[1]), (box[2], box[3]), 255, -1)
    # Bitwise AND the image with the mask to set pixels outside the box to white
    result = cv2.bitwise_and(mask, mask, mask=backgorund)
    return result


def detect_rotated_box(mask,frame):
    cnt1, completion = second_layer_accurate_cnt_estimator_and_draw(mask)
    rect = cv2.minAreaRect(cnt1)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    (tl, tr, br, bl) = box

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.line(frame, tl, tr, (0, 255, 0), 1)
    cv2.line(frame, bl, br, (0, 255, 0), 1)
    cv2.line(frame, tl, bl, (0, 255, 0), 1)
    cv2.line(frame, tr, br, (0, 255, 0), 1)



    return rect, cnt1


def volume(L,d):
    if L is not None and d is not None:
        vol = (math.pi * pow(d,2) * L ) / 4
        return volume
    else:
        return 0



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





def convert_depth_image_to_pointcloud(depth_image, intrinsics, frame):
    h, w = depth_image.shape

    pointcloud = np.zeros((h, w, 3), np.float32)
    allz = []
    allx = []
    ally = []

    for r in range(h): #y
        for c in range(w): #x
            distance = float((depth_image[r, c] + 50 ) * 10) # [y, x]

            if distance > 600:
                # if distance > 1110:
                # else:
                #     cv2.circle(frame, (c, r), 1, (0, 0, 255), 1)

                result = rs.rs2_deproject_pixel_to_point(intrinsics, [c, r], distance)  # [c,r] = [x,y]
                # result[0]: right, result[1]: down, result[2]: forward


                # if abs(result[0]) > 1000.0 or255 abs(result[1]) > 1000.0 or abs(result[2]) > 1000.0:
                # print(result)
                # z,x,y
                pointcloud[r, c] = [int(result[2]), int(-result[0]), int(-result[1])] #z,x,y
                if int(-result[0]) % 10 == 0:
                    cv2.circle(frame, (c, r), 1, (255, 0, 0), 1)

                if int(-result[1]) % 10 == 0:
                    cv2.circle(frame, (c, r), 1, (0, 255, 0), 1)
                if abs(int(-result[0]) ) > 400:
                    cv2.circle(frame, (c, r), 1, (0, 0, 255), 1)





                allz.append(int(result[2]))
                allx.append(int(-result[0]))
                ally.append(int(-result[1]))



                #z,x,y
                #pointcloud[r,c] = [0,0,0]
    return pointcloud, allz, allx, ally

def convert_u8_img_to_u16_d435_depth_image(u8_image):

    u8_image = u8_image + OFFSET_CM_COMPRESSION
    u16_image = u8_image.astype('uint16')
    u16_image_off = u16_image
    u16_image_off_mm = u16_image_off * 10
    return u16_image_off_mm



def rotate_image_width_horizontal_max(image):
    h = image.shape[0]
    w = image.shape[1]

    if h > w:
        #rotate cc
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image

def rotated_box_cropper(mask, depth, rgb, pointcloud):

    inh, inw, c = rgb.shape
    if inw > inh:
        widthside = True
    else:
        widthside = False


    BA4D = 3 #BORDER_AUGENTATION_4_DETECTION = 3

    # margin augmented
    mask = cv2.copyMakeBorder(src=mask, top=BA4D, bottom=BA4D, left=BA4D, right=BA4D, borderType=cv2.BORDER_CONSTANT,
                              value=(255))
    depth = cv2.copyMakeBorder(src=depth, top=BA4D, bottom=BA4D, left=BA4D, right=BA4D, borderType=cv2.BORDER_CONSTANT,
                               value=(255))
    rgb = cv2.copyMakeBorder(src=rgb, top=BA4D, bottom=BA4D, left=BA4D, right=BA4D, borderType=cv2.BORDER_CONSTANT, value=(255))

    pointcloud = cv2.copyMakeBorder(src=pointcloud, top=BA4D, bottom=BA4D, left=BA4D, right=BA4D, borderType=cv2.BORDER_CONSTANT, value=(255))

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

def draw_and_calculate_rotated_box(cnt, frame):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    return box


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

            if perimeter1 > 200 and perimeter1 < 7000:  # 1200
                if circularity1 > 0.01 and circularity1 < 0.6:  # 0.05 / 0.1, 0.02
                    if area1 > 800 and area1 < 150000:  # 2200
                        if M0 > 1.00 and M0 < 130:  # 2200
                            if M01 > 0.01 and M01 < 70:  # 2200
                                if M10 > 0.01 and M10 < 140:  #
                                    if M02 > 0.25 and M02 < 40:
                                        if M20 > 0.0002 and M20 < 150:
                                            if solidity > 0.1 and solidity < 0.5:
                                                if ratio > 0.00005 and ratio < 0.8:  # rapporto pixel contour e bounding box

                                                    # print("|____________________________________|")
                                                    # print("__|RATIO-CORRECT|__:", ratio)
                                                    print("|____________________|2nd LAYER CHOSEN", i, " area:",
                                                          int(area1), " perim:", int(perimeter1), " circul:",
                                                          round(circularity1, 5))
                                                    print(" M0:", round(M0, 3), " M01:", round(M01, 3), " M10:",
                                                          round(M10, 3), " M02:", round(M02, 3), " M20:", round(M20, 5),
                                                          "ratio", round(ratio, 5), " solidity", round(solidity, 4))
                                                    cv2.drawContours(frame, [cnt1], 0, (200, 200, 50), 1)

                                                    return cnt1, True
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
    return 0, False



intrinsics = obtain_intrinsics()
L_ = []
D_ = []
RRLLL = []
RRDDD = []
RPCLLL = []
RPCDDD = []
Z_all = []

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


        plt.close('all')
        ret, frame = video1.read()
        ret2, frame2 = video2.read()
        nrfr = nrfr + 1
        if nrfr > 5:

            if ret == True and ret2 == True:


                #gestionn depthimg
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((5, 5), np.float32) / 25
                #frame2of = cv2.filter2D(frame2, -1, kernel)
                #frame2of = cv2.GaussianBlur(frame2,(31,31),0) #medio
                #frame2of = cv2.blur(frame2, (51, 51)) #buono
                #frame2of = cv2.medianBlur(frame2, 51) #ottimo ####
                #frame2of = cv2.bilateralFilter(frame2, 15, 75, 75)
                frame2of = frame2

                ###!!!! BLURRA SOLO I VALORI INTERNI ALLA MASCHERA:
                #prendi i valori dentro la maschera (media o simile) ed estendili a tutta l immagine

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(gray, THRES_VALUE, 255, cv2.THRESH_BINARY)
                rect_point_of_interest, cnt = detect_rotated_box(mask,frame)

                #make white out of the bounfing refct
                x, y, w, h = cv2.boundingRect(cnt)
                white_bg = 255 * np.ones_like(mask)
                roi = mask[y:y + h, x:x + w]
                white_bg[y:y + h, x:x + w] = roi

                mask = white_bg



                mask_inverted = cv2.bitwise_not(mask)
                #pulisco da grigi nuova maschera
                #ret, mask_inverted = cv2.threshold(mask_inverted, 120, 255, cv2.THRESH_BINARY)
                #mask_inverted[mask_inverted > 0] = 255

                #maschero il depth con la maschera rgb
                black_out_depth_in = cv2.bitwise_and(frame2of, frame2of, mask=mask_inverted)
                #calcolo valore medio interno alla maschera
                avg_masked_value =  int(cv2.mean(frame2of, mask=mask_inverted) [0])
                print("depth avarage", avg_masked_value)
                #riempi l esterno con il valore medio
                black_out_depth_in_AVARAGED = black_out_depth_in.copy()
                #copio l immagine per editarla
                black_out_depth_in_AVARAGED[np.where((black_out_depth_in_AVARAGED == [0]))] = [avg_masked_value]
                #ora blurriamo e poi riaplicchiamo la maschera
                black_out_depth_in_blurred = cv2.medianBlur(black_out_depth_in_AVARAGED, 21)
                #riapplico la maschera sull immagine blurrata di depth con background stabile
                black_out_depth_in_remasked = cv2.bitwise_and(black_out_depth_in_blurred, black_out_depth_in_blurred, mask=mask_inverted)


                #extract only the pixels inside the ,ask:
                #qui stiamo cercando di visualizzare quanti pixel neri ci sono all interno della maschera che corrispondono al rumore TO DO
                pixels = frame2of[[mask_inverted]]
                arr = np.array(pixels)


                #black_out_dapth_in = frame2of * mask_inverted


                pc, allz,allx,ally = convert_depth_image_to_pointcloud(black_out_depth_in_remasked,intrinsics,frame)
                visible_pointcloud = ((pc - pc.min()) * (1 / (pc.max() - pc.min()) * 255)).astype('uint8')
                simple_geometrical_analisys(allx,ally,allz)

                #crop for imaging
                cropped_masked_depth = crop_with_rect_rot(black_out_depth_in_AVARAGED,rect_point_of_interest)
                cropped_frame = crop_with_rect_rot(frame, rect_point_of_interest)
                mask_inverted_crop = crop_with_rect_rot(mask_inverted, rect_point_of_interest)

                plt.hist(arr, bins=25)
                plt.show()

                cv2.imshow("or", resize_image(cropped_frame,300))
                cv2.imshow("initial mask", resize_image(black_out_depth_in_remasked,60))

                cv2.imshow("initial frame2", resize_image(cropped_masked_depth, 300))



                key = cv2.waitKey(CONTINOUS_STREAM)
                if key == ord('q') or key == 27:
                    sys.exit()


            else:
                break
        else:
            continue

    video1.release()
    video2.release()
    cv2.destroyAllWindows()

    print("COMPLETED ", nrfr," frames")

plt.close('all')





fig,  ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(L_)
ax1.set_title('lenght medium =' + str(round(statistics.mean(L_),2)))
ax1.axhline(y= statistics.mean(L_), color='r', linestyle='-')


ax2.plot(D_)
ax2.set_title('diameter medium =' + str(round(statistics.mean(D_),2)))
ax2.axhline(y=statistics.mean(D_), color='r', linestyle='-')





#obtain m (slope) and b(intercept) of linear regression lin
#add linear regression line to scatterplot
Z_all = [float(i) for i in Z_all]



deviation  = [x - L_real for x in L_]
absolute_dev = [abs(ele) for ele in deviation]
squares = [i*i for i in deviation]
RMSE = np.sqrt(np.mean(squares))

ax3.hist(deviation, color='g')
ax3.legend()
ax3.grid(True)
ax3.set_xlabel('error [mm]')
ax3.set_ylabel('frequency')


ax4.scatter(Z_all,absolute_dev, color='b')
ax4.set_title('RMSE= ' + str(round(RMSE,2)) + " mm")
ax4.set_xlabel('Z [mm]')
ax4.set_ylabel('error [mm]')

#ax4.scatter(Z_all,RPCDDD, color='b')
ax4.legend()
ax4.grid(True)


fig.suptitle("analysis of results")

plt.show()



folder_path = 'results'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created folder: {folder_path}")
else:
    print(f"Folder already exists: {folder_path}")




#save data

header = ['L_estimated', 'D_estimated','depth', 'L_deviation']

file_path = 'results/length_results.csv'

with open(file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row
    writer.writerow(header)

    # Write data columns
    for data in zip(L_, D_, Z_all,deviation):
        writer.writerow(data)

print(f"CSV file '{file_path}' created successfully.")



#save results

header = ['RMSE', 'L_med', 'D_med']

file_path = 'results/statistical_results.csv'

with open(file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row
    writer.writerow(header)

    # Write data columns
    data = [RMSE, round(statistics.mean(L_),4), round(statistics.mean(D_),4)]
    writer.writerow(data)

print(f"CSV file '{file_path}' created successfully.")












