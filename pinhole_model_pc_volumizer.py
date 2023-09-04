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
from configuration_path import *
import open3d as o3d
from sklearn.preprocessing import RobustScaler
import pyntcloud
import pandas as pd


print(sys.version)

#allora invece che stimare d e l stiamo stimando le componenti verticali e orizz di un ramo, che se inclinato a 45 gradi fanno risultare il diametro  = L e quindi sballano di brutto! (


print("PATH! ", PATH_EXAMPLE)


def remove_far_points(point_cloud, std_threshold=1.0):
    # Calculate the mean and standard deviation of x, y, and z coordinates
    means = np.mean(point_cloud, axis=0)
    stds = np.std(point_cloud, axis=0)

    # Define a mask to filter out points that are far from the mean
    mask = np.all(np.abs((point_cloud - means) / stds) <= std_threshold, axis=1)

    # Apply the mask to keep points that are not far from the mean
    filtered_cloud = point_cloud[mask]

    print("removed: ", len(point_cloud) - len(filtered_cloud), "/",len(point_cloud), " far points")

    return filtered_cloud


def radius_outlier_removal(point_cloud, radius_threshold = 3, min_neighbors =8):
    # Calculate the distance to the k-nearest neighbors
    kdtree = cKDTree(point_cloud)
    distances, _ = kdtree.query(point_cloud, k=min_neighbors + 1)

    # Calculate mean distance to neighbors excluding itself
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # Apply radius outlier removal
    outlier_indices = mean_distances > radius_threshold
    #print(outlier_indices)
    filtered_cloud = point_cloud[~outlier_indices]

    print("removed:", len(point_cloud) - len(filtered_cloud), "/",len(point_cloud), " outl. points")

    return filtered_cloud


def z_score_normalization(point_cloud, threshold_factor=0.01):
    # Extract z-values
    z_values = point_cloud[:, 2]

    # Calculate mean and standard deviation of z-values
    z_mean = np.mean(z_values)
    z_std = np.std(z_values)

    # Calculate z-scores
    z_scores = (z_values - z_mean) / z_std

    # Apply a correction to the z-values based on their z-scores
    corrected_z_values = z_mean + (z_scores * threshold_factor * z_std)

    # Update the z-values in the point cloud
    corrected_point_cloud = np.copy(point_cloud)
    corrected_point_cloud[:, 2] = corrected_z_values



    return corrected_point_cloud


#def normalize_all_z_with_z_mean()




def show_PCA(x,y,z):




    # Stack the arrays horizontally to create the final matrix
    points_mm = np.column_stack((x, y, z))

    points_mm_or = points_mm
    #
    #

    points_mm = radius_outlier_removal(points_mm)
    points_mm = remove_far_points(points_mm)
    points_mm = z_score_normalization(points_mm)

    #plot_histogram(points_mm[:,2])


    #points_mm = remove_points_if_z_is_out_of_std(points_mm)
    #points_mm = z_score_normalization(points_mm)

    # Calculate the centroid
    centroid = np.mean(points_mm, axis=0)

    # Set the centroid color
    centroid_color = [1.0, 0.0, 0.0]  # Red color



    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_mm)

    # Create an Open3D point cloud object
    pcd_or = o3d.geometry.PointCloud()
    pcd_or.points = o3d.utility.Vector3dVector(points_mm_or)

    # Calculate the oriented bounding box
    obb = pcd.get_oriented_bounding_box()

    # Get the dimensions and transformation matrix of the oriented bounding box
    dimensions = obb.extent
    transformation_matrix = obb.R

    # Extract dimensions along x, y, and z
    length = dimensions[0]
    width = dimensions[1]
    height = dimensions[2]




    if SHOW_PC_WITH_BOX:
        print("Estimated L (mm):", length)
        print("Estimated H (mm):", height)
        print("Estimated W (mm):", width)
        print("____________________________")

        # Create a 3D grid using Open3D
        box = 1000
        grid_resolution = 100
        x_grid = np.arange(0, box, grid_resolution)
        y_grid = np.arange(0, box, grid_resolution)
        z_grid = np.arange(0, box, grid_resolution)
        x_grid, y_grid, z_grid = np.meshgrid(x_grid, y_grid, z_grid)
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))

        # Create a grid point cloud
        grid_pcd = o3d.geometry.PointCloud()
        grid_pcd.points = o3d.utility.Vector3dVector(grid_points)

        # Create a visualization object
        vis = o3d.visualization.Visualizer()

        # Add the point clouds and oriented bounding box
        vis.create_window()
        vis.add_geometry(pcd)
        #vis.add_geometry(pcd_or)
        vis.add_geometry(obb)
        # Add the grid point cloud
        #vis.add_geometry(grid_pcd)

        # Add a sphere to represent the centroid
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        centroid_sphere.translate(centroid)
        centroid_sphere.paint_uniform_color(centroid_color)
        vis.add_geometry(centroid_sphere)

        # Set background color (light gray)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.8, 0.8, 0.8])

        # Visualize the scene
        vis.run()
        vis.destroy_window()

    return length,width



def delete_csv_file(filenam_path):
    # Delete the existing CSV file (if it exists)
    if os.path.exists(filenam_path):
        print("REMOVE:", filenam_path)
        os.remove(filenam_path)
    else:
        print(filenam_path, " no such file in memory")


def write_number_at_center(image, number):
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the position to write the number at the center
    text = str(int(number))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Write the number at the center of the image
    color = (0, 255, 0)  # Red color in BGR format (OpenCV uses BGR)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

    return image


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time of {func.__name__}: {processing_time:.6f} seconds")
        return result

    return wrapper


def plot_histogram(values):
    plt.hist(values, bins=100, edgecolor='black')
    mean_vol = np.mean(values)
    plt.vlines(mean_vol, 0, 50, linestyles ="dashed", colors ="g")
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values, mean:' +  str(int(mean_vol)) +  " STD:"  +  str(int(np.std(values))))
    plt.show()


def save_to_csv_recursive(value, file_path):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([value])





def resize_image_imperfection_for_vconcaten(images):
    # Find the maximum width among all images
    max_width = max(image.shape[1] for image in images)

    # Resize images to have the same width
    resized_images = []
    for image in images:
        if image.shape[1] < max_width:
            pad_right = max_width - image.shape[1]
            resized_image = cv2.copyMakeBorder(image, 0, 0, 0, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            resized_image = image.copy()
        resized_images.append(resized_image)

    # Concatenate the resized images vertically

    return resized_images


def draw_histograms_and_calc_dimension(x_values, y_values, z_values, show=False):
    if show:
        plt.figure(figsize=(15, 5))

    # print(x_values)
    means = []
    deltas = []

    for idx, (values, title) in enumerate(zip([x_values, y_values, z_values], ['X', 'Y', 'Z'])):
        # Calculate the 95th and 5th percentiles
        percentile_95 = np.percentile(values, 98)
        percentile_5 = np.percentile(values, 2)
        mean_i = np.mean(values)

        # Calculate the delta between 95th and 5th percentiles
        delta = percentile_95 - percentile_5

        if show:
            plt.subplot(131 + idx)
            plt.hist(values, bins=50, color='C{}'.format(idx), alpha=0.7)
            plt.title(f'Histogram of {title} values\nDelta: {delta:.2f}')
            plt.xlabel(f'{title} Values')
            plt.ylabel('Frequency')
        deltas.append(delta)
        means.append(mean_i)
    dx = deltas[0]
    dy = deltas[1]
    dz = deltas[2]
    z_med = means[2]
    if show:
        plt.tight_layout()
        plt.show()
    return dx, dy, dz, z_med


def remove_x_and_y_for_zero_z(z_list, x_list, y_list):
    new_z = []
    new_x = []
    new_y = []
    def_z = []
    def_x = []
    def_y = []

    for z, x, y in zip(z_list, x_list, y_list):

        if z > MIN_DEPTH:
            new_z.append(z)
            new_x.append(x)
            new_y.append(y)

    return new_z,new_x,new_y



def dimension_image_list_analyzer(image_list, RGB_list):

    dxx = []
    dyy = []
    vol_all = []
    dyy_expected = []
    vol_expected_from_RGB = []
    for i in range(len(image_list)):

        image = image_list[i]
        imageRGB = RGB_list[i]
        # Split the image into its individual channels
        draw_system_all_r = []
        draw_system_all_c = []
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                # Extract blue channel value of the pixel in the first image
                depth = image[r, c, 0]

                # Check if the blue channel value is less than 200

                # print(depth)
                if depth > MIN_DEPTH:
                    draw_system_all_r.append(r)
                    draw_system_all_c.append(c)


        # Find the starting point (top-left corner) of the rectangle
        # print(len(draw_system_all_c), len(draw_system_all_r))

        start_point = (min(draw_system_all_c), min(draw_system_all_r))
        width = max(draw_system_all_c) - min(draw_system_all_c)
        height = max(draw_system_all_r) - min(draw_system_all_r)

        # Calculate the dimensions of the rectangle
        ratio_cylinder = height / width

        z, x, y = cv2.split(image)

        z = z.flatten().tolist()
        x = x.flatten().tolist()
        y = y.flatten().tolist()

        z, x, y = remove_x_and_y_for_zero_z(z, x, y)
        # Create a mask where B is not equal to zero
        mask = (z != 0)

        # Apply the mask to G and R channels
        x[mask == 0] = 0
        y[mask == 0] = 0

        # cv2.imshow("edwd",imageRGB)
        # key = cv2.waitKey(CONTINOUS_STREAM)
        # if key == ord('q') or key == 27:
        #     break

        #display3Dpointcloud(x,y,z)


        dx, dy, dz, z_med = draw_histograms_and_calc_dimension(x, y, z, False)

        try:
            l,w = show_PCA(x,y,z)

        except Exception as e:
            print(e)
            l,w = 0,0
        # print("expected",expected_from_rgb_contours_dy_diameter,' real',dy, " length", dx)

        dxx.append(l)  # lungheezaa
        dyy.append(w)  # diametro



        vol = volume(l, w)

        vol_all.append(vol)
        write_number_at_center(imageRGB, l)



    volume_tot = sum(vol_all)
    length = sum(dxx)
    diam_med = np.mean(dyy)
    std_len = np.std(dxx)
    std_dia = np.std(dyy)
    std_vol = np.std(vol_all)

    print("length: ", length, "STDlen:", std_len, "diam med", diam_med, "STDdiam", std_dia)
    # print("diam: ", dyy)
    print("volume: ", volume_tot, " STD:", std_vol, "real", volume(320, 8))

    if SHOW_HIST_OF_DIAMETERS:
        # Draw the rectangle on the image (assuming you want to draw a blue rectangle)
        color = (0, 255, 0)  # Blue color in BGR format (OpenCV uses BGR)
        thickness = 1  # Thickness of the rectangle border
        cv2.rectangle(imageRGB, start_point, (start_point[0] + width, start_point[1] + height), color,
                                 thickness)
        plt.hist(dxx, bins=len(dxx), alpha=0.5, label='pointcloud')
        # plt.hist(dyy_expected, bins=len(dyy), alpha=0.5, label='expected from rgb')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram ')

        plt.show()

    save_to_csv_recursive(volume_tot, csv_file_path)
        #saved :


    return volume_tot

    # print("z,x,y", delta_z,delta_x,delta_y)


@timeit
def advanced_cylindricator_v2(mask_inverted_crop, pc_cropped, cropped_frame, black_out_depth_in_remasked_cropped):
    mask_inverted_crop = (255 - mask_inverted_crop)

    # for iteration i

    actual_collector = []
    actual_collector_pc = []
    actual_collector_RGB = []
    actual_collector_depth = []

    actual_collector.append(mask_inverted_crop)
    actual_collector_pc.append(pc_cropped)
    actual_collector_RGB.append(cropped_frame)
    actual_collector_depth.append(black_out_depth_in_remasked_cropped)
    for i in range(iteration):
        # print("LEN PC",len(actual_collector_pc))
        # create empty new collector
        new_collector = []
        new_collector_pc = []
        new_collector_RGB = []
        new_collector_depth = []
        # for every image of the collector of past iteration...

        for a in range(len(actual_collector)):
            # recognize contour
            cnt = recognize_contour(actual_collector[a])
            # create  box for contour
            rect = cv2.minAreaRect(cnt)

            # crop rect inclinated

            cropped_inclinated = crop_with_rect_rot(actual_collector[a], rect)
            imageR, imageL = image_splitter(cropped_inclinated)

            cropped_inclinated_pc = crop_with_rect_rot(actual_collector_pc[a], rect)
            imageR_pc, imageL_pc = image_splitter(cropped_inclinated_pc)

            cropped_inclinated_RGB = crop_with_rect_rot(actual_collector_RGB[a], rect)
            imageR_RGB, imageL_RGB = image_splitter(cropped_inclinated_RGB)

            cropped_inclinated_depth = crop_with_rect_rot(actual_collector_depth[a], rect)
            imageR_depth, imageL_depth = image_splitter(cropped_inclinated_depth)

            # split 2 image in max dimension

            # save the  two new  image in the collector
            new_collector.append(imageR)
            new_collector.append(imageL)

            new_collector_pc.append(imageR_pc)
            new_collector_pc.append(imageL_pc)

            new_collector_RGB.append(imageR_RGB)
            new_collector_RGB.append(imageL_RGB)

            new_collector_depth.append(imageR_depth)
            new_collector_depth.append(imageL_depth)

        # the new images becomes the old image for next iteration
        actual_collector = new_collector

        actual_collector_pc = new_collector_pc

        actual_collector_RGB = new_collector_RGB

        actual_collector_depth = new_collector_depth

    final_collector = actual_collector

    final_collector_pc = actual_collector_pc

    final_collector_RGB = actual_collector_RGB

    final_collector_depth = actual_collector_depth

    # i = 0
    # for imagei in final_collector:
    #     cv2.imshow("processing" + str(i), imagei)
    #     print(imagei.shape)
    #     i = i+1

    volume_tot_all_cyl = dimension_image_list_analyzer(final_collector_pc, final_collector_RGB)

    if SHOW_CYLINDRIFICATION  or volume_tot_all_cyl > 60000:
        final_collector = resize_image_imperfection_for_vconcaten(final_collector)
        final_collector_pc = resize_image_imperfection_for_vconcaten(final_collector_pc)
        final_collector_depth = resize_image_imperfection_for_vconcaten(final_collector_depth)

        cylinders = cv2.vconcat(final_collector)

        cylinders_pc = cv2.vconcat(final_collector_pc)

        cylinders_depth = cv2.vconcat(final_collector_depth)







        cv2.imshow("processing", resize_image(cylinders, 300))
        cv2.imshow("processing_pc", resize_image(cylinders_pc, 300))
        cv2.imshow("processing_depth", resize_image(cylinders_depth, 300))

    if SHOW_CYLINDRIFICATION_RGB or volume_tot_all_cyl > 60000:
        max_height = max(image.shape[0] for image in final_collector_RGB)
        # Resize all images to have the same height
        resized_images = []
        for image in final_collector_RGB:
            if image.shape[0] < max_height:
                pad_bottom = max_height - image.shape[0]
                resized_image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT,
                                                   value=(255, 255, 255))
            else:
                resized_image = image.copy()
            resized_images.append(resized_image)



        cylinders_RGB = cv2.hconcat(resized_images)

        cv2.imshow("processing_RGB", resize_image(cylinders_RGB, 300))


    return volume_tot_all_cyl


def image_splitter(frame):
    h = frame.shape[0]
    w = frame.shape[1]
    # channels = frame.shape[2]

    # decido se tagliare in altezza o larghezza
    if h > w:
        # top bottom
        new_h = h if h % 2 == 0 else h - 1

        frame = frame[0:new_h, 0:w]
        half2 = new_h // 2
        # print("h",h,"newh",new_h,"half",half2)

        img1 = frame[:half2, :]
        img2 = frame[half2:, :]
        # print(img1.shape(),img2.shape())

    else:
        # left right

        new_w = w if w % 2 == 0 else w - 1
        frame = frame[0:h, 0:new_w]
        half = new_w // 2
        # print("w", w, "neww", new_w,"half",half)

        img1 = frame[:, 0:half]
        img2 = frame[:, half:]
        # print(img1.shape, img2.shape)

    return img1, img2


def recognize_contour(mask):
    # if needed add contour for better detection
    # mask = cv2.copyMakeBorder(src=mask, top=BA4D, bottom=BA4D, left=BA4D, right=BA4D, borderType=cv2.BORDER_CONSTANT,
    #                           value=(255))
    # depth = cv2.copyMakeBorder(src=depth, top=BA4D, bottom=BA4D, left=BA4D, right=BA4D, borderType=cv2.BORDER_CONSTANT,
    #                            value=(255))
    # rgb = cv2.copyMakeBorder(src=rgb, top=BA4D, bottom=BA4D, left=BA4D, right=BA4D, borderType=cv2.BORDER_CONSTANT,
    #                          value=(255))
    #
    # pointcloud = cv2.copyMakeBorder(src=pointcloud, top=BA4D, bottom=BA4D, left=BA4D, right=BA4D,
    #                                 borderType=cv2.BORDER_CONSTANT, value=(255))

    mask = (255 - mask)
    # cv2.imshow("mcdo", imagem1)
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
        if area > 200:
            cnt = cnt_i

            return cnt
    return [0]


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

    # plt.hist(allx, bins=50)
    # plt.show()

    # Calculate the sum vector using the Pythagorean theorem
    sum_vector = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)

    # # Print the results
    print("LENGTH EXTERNAL:", diff_x)
    print("DIAMETER EXTERNAL:", diff_y)
    # print("Difference for Z:", diff_z)
    # print("Sum Vector:", sum_vector)
    # print("perc1z", z_01)
    # print("perc99z", z_99)
    # print("perc50z", z_50)


def detect_rotated_box(mask, frame):
    cnt1, completion = second_layer_accurate_cnt_estimator_and_draw(mask)
    rect = cv2.minAreaRect(cnt1)

    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    #
    # (tl, tr, br, bl) = box
    #
    # (tltrX, tltrY) = midpoint(tl, tr)
    # (blbrX, blbrY) = midpoint(bl, br)
    # # compute the midpoint between the top-left and top-right points,
    # # followed by the midpoint between the top-righ and bottom-right
    # (tlblX, tlblY) = midpoint(tl, bl)
    # (trbrX, trbrY) = midpoint(tr, br)
    #
    # cv2.line(frame, tl, tr, (0, 255, 0), 1)
    # cv2.line(frame, bl, br, (0, 255, 0), 1)
    # cv2.line(frame, tl, bl, (0, 255, 0), 1)
    # cv2.line(frame, tr, br, (0, 255, 0), 1)

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

            else:

                pointcloud[r, c] = [0, 0, 0]

                # z,x,y
                # pointcloud[r,c] = [0,0,0]

    return pointcloud, allz, allx, ally


def convert_u8_img_to_u16_d435_depth_image(u8_image):
    u8_image = u8_image + OFFSET_CM_COMPRESSION
    u16_image = u8_image.astype('uint16')
    u16_image_off = u16_image
    u16_image_off_mm = u16_image_off * 10
    return u16_image_off_mm


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


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
                                                    # cv2.drawContours(frame, [cnt1], 0, (200, 200, 50), 1)

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


delete_csv_file(csv_file_path)

for folders in os.listdir(PATH_HERE + PATH_2_AQUIS):
    print("files:", os.listdir(PATH_HERE + PATH_2_AQUIS))
    folder_name = folders
    # videos = os.listdir(PATH_HERE + PATH_2_AQUIS+ "/" + folder_name)
    # writeCSVdata(folder_name, ["frame", "pixel", "volume", "distance_med", "volumes", "distances"])

    for videos in os.listdir(PATH_HERE + PATH_2_AQUIS + "/" + folder_name):

        # print(videos)
        print("ITERATION:", folder_name)

        if videos.endswith(".mkv"):
            print(videos.split(".")[0])

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

        plt.close('all')
        ret, frame = video1.read()
        ret2, frame2 = video2.read()
        nrfr = nrfr + 1
        if nrfr > 5:

            if ret and ret2:

                # gestionn depthimg
                frame2of = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((5, 5), np.float32) / 25
                ###!!!! BLURRA SOLO I VALORI INTERNI ALLA MASCHERA:
                # prendi i valori dentro la maschera (media o simile) ed estendili a tutta l immagine

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(gray, THRES_VALUE, 255, cv2.THRESH_BINARY)
                rect_point_of_interest, cnt = detect_rotated_box(mask, frame)

                # make white out of the bounfing refct
                x, y, w, h = cv2.boundingRect(cnt)
                white_bg = 255 * np.ones_like(mask)
                roi = mask[y:y + h, x:x + w]
                white_bg[y:y + h, x:x + w] = roi
                mask = white_bg
                mask_inverted = cv2.bitwise_not(mask)

                # rimuovo ora i puntini neri con un closing
                kernel_noise = np.ones((3, 3), np.uint8)
                frame2of_no_noise = cv2.morphologyEx(frame2of, cv2.MORPH_OPEN, kernel_noise)
                # punti chiari sui neri spariscono

                # maschero il depth con la maschera rgb
                black_out_depth_in = cv2.bitwise_and(frame2of_no_noise, frame2of_no_noise, mask=mask_inverted)

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
                print("depth avarage", avg_masked_value)
                # riempi l esterno con il valore medio
                black_out_depth_in_AVARAGED = black_out_depth_in.copy()
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

                # # Convert the 2D grayscale image to a 1D array
                # one_dimension_array = black_out_depth_in_remasked.flatten()
                #
                # # Remove zero values from the array
                # one_dimension_array_without_zero = one_dimension_array[one_dimension_array != 0]
                #
                # # Display the resulting one-dimensional array without zero values
                # print(one_dimension_array_without_zero)
                # plt.hist(one_dimension_array_without_zero, bins=30)
                # plt.show()

                # extract only the pixels inside the ,ask:
                # qui stiamo cercando di visualizzare quanti pixel neri ci sono all interno della maschera che corrispondono al rumore TO DO
                pixels = frame2of[[mask_inverted]]

                # black_out_dapth_in = frame2of * mask_inverted

                # maschero anche la pointcloud prendendo in considerazione solo la mask
                pc, allz, allx, ally = convert_depth_image_to_pointcloud(black_out_depth_in_remasked, intrinsics, frame,
                                                                         x, y, w, h)
                simple_geometrical_analisys(allx, ally, allz)

                #cv2.imshow("depthhh", black_out_depth_in_remasked)
                #display3Dpointcloud(allx, ally , allz)
                #show_PCA(allx, ally , allz)
                # lo maschero srtingendo la mask per effetti di bordo
                #black_out_depth_in_remasked
                # kernel_pc = np.ones((3, 3), np.uint8)
                # mask_inverted_pc = cv2.erode(mask_inverted, kernel_pc, iterations=1)

                # pc = cv2.bitwise_and(pc, pc, mask=mask_inverted)

                pc_cropped = crop_with_rect_rot(pc, rect_point_of_interest)

                # visible_pointcloud = ((pc_cropped - pc_cropped.min()) * (1 / (pc_cropped.max() - pc_cropped.min()) * 255)).astype('uint8')

                # END _ ONLY DISPLAY

                # crop for imaging
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

                volume_tot_cylinders = advanced_cylindricator_v2(mask_inverted_crop, pc_cropped, cropped_frame,
                                                                 black_out_depth_in_remasked_cropped)

                # plt.hist(allz, bins=300)
                # plt.show()
                if SHOW_FILTERING_PROCESS:
                    processing = cv2.vconcat([frame2of_cropped, frame2of_no_noise_cropped, black_out_depth_in_cropped,
                                              black_out_depth_in_AVARAGED_cropped, black_out_depth_in_blurred_cropped,
                                              black_out_depth_in_remasked_cropped, mask_inverted_crop])
                    # print("frame2of_cropped, frame2of_no_noise_cropped,black_out_depth_in_cropped, black_out_depth_in_AVARAGED_cropped, black_out_depth_in_blurred_cropped,black_out_depth_in_remasked_cropped")

                    cv2.imshow("processing", resize_image(processing, 300))

                # cv2.imshow("or", resize_image(cropped_frame,300))
                # cv2.imshow("pc mask", resize_image(visible_pointcloud,300))

                key = cv2.waitKey(CONTINOUS_STREAM)
                if key == ord('q') or key == 27:
                    break


            else:
                break
        else:
            continue

    video1.release()
    video2.release()
    cv2.destroyAllWindows()

    print("COMPLETED ", nrfr, " frames")

plt.close('all')


