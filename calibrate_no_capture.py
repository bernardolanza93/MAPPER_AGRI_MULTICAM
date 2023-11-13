import cv2
import time
import numpy as np
import os
import glob
from CONFIG_ODOMETRY_SYSTEM import *


def calibrate_v2():
    # Ensure the folder exists
    if not os.path.exists(IMAGE_CALIBRATION_PATH):
        print("The folder ", IMAGE_CALIBRATION_PATH , " does not exist.")
    else:
        calibration_images = []  # List to store loaded images

        for filename in os.listdir(IMAGE_CALIBRATION_PATH):
            if filename.endswith((".jpg", ".png", ".jpeg")):  # Adjust file extensions as needed
                image_path = os.path.join(IMAGE_CALIBRATION_PATH, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    calibration_images.append(image)
                    print("Loaded image: ", image_path, end = "")
                else:
                    print("Unable to load image: ", image_path)


        if not calibration_images:
            print("No images found in ", IMAGE_CALIBRATION_PATH)

    print("ALL IMAGES LOADED")
    print("CALIBRATION...")
    # Define the calibration board size (e.g., chessboard)
      # Change this to your board's dimensions

    # Arrays to store object points and image points
    obj_points = []  # 3D points in real-world space
    img_points = []  # 2D points in image plane

    # Create a grid of points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)
        print("|_| ", end = '')

    # Calibrate the camera
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None,
                                                                        None)


    if not os.path.exists(FOLDER_CALIBRATION_CAMERA):
        os.makedirs(FOLDER_CALIBRATION_CAMERA)

    # Save the data as .npy files in the specified folder
    np.save(os.path.join(FOLDER_CALIBRATION_CAMERA, "camera_matrix.npy"), K)
    np.save(os.path.join(FOLDER_CALIBRATION_CAMERA, "dist_coeffs.npy"), D)
    print("CALIBRATION COMPLETED")



def calibrate():
    CHECKERBOARD = board_size
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(IMAGE_CALIBRATION_PATH + "/"+'*.jpg')
    print(len(images))

    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # print(corners)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    print(N_OK)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs,
                                                       calibration_flags,
                                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    if not os.path.exists(FOLDER_CALIBRATION_CAMERA):
        os.makedirs(FOLDER_CALIBRATION_CAMERA)

    # Save the data as .npy files in the specified folder
    np.save(os.path.join(FOLDER_CALIBRATION_CAMERA, "camera_matrix.npy"), K)
    np.save(os.path.join(FOLDER_CALIBRATION_CAMERA, "dist_coeffs.npy"), D)


# You should replace these 3 lines with the output in calibration step



if __name__ == "__main__":
    #mi assicuri che questo file runni come principale

    # Specify the folder path


    # Check if the folder exists, and create it if not
    if not os.path.exists(IMAGE_CALIBRATION_PATH):
        os.makedirs(IMAGE_CALIBRATION_PATH)
        print("Folder  has been created:",IMAGE_CALIBRATION_PATH)

    # Check if the folder exists, and create it if not
    if not os.path.exists(FOLDER_CALIBRATION_CAMERA):
        os.makedirs(FOLDER_CALIBRATION_CAMERA)
        print("Folder  has been created:",FOLDER_CALIBRATION_CAMERA)



    # List all files in the folder
    file_list = os.listdir(IMAGE_CALIBRATION_PATH)
    print(file_list)

    # Count the number of JPG images in the folder
    jpg_count = sum(1 for file in file_list if file.lower().endswith(".jpg"))

    # Check if at least 20 JPG images are present
    if jpg_count >= 20:
        print(jpg_count, " JPG images found in the f older. IMAGE FOR CALIBRATION OK")
        if os.path.exists(os.path.join(FOLDER_CALIBRATION_CAMERA, "camera_matrix.npy")) and os.path.exists(
                os.path.join(FOLDER_CALIBRATION_CAMERA, "dist_coeffs.npy")):
            print("CALIBRATION ALREADY COMPLETED")
        else:
            print("CALIBRATION STARTING....")
            calibrate_v2()
    else:
        print("Error: Only", jpg_count ,"JPG images found in the folder. You need at least 20.")
        print("ERROR CAPTURING IMAGES.... NO IMAGE IN PATH :", IMAGE_CALIBRATION_PATH)










