from embedded_platform_utils import *
import cv2.aruco as aruco



# ArUco marker size in meters (if known)
marker_size = MARKER_SIZE  # Adjust this to the actual marker size
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
# Load the camera matrix and distortion coefficients
# Check if the files exist before loading
if os.path.exists(os.path.join(FOLDER_CALIBRATION_CAMERA, "camera_matrix.npy")) and os.path.exists(
        os.path.join(FOLDER_CALIBRATION_CAMERA, "dist_coeffs.npy")):
    # Load the camera matrix and distortion coefficients
    camera_matrix = np.load(os.path.join(FOLDER_CALIBRATION_CAMERA, "camera_matrix.npy"))
    dist_coeffs = np.load(os.path.join(FOLDER_CALIBRATION_CAMERA, "dist_coeffs.npy"))
    """
           K =
           | fx   0    cx |
           | 0    fy   cy |
           | 0    0    1  |

           dist_coeffs = [k1, k2, p1, p2, k3]

       """

    parameters = aruco.DetectorParameters_create()
    print("ARUCO DETECTOR OBJECT CREATED! ")
else:
    print("ARUCO MODULE FAIL // NO CALIBRATION")
    sys.exit()

def aruco_detection(image1,frame_id,timing_abs_ar):





    """When you create an cv::aruco::ArucoDetector object, you need to pass the following parameters to the constructor:

    A dictionary object, in this case one of the predefined dictionaries (cv::aruco::DICT_6X6_250).
    Object of type cv::aruco::DetectorPartiming_abs_arameters. This object includes all parameters that can be customized during
    the detection process. These parameters will be explained in the next section.

    The parameters of detectMarkers are:

    The first parameter is the image containing the markers to be detected.
    The detected markers are stored in the markerCorners and markerIds structures:
        markerCorners is the list of corners of the detected markers. For each marker, its four corners are returned
        in their original order (which is clockwise starting with top left). So, the first corner is the top left corner,
         followed by the top right, bottom right and bottom left.
        markerIds is the list of ids of each of the detected markers in markerCorners. Note that the returned
        markerCorners and markerIds vectors have the same size.
    The final parameter, rejectedCandidates, is a returned list of marker candidates, i.e. shapes that were found and
    considered but did not contain a valid marker. Each candidate is also defined by its four corners, and its format
    is the same as the markerCorners parameter. This parameter can be omitted and is only useful for debugging purposes
    and for ‘refind’ strategies (see refineDetectedMarkers() ).
    """

    corners, ids, rejectedImgPoints = aruco.detectMarkers(image1, aruco_dict, parameters=parameters)





    """cameraMatrix and distCoeffs are the camera calibration parameters that were created during the camera calibration process.
    The output parameters rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in markerCorners.
    """

    if ids is not None:
        # Draw markers and estimate pose
        rvecs_all, tvecs_all = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        print(len(rvecs_all),len(tvecs_all),len(ids))
        for i in range(len(ids)):
            marker_id = ids[i]
            rvecs = rvecs_all[i]
            tvecs = tvecs_all[i]

            print(marker_id,tvecs,rvecs)

            #cv2.aruco.drawAxis(image1, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
            #cv2.aruco.drawDetectedMarkers(image1, corners)

            # Extract the center of the ArUco marker
            # center_x = int(np.mean(corners[i][0][:, 0]))
            # center_y = int(np.mean(corners[i][0][:, 1]))

            # rvecs and tvecs represent the rotation and translation vectors of the marker with respect to the camera.

            """
            The rotation vector (rvec) is a 3x1 vector representing the 3D rotation in axis-angle representation.
            The translation vector (tvec) is a 3x1 vector representing the 3D translation.
    
            [P_camera] = [R|t] * [P_world]
    
    
            [P_camera] is the 3D point in the camera coordinate system.
            [R|t] is the concatenation of the rotation matrix R and the translation vector t (extrinsic parameters).
            [P_world] is the 3D point in the world coordinate system.
    
            """

            # Convert the rotation vector to a rotation matrix
            try:
                R, _ = cv2.Rodrigues(rvecs)
            except Exception as e:
                print("|_|_|_| rvecs err or R", rvecs,ids, e)

            """
    
            R = Rz(θz) * Ry(θy) * Rx(θx)
    
            Rx(θx) = 
                 | 1       0           0   |
                 | 0    cos(θx)   -sin(θx) |
                 | 0    sin(θx)    cos(θx) |
    
            Ry(θy) = | cos(θy)    0    sin(θy) |
                     | 0           1       0   |
                     | -sin(θy)   0    cos(θy) |
    
            Rz(θz) = 
                 | cos(θz)   -sin(θz)    0 |
                 | sin(θz)    cos(θz)    0 |
                 | 0          0          1 |
    
    
            you can use the cv2.Rodrigues function in OpenCV to convert a rotation vector (rvec) into a rotation matrix (R). Here's an example in Python: 
    
            """

            # Convert the rotation matrix to a rotation vector

            rvec, _ = cv2.Rodrigues(R)


            # Calculate Euler angles from the rotation vector
            roll, pitch, yaw = rvec.flatten()

            # Convert radians to degrees
            roll *= 180.0 / m.pi
            pitch *= 180.0 / m.pi
            yaw *= 180.0 / m.pi

            # Extract x, y, and z from tvec
            if tvecs is not None:
                for i in range(len(tvecs)):
                    # Ensure the translation vector contains three elements
                    if len(tvecs[i][0]) == 3:
                        x, y, z = tvecs[i][0]
                    else:
                        print(f"Translation vector for marker {i} does not contain three values.")
            else:
                print("No translation vectors found.")

            pose = [frame_id,marker_id[0], x, y, z, roll, pitch, yaw]


            """
    
            If you want the pose of the camera with respect to the marker (i.e., the camera's position and orientation relative
             to the marker), you can easily obtain it by taking the inverse of the pose provided by 
             cv2.aruco.estimatePoseSingleMarkers. Here's how you can modify the code to get the camera pose relative 
             to the marker:
    
            """

            INVERSE = 0
            if INVERSE:

                inv_rotation_matrix = np.linalg.inv(R)

                # Invert the translation vector
                inv_tvec = -np.dot(inv_rotation_matrix, tvecs[0])

                # Extract x, y, and z from inv_tvec
                x, y, z = inv_tvec

                # Convert the rotation matrix to Euler angles (roll, pitch, yaw)
                inv_roll, inv_pitch, inv_yaw = cv2.RQDecomp3x3(inv_rotation_matrix)

            writeCSVdata_odometry("_ARUCO_" + timing_abs_ar, pose)

    else:

        pose = [frame_id, 0]




#scrivi qui il marker che in caso ne scrive 2
    return pose



