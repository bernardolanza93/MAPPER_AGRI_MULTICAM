
from embedded_platform_utils import *
from embedded_platform_realsese import *
import aruco_library as ARUCO
print(" CV2  version: ",cv2.__version__)

local_status = 0


def search_aruco_in_frames(image):




    pose = ARUCO.aruco_detection(image)

    return pose


def odometry_capture(global_status):
    print("one shot INIZIALIZATION T265")
    ctx = rs.context()
    enable_D435i, enable_T265, device_aviable = search_device(ctx)

    print(" | T265:", enable_T265)

    if enable_T265:
        # T265_________________________________________________

        pipelineT265 = rs.pipeline(ctx)
        configT265 = rs.config()
        serialt265 = str(device_aviable['T265'][0])
        print(serialt265)
        configT265.enable_device(serialt265)
        configT265.enable_stream(rs.stream.pose)
        configT265.enable_stream(rs.stream.gyro)
        print("IMU CONFIG")
        #FISHEY!
        configT265.enable_stream(rs.stream.fisheye, 1)
        configT265.enable_stream(rs.stream.fisheye, 2)
        print("FISHEYE CONFIG")
        print("configured succesfully T265...")
    while 1:
        local_status = global_status.value
        print("LOCAL STAT INI = ",local_status)
        time.sleep(1)

        check_folder("/data/")
        now_file_ar = datetime.now()
        timing_abs_ar = now_file_ar.strftime("%Y_%m_%d_%H_%M_%S")
        writeCSVdata_odometry("_ARUCO_" + timing_abs_ar, ["frame","id_marker","x","y","z","roll", "pitch", "yaw"])
        writeCSVdata_odometry(timing_abs_ar,[ "frame","x","y","z","vx","vy","vz","roll", "pitch", "yaw"])



        print("PIPELINE CONFIG T265...")



        if enable_T265:

            try:
                # Start streaming
                started = pipelineT265.start(configT265)
                print("T265 started OK",started)
            except Exception as e:
                print("error pipeline T265 starting:||||:: %s", str(e))
            # _______________________________________________________
        else:
            print("no T265 MODE")

        frame_c = 0

        print("T265 inizialized, DEVICE READY!")
        print("STATUS LOOP, checking buttons status:", local_status)
        print("T265 inizialized, DEVICE READY!")
        print("STATUS LOOP, checking buttons status:", local_status)
        print("=====================================================")
        print("||                                                 ||")
        print("||                   CLICK PLAY!                   ||")
        print("||                                                 ||")
        print("=====================================================")


        while local_status == 0:
            local_status = global_status.value
            time.sleep(0.5)
            print(".", end="")
        print(".<-")
        print("|_> STARTING!, STATUS LOOP EXIT,  local_status:", local_status)
        while local_status == 1:

            if  PRINT_FPS_ODOMETRY:
                start_time = time.time()


            if enable_T265 or enable_D435i:


                frame_c += 1


                if enable_T265:
                    try:
                        tframes = pipelineT265.wait_for_frames()
                    except Exception as e:
                        print("ERROR T265 wait4fr: %s", e, "object ideally not present",started)
                        #started = pipelineT265.start(configT265)
                        #tframes = pipelineT265.wait_for_frames()
                        pose = 0
                    try:

                        pose = tframes.get_pose_frame()


                    except Exception as e:
                        print("ERROR T265 getFr: %s", e)
                        pose = 0
                    if pose:

                        if DETECT_MARKER:


                            # try:


                            f1 = tframes.get_fisheye_frame(1)
                            if not f1:
                                print("ERROR NO FISHEYE FRAME")
                                continue

                            image1 = np.asanyarray(f1.get_data())

                            pose_aruco = search_aruco_in_frames(image1)

                            if pose_aruco == 0:
                                pose_aruco = [frame_c,0]
                            else:

                                pose_aruco.insert(0, frame_c)


                            # except Exception as e:
                            #     print("DETECT ARUCO ERROR",e)
                            #     pose_aruco = [0]

                            writeCSVdata_odometry("_ARUCO_" + timing_abs_ar, pose_aruco)

                        data = pose.get_pose_data()

                        w = data.rotation.w
                        x = -data.rotation.z
                        y = data.rotation.x
                        z = -data.rotation.y

                        roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z) * 180.0 / m.pi;
                        pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi;
                        yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) * 180.0 / m.pi;
                        pose_list = [data.translation.x, data.translation.y, data.translation.z, data.velocity.x, data.velocity.y,data.velocity.z, roll, pitch, yaw]
                        pose_list.insert(0, frame_c)

                        # print("Frame #{}".format(pose.frame_number))
                        # print("Position: {}".format(data.translation))
                        # print("Velocity: {}".format(data.velocity))
                        # print("Acceleration: {}\n".format(data.acceleration))
                        writeCSVdata_odometry(timing_abs_ar, pose_list)
                        if not DETECT_MARKER:
                            #converte la velocita di salvataggio dai 1500 FPS (T265 standalone)  ad un acquisizione piu realistica (15 FPS della D435)
                            time.sleep(DIVIDER_FPS_REDUCTION)
                    local_status = global_status.value
                    if local_status == 0:
                        print("TERMINATION SIGNAL DETECTED")
            if PRINT_FPS_ODOMETRY:
                # End time
                end_time = time.time()

                # Calculate time taken
                time_taken = end_time - start_time

                # Calculate FPS
                fps = int(1 / time_taken)
                print("FPS:", fps)






        if enable_T265:
            print("PIPELINE STOPPED!")
            pipelineT265.stop()



def processor():
    try:

        global_status = multiprocessing.Value("i", 0)

        p0 = multiprocessing.Process(target=process_1_GPIO, args=(global_status,))
        p1 = multiprocessing.Process(target=odometry_capture, args=(global_status,))

        p0.start()
        p1.start()

        p0.join()
        p1.join()

        print("pinout cap? -> {}".format(p0.is_alive()))
        print("odometry save?    -> {}".format(p1.is_alive()))

        # both processes finished
        print("Both processes finished execution!")

    except KeyboardInterrupt:
        print(' KILLED ..{} '.format(datetime.now()))
        print("STATUS PROCESSOR ZERO")
        time.sleep(1)
        global_status.value = 0



        sys.exit()


processor()

















