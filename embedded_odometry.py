
from embedded_platform_utils import *
from embedded_platform_realsese import *
import aruco_library as ARUCO
print(" CV2  version: ",cv2.__version__)

local_status = 0

def search_aruco_in_frames(frames):
    f1 = frames.get_fisheye_frame(1)
    f2 = frames.get_fisheye_frame(2)


    image1 = np.asanyarray(f1.get_data())

    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    pose = ARUCO.aruco_detection(gray)


    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', image1)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        return
    if key == ord('s'):
        cv2.imwrite("a.jpg", image1)
    print("search aruco")
    return pose


def odometry_capture(global_status):
    while 1:
        local_status = global_status.value
        print("LOCAL STAT INI = ",local_status)
        time.sleep(1)

        check_folder("/data/")
        now = datetime.now()
        timing = now.strftime("%Y_%m_%d_%H_%M_%S")
        writeCSVdata_odometry("_ARUCO_" + timing, ["frame","id_marker","x","y","z","roll", "pitch", "yaw"])
        writeCSVdata_odometry(timing,[ "frame","x","y","z","vx","vy","vz","roll", "pitch", "yaw"])



        ##config.enable_device('947122110515')
        print("CONFIGURING T265...")

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
            print("configured succesfully T265...")

            # saver.set_option()

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
        print("LOCAL ODOMOETRY STATUS LOOP, checking buttons...status:", local_status)

        while local_status == 0:
            local_status = global_status.value
            time.sleep(0.5)
            print(".", end="")
        print("|_> LOCAL ODOMOETRY STATUS LOOP EXIT, started!, local_status", local_status)
        while local_status == 1:

            if enable_T265 or enable_D435i:

                now = datetime.now()
                time_st = now.strftime("%d-%m-%Y|%H:%M:%S")

                start = time.time()
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


                            try:
                                f1 = tframes.get_fisheye_frame(1)
                                if not f1:
                                    print("FISHEYE CAMERA 1 ERROR")
                                image1 = np.asanyarray(f1.get_data())
                                pose_aruco = search_aruco_in_frames(image1)
                                pose_aruco.insert(0, frame_c)


                            except Exception as e:
                                print("ARUCO ERROR",e)
                                pose_aruco = [0]


                            writeCSVdata_odometry("_ARUCO_" + timing, pose_aruco)

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

                        writeCSVdata_odometry(timing, pose_list)
                        if not enable_D435i:
                            #converte la velocita di salvataggio dai 1500 FPS (T265 standalone)  ad un acquisizione piu realistica (15 FPS della D435)
                            time.sleep(DIVIDER_FPS_REDUCTION)
                    local_status = global_status.value
                    if local_status == 0:
                        print("TERMINATION SIGNAL DETECTED")





        if enable_T265:
            print("PIPELINE STOPPED!")
            pipelineT265.stop()




def processor():
    try:


        global_status = multiprocessing.Value("i", 0)

        p0 = multiprocessing.Process(target=process_1_GPIO, args=(global_status,))
        p1 = multiprocessing.Process(target=odometry_capture(), args=(global_status,))

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















