import time

from embedded_platform_utils import *
from CONFIGURATION_VISION import *


def RS_saver(queue_RGB, queue_DEPTH, global_status):

    while 1:
        time.sleep(0.5)


        internal_saver_status = global_status.value

        size = (1920, 1080)
        fps = 20.0


        gst_out = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=RGB.mkv "
        out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER,  fps, size)

        #gst_out_depth   = "appsrc ! video/x-raw, format=GRAY ! queue ! videoconvert ! video/x-raw,format=GRAY ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=DEPTH.mkv "
        gst_out_depth = "appsrc caps=video/x-raw,format=GRAY8 ! videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! h265parse ! filesink location=DEPTH.mkv "
        #gst_out_depth = ("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=test.mkv" )
        #gst_out_depth = ('appsrc caps=video/x-raw,format=GRAY8,width=1920,height=1080,framerate=30/1 ! '' videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! ''h265parse ! filesink location=test.h265 ')
        out_depth = cv2.VideoWriter(gst_out_depth, cv2.CAP_GSTREAMER,  fps, size, 0)





        print("WAIT RS SAVER LOOP ")


        while internal_saver_status == 0:
            internal_saver_status = global_status.value
            time.sleep(0.5)

        print("|_|_| SAVER READY!,  local_status:", internal_saver_status)

        while internal_saver_status == 1 or queue_RGB.qsize() > 0 or queue_DEPTH.qsize() > 0:
            # start_time_sa = time.time()
            qsize_RGB = queue_RGB.qsize()
            #print("Q long: ", qsize_RGB)
            rgb = queue_RGB.get()
            depth = queue_DEPTH.get()
            try:
                out.write(rgb)
                out_depth.write(depth)

            except Exception as e:
                print("ERROR SAVE RS %s", str(e))

            #
            #     # save here depth map


            # end_time_sa = time.time()
            #
            # # Calculate the total time taken
            # total_time_sa = end_time_sa - start_time_sa
            #
            # # Calculate FPS
            # fps_sa = 1 / total_time_sa
            # print(f"FPS_SAV: {fps_sa}")

        print("REALSENSE SAVER RELEASED")
        # out_BASLER.release()
        out.release()
        out_depth.release()

        print("___SAVER_RS___ENDED RECORDING_____")



def RS_capture(queue_RGB,queue_DEPTH,global_status):

    #configure
    #wait loop
    #acquire D435

    # CONFIG
    # VIDEO LOOP
    # WAIT PLAY LOOP
    # RECORD LOOP

    check_folder("/data/")
    ##config.enable_device('947122110515')


    ctx = rs.context()
    enable_D435i, enable_T265, device_aviable = search_device(ctx)

    print("DEVICE: | D435:", enable_D435i, " | T265:", enable_T265, " |")

    # D435____________________________________________
    print("START CONFIG D435")
    if enable_D435i:

        pipeline = rs.pipeline(ctx)
        config = rs.config()
        try:
            seriald435 = str(device_aviable['D435I'][0])
        except:
            print("no d435i try classic model d435")
            seriald435 = str(device_aviable['D435'][0])

        print("serial : ", type(seriald435))
        config.enable_device(seriald435)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        align_to = rs.stream.color
        align = rs.align(align_to)
    else:
        if enable_T265:
            print("______|_|_|______ONLY ODOMETRY SYSTEM_______|_|_|______")


    print("START CONFIG T265")
    if enable_T265:
        # T265_________________________________________________
        pipelineT265 = rs.pipeline(ctx)
        configT265 = rs.config()
        serialt265 = str(device_aviable['T265'][0])
        print(serialt265)
        configT265.enable_device(serialt265)
        configT265.enable_stream(rs.stream.pose)
        configT265.enable_stream(rs.stream.gyro)
        print("T265 CONFIGURED!")

    else:
        print("no T265 MODE")

    while 1:
        local_status = global_status.value




        print("T265/D435 inizialized, DEVICE READY!")
        print("STATUS RS LOOP, checking buttons status:", local_status)
        print("=====================================================")
        print("||                                                 ||")
        print("||            D435/T265 CLICK PLAY!                ||")
        print("||                                                 ||")
        print("=====================================================")

        time.sleep(0.3)



        while local_status == 0:
            local_status = global_status.value
            time.sleep(0.5)
            print(".", end="")
        print(".<-")
        print("|_> STARTING!, STATUS LOOP EXIT,  local_status:", local_status)


        now_file_ar = datetime.now()
        timing_abs_ar = now_file_ar.strftime("%Y_%m_%d_%H_%M_%S")


        frame_c = 0
        local_status = global_status.value

        writeCSVdata_odometry(timing_abs_ar, ["frame", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw"])

        print("LOCAL REALSENSE STATUS INIZIALIZED = ", local_status)
        time.sleep(0.2)

        if enable_T265:
            print("INITIALIZED MAPPER LOCALIZER FILE")

            try:
                # Start streaming
                started = pipelineT265.start(configT265)
                print("T265 started OK", started)
            except Exception as e:
                print("ERROR PIPELINE T265: %s", str(e))

        if enable_D435i:
            try:
                # Start streaming
                pipeline.start(config)
                # colorizer = rs.colorizer()
                print("PIPELINE D435 STARTED")
            except Exception as e:
                print("ERROR PIPELINE D435 %s", str(e))



        #____________MAIN IMAGE LOOP"_____________
        while local_status == 1:

            local_status = global_status.value

            if enable_T265 or enable_D435i:

                frame_c += 1


                if enable_T265:
                    try:
                        tframes = pipelineT265.wait_for_frames()
                        print("PIPELINE T265 STARTED")

                    except Exception as e:
                        print("ERROR PIPELINE T265 %s", e)
                        pose = 0

                    pose = tframes.get_pose_frame()
                    data = pose.get_pose_data()
                    w = data.rotation.w
                    x = -data.rotation.z
                    y = data.rotation.x
                    z = -data.rotation.y

                    roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z) * 180.0 / m.pi;
                    pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi;
                    yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) * 180.0 / m.pi;
                    pose_list = [data.translation.x, data.translation.y, data.translation.z, data.velocity.x,
                                 data.velocity.y, data.velocity.z, roll, pitch, yaw]
                    pose_list.insert(0, frame_c)

                    # print("Frame #{}".format(pose.frame_number))
                    # print("Position: {}".format(data.translation))
                    # print("Velocity: {}".format(data.velocity))
                    # print("Acceleration: {}\n".format(data.acceleration))
                    writeCSVdata_odometry(timing_abs_ar, pose_list)


                if enable_D435i:
                    # Wait for a coherent pair of frames: depth and color
                    try:
                        frames = pipeline.wait_for_frames()
                    except Exception as e:
                        print("PIPELINE error:||||:: %s", str(e))


                    aligned_frames = align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    if frame_c < 3:
                        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                        calculate_and_save_intrinsics(depth_intrin)
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    #print("DEPTH before", depth_image.shape)
                    # convert u16 mm bw image to u16 cm bw
                    resized = depth_image / 10
                    # rescale without first 50 cm of offset unwanted
                    resized = resized - offset
                    # stretchin all in the 0-255 cm interval
                    maxi = np.clip(resized, 0, 255)
                    # convert to 8 bit
                    intcm = maxi.astype('uint8')

                    try:
                        queue_RGB.put(color_image)
                        queue_DEPTH.put(intcm)

                    except:
                        print("QUEUE ERROR REALSENSE")
                    #Dormiamo ora per non pesare sull acquisizione basler
                    time.sleep(TIME_WAITER_REALSENSE_FREEZER)


                    if DISPLAY_RGB:
                        # cv2.imshow('depth Stream', color_image)
                        cv2.imshow('dept!!!h Stream', intcm)

                        key = cv2.waitKey(1)
                        if key == 27:
                            # result.release()
                            # cv2.destroyAllWindows()
                            break
                else:
                    # converte la velocita di salvataggio dai 1500 FPS (T265 standalone)  ad un acquisizione piu realistica (15 FPS della D435)
                    time.sleep(0.067)


            else:
                print("REALSENSE NONE !!!")
                time.sleep(5)
            #END CYCLE



        if enable_D435i:
            pipeline.stop()
            cv2.destroyAllWindows()
            print("closing object...")
            time.sleep(0.3)
        if enable_T265:
            pipelineT265.stop()

        print("TERMINATING RS CAPTURE")
        time.sleep(0.5)




def search_device(ctx):
    enable_D435i = False
    enable_T265 = False
    device_aviable = {}

    print("ctx.devices  = ",ctx.devices)
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print(d)
            device = d.get_info(rs.camera_info.name)
            serial = d.get_info(rs.camera_info.serial_number)
            model = str(device.split(' ')[-1])

            device_aviable[model] = [serial,device]


            print('Found device: ', device_aviable)

    else:
        print("No Intel Device connected")
    keys = list(device_aviable.keys())
    for i in range(len(keys)):
        if keys[i] == "D435I" or keys[i] == "D435":
            print("found: ", keys[i])
            enable_D435i = True
        elif keys[i] == "T265":
            enable_T265 = True
        else:
            print("camera not recognized")

    print("D435i ok ? => ", enable_D435i, "___|||____  T265 ok ? => ", enable_T265)

    return enable_D435i, enable_T265, device_aviable



def calculate_and_save_intrinsics(intrinsics):

    #print("intrinsics create...", intrinsics, type(intrinsics))

    title = "intrinsics.csv"

    if not os.path.exists(title):
        int = [intrinsics.width, intrinsics.height, intrinsics.ppx, intrinsics.ppy, intrinsics.fx, intrinsics.fy, intrinsics.model, intrinsics.coeffs]
        writeCSVdata_generic(title, int)
        print("new file intrinsics written")
        print(int)
