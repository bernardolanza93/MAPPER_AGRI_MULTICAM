
from embedded_platform_utils import *

def RS_saver(queue,status):
    print("TODO")



def RS_capture(queue,status):

    check_folder("/data/")




    print("FOLDER ORGANIZED COMPLETED!")
    ##config.enable_device('947122110515')

    ctx = rs.context()
    enable_D435i, enable_T265, device_aviable = search_device(ctx)

    print("D435:", enable_D435i, " | T265:", enable_T265)

    # D435____________________________________________
    if enable_D435i:
        """
        # Declare pointcloud object, for calculating pointclouds and texture mappings
        pc = rs.pointcloud()
        # We want the points object to be persistent so we can display the last cloud when a frame drops
        points = rs.points()
        """
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

        # We'll use the colorizer to generate texture for our PLY

        # config.enable_stream(rs.stream.accel,rs.format.motion_xyz32f,200)
        # config.enable_stream(rs.stream.gyro,rs.format.motion_xyz32f,200)
        saver = rs.save_single_frameset()
        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            # Start streaming
            pipeline.start(config)
            # colorizer = rs.colorizer()
            print("D435I START OK ___________")
        except Exception as e:
            print("ERROR ON START D435:||||:: %s", str(e))
        # _________________________________________________

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

    if SAVE_VIDEO_TIME != 0:
        if enable_D435i:
            gst_out = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=RGB.mkv "
            out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER,  20.0, (1900, 1080))

            #gst_out_depth   = "appsrc ! video/x-raw, format=GRAY ! queue ! videoconvert ! video/x-raw,format=GRAY ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=DEPTH.mkv "
            gst_out_depth = "appsrc caps=video/x-raw,format=GRAY8 ! videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! h265parse ! filesink location=DEPTH.mkv "
            #gst_out_depth = ("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=test.mkv" )
            #gst_out_depth = ('appsrc caps=video/x-raw,format=GRAY8,width=1920,height=1080,framerate=30/1 ! '' videoconvert ! omxh265enc ! video/x-h265, stream-format=byte-stream ! ''h265parse ! filesink location=test.h265 ')
            out_depth = cv2.VideoWriter(gst_out_depth, cv2.CAP_GSTREAMER,  20.0, (1920, 1080), 0)




    else:
        print("NO SAVE VIDEO D435 MODE")
    frame_c = 0

    print("START LOOP")

    for i in range(FRAMES_TO_ACQUIRE):
        if enable_T265 or enable_D435i:
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
                    data = pose.get_pose_data()
                    w = data.rotation.w
                    x = -data.rotation.z
                    y = data.rotation.x
                    z = -data.rotation.y

                    pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi;
                    roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z) * 180.0 / m.pi;
                    yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) * 180.0 / m.pi;
                    anglePRY = [pitch, roll, yaw]

                    # print("Frame #{}".format(pose.frame_number))
                    # print("Position: {}".format(data.translation))
                    # print("Velocity: {}".format(data.velocity))
                    # print("Acceleration: {}\n".format(data.acceleration))
                    now = datetime.now()
                    time_st = now.strftime("%d-%m-%Y|%H:%M:%S")
                    writeCSVdata(date_time, [frame_c, time_st, data.translation, data.velocity, anglePRY])
                    if not enable_D435i:
                        #converte la velocita di salvataggio dai 1500 FPS (T265 standalone)  ad un acquisizione piu realistica (15 FPS della D435)
                        time.sleep(0.067)

            if enable_D435i:
                # Wait for a coherent pair of frames: depth and color

                try:
                    frames = pipeline.wait_for_frames()


                except Exception as e:
                    print("PIPELINE error:||||:: %s", str(e))
                    sys.exit()

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if frame_c < 3:
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                    calculate_and_save_intrinsics(depth_intrin)
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                print("DEPTH before", depth_image.shape)



                # width = int(1920)
                # height = int(1080)
                # dim = (width, height)
                #
                # # resize image depth to fit rgb
                # print("DEPTH before", depth_image.shape)
                # resized = cv2.resize(depth_image, dim, interpolation=cv2.INTER_AREA)
                # print("DEPTH after", resized.shape)
                # convert u16 mm bw image to u16 cm bw
                resized = depth_image / 10
                # rescale without first 50 cm of offset unwanted
                resized = resized - offset

                # stretchin all in the 0-255 cm interval
                maxi = np.clip(resized, 0, 255)
                # convert to 8 bit
                intcm = maxi.astype('uint8')

                # print("RGB", color_image.shape)
                print("DEPTH", intcm.shape)

                if SAVE_VIDEO_TIME != 0:
                    try:
                        out.write(color_image)


                    except Exception as e:
                        print("error save video:||||:: %s", str(e))

                    try:
                        # save here depth map
                        out_depth.write(intcm)

                    except Exception as e:
                        print("error saving depth 1 ch:||||:: %s", str(e))

            if FPS_DISPLAY:
                end = time.time()
                seconds = end - start
                fps = round(1 / seconds, 3)
                print(fps)

            if DISPLAY_RGB:
                # cv2.imshow('depth Stream', color_image)
                cv2.imshow('dept!!!h Stream', intcm)

                key = cv2.waitKey(1)
                if key == 27:
                    # result.release()
                    # cv2.destroyAllWindows()
                    break
        else:
            print("no realsense error!!!!")
            time.sleep(5)

        if enable_T265 == False and enable_D435i == False :
            print("no device, termination...")
            sys.exit()

    if enable_D435i:
        pipeline.stop()
        try:
            out.release()
        except Exception as e:
            print("ERROR RELEASE RGB: %s", str(e))
        try:
            out_depth.release()
        except Exception as e:
            print(out_depth)
            print(intcm.shape)
            print("ERROR RELEASE DEPTH: %s", str(e))

        cv2.destroyAllWindows()
        print("closing object...")
        time.sleep(2)
    if enable_T265:
        pipelineT265.stop()
    print("TERMINATING RS CAPTURE")
    status.value = 0
    time.sleep(2)




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
