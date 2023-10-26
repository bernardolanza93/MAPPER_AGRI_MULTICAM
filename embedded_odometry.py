
from embedded_platform_utils import *
from embedded_platform_realsese import *
DIVIDER_FPS_REDUCTION = 0.067



def odometry_capture(global_status):
    while 1:
        time.sleep(1)

        check_folder("/data/")
        timing = now.strftime("%Y_%m_%d_%H_%M_%S")



        print("FOLDER ORGANIZED COMPLETED!")
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
        while global_status.value == 0:
            time.sleep(0.5)
            print(".", end="")
        print("started!")

        while global_status.value == 1:
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
                        writeCSVdata(timing, [frame_c, time_st, data.translation, data.velocity, anglePRY])
                        if not enable_D435i:
                            #converte la velocita di salvataggio dai 1500 FPS (T265 standalone)  ad un acquisizione piu realistica (15 FPS della D435)
                            time.sleep(DIVIDER_FPS_REDUCTION)




        if enable_T265:
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















