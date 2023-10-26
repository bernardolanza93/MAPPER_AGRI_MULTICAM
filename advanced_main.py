

from embedded_platform_basler import *
from embedded_platform_realsese import *









def processor():
    try:

        basler = False
        realsense = False

        organize_video_from_last_acquisition()

        global_status = multiprocessing.Value("i", 0)



        q_RS = multiprocessing.Queue(maxsize=100)
        q_BS = multiprocessing.Queue(maxsize=100)
        status_basler = multiprocessing.Value("i", 0)

        p0 = multiprocessing.Process(target=process_1_GPIO, args=(global_status,))
        p1 = multiprocessing.Process(target=BASLER_capture, args=(q_BS,status_basler,global_status))
        p2 = multiprocessing.Process(target=basler_saver, args=(q_BS,status_basler,global_status))
        p3 = multiprocessing.Process(target=RS_capture, args=(q_RS,global_status))
        p4 = multiprocessing.Process(target=RS_saver, args=(q_RS,global_status))



        p0.start()
        if basler:
            p1.start()
            p2.start()
        if realsense:
            p3.start()
            p4.start()



        p0.join()
        if basler:
            p1.join()
            p2.join()
            print("Basler cap? -> {}".format(p1.is_alive()))
            print("Basler save?    -> {}".format(p2.is_alive()))
        if realsense:
            p3.join()
            p4.join()
            print("Realsense cap? -> {}".format(p3.is_alive()))
            print("Realsense save?    -> {}".format(p4.is_alive()))


        # both processes finished
        print("Both processes finished execution!")

    except KeyboardInterrupt:
        print(' KILLED ..{} '.format(datetime.now()))
        print("STATUS PROCESSOR ZERO")
        time.sleep(1)
        status.value = 0
        sys.exit()






processor()































