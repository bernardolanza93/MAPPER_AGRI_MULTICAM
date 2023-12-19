import time

from embedded_platform_utils import *
from pypylon import pylon


def BASLER_capture(q,status,global_status):
    internal_global_status = global_status.value

    """
    functioon to acquire images from basler dart camera with pylon

    :param q: queue where to put the images extracted
    :param status: status to control the events of the process
    :return: nothing
    """

    # conecting to the first available camera
    try:

        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # lo usa la cri vediamo a che serve
        camera.Open()

        print('Using device: ', camera.GetDeviceInfo().GetModelName())
        try:
            pylon.FeaturePersistence.Load(config_file, camera.GetNodeMap(), True)
            # pylon.FeaturePersistence.Save(config_file, camera.GetNodeMap())
        except Exception as e:

            print("basler failed load config", e)
            print("basler failed", config_file)
            status.value = 0

        #

        # Grabing Continusely (video) with minimal delay
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        converter = pylon.ImageFormatConverter()

        # converting to opencv bgr format
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Set video resolution
        frame_width = 2592
        frame_height = 1944
        size = (frame_width, frame_height)

        # result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, size)

        print("BASLER CONFIGURED")
    except Exception as e:
        basler_presence = False
        status.value = 0
        print("basler configuration failed", e)



    #stato di continuo try di acquisizione
    while 1:

        print("WAIT LOOP LOOP")

        while internal_global_status == 0:
            internal_global_status = global_status.value
            time.sleep(0.5)
            print(".")
        print(".<-")
        print("|_> STARTING! _PLAY_, STATUS LOOP EXIT,  local_status:", internal_global_status)
        frame_c = 0
        while internal_global_status == 1:
            internal_global_status = global_status.value
            frame_c += 1
            try:

                if camera.IsGrabbing():

                    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                    if grabResult.GrabSucceeded():
                        # Access the image data
                        image = converter.Convert(grabResult)
                        img_basler = image.GetArray()
                        print(frame_c)



                        if SAVE_VIDEO_TIME != 0:
                            try:
                                q.put(img_basler)

                            except:
                                print("error save basler")

                        key = cv2.waitKey(1)
                        if key == 27:
                            break

                    else:
                        print("ERROR: camera not succeded, no image")
                        status.value = 0
                else:
                    print("ERROR: camera is not grabbing")
                    status.value = 0

            except Exception as e:
                print("ERROR basler in loop wait4fr: %s", e)
                basler_presence = False
                status.value = 0

        print("CYCLE TERMINATED-READY NEW AQUISITION")

def basler_saver(q,basler_status,global_status):
    """

    :param q: multiprocessing queue, here there are all the image , one by one, aquired by the basler dart pylon process
    :return: nothing

    """
    internal_saver_status = global_status.value

    while 1:
        time.sleep(0.5)
        #organize_video_from_last_acquisition()



        internal_saver_status = global_status.value







        frame_width = 2592
        frame_height = 1944

        gst_out_BASLER = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc ! h264parse ! matroskamux ! filesink location=RGB_BAS.mkv "
        out_BASLER = cv2.VideoWriter(gst_out_BASLER, cv2.CAP_GSTREAMER, 10, (frame_width, frame_height))

        print("WAIT SAVER LOOP ")

        while internal_saver_status == 0:
            internal_saver_status = global_status.value
            time.sleep(0.5)


        print("|_|_| SAVER READY!,  local_status:", internal_saver_status)


        while internal_saver_status == 1 or q.qsize() > 0:
            qsize = q.qsize()
            # print("Q long: ", qsize)
            img_basler = q.get()
            print(img_basler)
            out_BASLER.write(img_basler)

        print("BASLER SAVER RELEASED")
        out_BASLER.release()

        print("___SAVER___ENDED RECORDING_____")

