import time

from embedded_platform_utils import *
from pypylon import pylon

print(" CV2  version: ", cv2.__version__)
print("build info: ",cv2.getBuildInformation())


#CONFIG
#VIDEO LOOP
  #WAIT PLAY LOOP
  #RECORD LOOP

def BASLER_capture():


    """
    functioon to acquire images from basler dart camera with pylon

    :param q: queue where to put the images extracted
    :param status: status to control the events of the process
    :return: nothing
    """

    # conecting to the first available camera
    try:
        print("BASLER CONFIGURATION...")

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
        print("basler configuration failed", e)





    frame_c = 0
    while frame_c < 100:

        # start_time = time.time()

        frame_c += 1
        try:

            if camera.IsGrabbing():

                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    # Access the image data
                    image = converter.Convert(grabResult)
                    img_basler = image.GetArray()
                    #print(frame_c)

                    cv2.imshow('dept!!!h Stream', resize_image(img_basler,200))

                    key = cv2.waitKey(1)
                    if key == 27:
                        # result.release()
                        # cv2.destroyAllWindows()
                        break
                    # end_time = time.time()
                    #
                    # # Calculate the total time taken
                    # total_time = end_time - start_time
                    #
                    # # Calculate FPS
                    # fps = 1 / total_time
                    # print(f"FPS_IMG: {fps}")

                else:
                    print("ERROR: camera not succeded, no image")

            else:
                print("ERROR: camera is not grabbing")

        except Exception as e:
            print("ERROR basler in loop wait4fr: %s", e)


    print("CYCLE TERMINATED-READY NEW AQUISITION")


BASLER_capture()


