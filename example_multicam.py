# Libraries
import pyrealsense2 as rs
import numpy as np
import cv2 as cv


def findDevices():
    ctx = rs.context()  # Create librealsense context for managing devices
    serials = []
    if (len(ctx.devices) > 0):
        for dev in ctx.devices:
            print('Found device: ', \
                  dev.get_info(rs.camera_info.name), ' ', \
                  dev.get_info(rs.camera_info.serial_number))
            serials.append(dev.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")

    return serials, ctx


def enableDevices(serials, ctx, resolution_width=640, resolution_height=480, frame_rate=30):
    pipelines = []
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        cfg.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        pipe.start(cfg)
        pipelines.append([serial, pipe])

    return pipelines


def Visualize(pipelines):
    align_to = rs.stream.color
    align = rs.align(align_to)

    for (device, pipe) in pipelines:
        # Get frameset of color and depth
        frames = pipe.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Render images
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        cv.imshow('RealSense' + device, images)
        key = cv.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            return True

        # Save images and depth maps from both cameras by pressing 's'
        if key == 115:
            cv.imwrite(str(device) + '_aligned_depth.png', depth_image)
            cv.imwrite(str(device) + '_aligned_color.png', color_image)
            print('Save')


def pipelineStop(pipelines):
    for (device, pipe) in pipelines:
        # Stop streaming
        pipe.stop()

    # -------Main program--------


serials, ctx = findDevices()

# Define some constants
resolution_width = 640  # pixels
resolution_height = 480  # pixels
frame_rate = 30  # fps

pipelines = enableDevices(serials, ctx, resolution_width, resolution_height, frame_rate)

try:
    while True:
        exit = Visualize(pipelines)
        if exit == True:
            print('Program closing...')
            break
finally:
    pipelineStop(pipelines)