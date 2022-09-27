import pyrealsense2 as rs
import cv2


'''
pip install pyrealsense2
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev


'''

import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.accel,rs.format.motion_xyz32f,200)
config.enable_stream(rs.stream.gyro,rs.format.motion_xyz32f,200)

align_to = rs.stream.color
align = rs.align(align_to)
try:
# Start streaming
    pipeline.start(config)
except Exception as e:
    print("error pipeline starting:||||:: %s", str(e))

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames.as_motion_frame().get_motion_data()

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('Color Stream', color_image)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
except Exception as e:
    print("error:||||:: %s", str(e))
#pipeline.stop()