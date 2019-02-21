from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import sys
import numpy as np
import cv2
import math
import os

path_dataset = "dataset"
path_right_img = os.path.join(path_dataset, "right_image")
path_lift_img = os.path.join(path_dataset, "lift_image")
path_depth_frame = os.path.join(path_dataset, "depth_frame")


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def click_and_crop(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        FRAME_WIDTH = kinect.depth_frame_desc.Width
        FRAME_HEIGHT = 424
        center = int(x+y*kinect.depth_frame_desc.Width)
        # distanse = 0.1236 * \
        #     math.tan(float(frame[center] / 2842.5 + 1.1863))

        distanse = frame[center]
        cv2.imwrite(os.path.join(path_depth_frame,
                                 "img_d"+str(distanse)+".png"), cv2.flip(frame_, 1))
        np.savetxt(os.path.join(path_depth_frame,
                                "frame"+str(distanse)+".txt"), frame)
        cv2.imwrite(os.path.join(path_right_img, "img_r" +
                                 str(distanse)+".png"), righ_cam)
        cv2.imwrite(os.path.join(path_lift_img, "img_l" +
                                 str(distanse)+".png"), left_cam)


# Kinect runtime object, we want only color and body frames
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
capR = cv2.VideoCapture(0)
capL = cv2.VideoCapture(1)
count = 0
# print("Camer Property")
# print("CAP_PROP_POS_MSEC",capL.get(0), capR.get(0))
# print("CAP_PROP_POS_FRAMES",capL.get(1), capR.get(1))
# print("CAP_PROP_POS_AVI_RATIO", capL.get(2), capR.get(2))
# print("CAP_PROP_FRAME_WIDTH", capL.get(3), capR.get(3))
# print("CAP_PROP_FRAME_HEIGHT", capL.get(4), capR.get(4))
# print("CAP_PROP_FPS", capL.get(5), capR.get(5))
# print("CAP_PROP_FOURCC", capL.get(6), capR.get(6))
# print("CAP_PROP_FRAME_COUNT", capL.get(7), capR.get(7))
# print("CAP_PROP_FORMAT", capL.get(8), capR.get(8))
# print("CAP_PROP_MODE", capL.get(9), capR.get(9))
# print("CAP_PROP_BRIGHTNESS", capL.get(10), capR.get(10))
# print("CAP_PROP_CONTRAST", capL.get(11), capR.get(11))
# print("CAP_PROP_SATURATION", capL.get(12), capR.get(12))
# print("CAP_PROP_HUE", capL.get(13), capR.get(13))
# print("CAP_PROP_GAIN", capL.get(14), capR.get(14))
# print("CAP_PROP_EXPOSURE", capL.get(15), capR.get(15))
# print("CAP_PROP_CONVERT_RGB", capL.get(16), capR.get(16))
# print("CAP_PROP_WHITE_BALANCE", capL.get(17), capR.get(17))
# print("CAP_PROP_RECTIFICATION", capL.get(18), capR.get(18))
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
while True:
    # --- Getting frames and drawing
    if kinect.has_new_depth_frame():
        frame = kinect.get_last_depth_frame()

        # Capture frame-by-frame
        retR, frameR = capR.read()
        retL, frameL = capL.read()

        # Our operations on the frame come here
        righ_cam = frameR
        left_cam = frameL

        # Display the resulting frame
        cv2.imshow('Rframe', righ_cam)
        cv2.imshow('Lframe', left_cam)

        # frame_ = frame.reshape(424, 512)
        depth_array = np.array(frame.reshape(424, 512), dtype=np.float32)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)

        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        
        disparity = stereo.compute(cv2.cvtColor(
            left_cam, cv2.COLOR_RGB2GRAY), cv2.cvtColor(righ_cam, cv2.COLOR_RGB2GRAY))

        cv2.imshow("depth_map", disparity)

        frame_ = depth_array*255
        cv2.imshow("image", cv2.flip(frame_, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # self.draw_depth_frame(frame, self._frame_surface)
        frame = None
        # break
# When everything done, release the capture
capR.release()
capL.release()
kinect.close()
cv2.destroyAllWindows()
