import numpy as np
import cv2
import tqdm

CROP_WIDTH = 960


def cropHorizontal(image):
    CAMERA_WIDTH = image.shape[1]
    return image[:,
           int((CAMERA_WIDTH - CROP_WIDTH) / 2):
           int(CROP_WIDTH + (CAMERA_WIDTH - CROP_WIDTH) / 2)]


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 27, 0.001)
box_r = 9
box_c = 6
cxr = box_c * box_r
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((box_c * box_r, 3), np.float32)
objp[:, :2] = np.mgrid[0:box_r, 0:box_c].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
all_3d_points = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.

# get cameras
cameraL = cv2.VideoCapture(1)
cameraR = cv2.VideoCapture(0)
cameraL.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cameraL.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
cameraR.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cameraR.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
print("CameraL", cameraR.get(cv2.CAP_PROP_FRAME_WIDTH), cameraR.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("CameraR", cameraR.get(cv2.CAP_PROP_FRAME_WIDTH), cameraR.get(cv2.CAP_PROP_FRAME_HEIGHT))
cameraL.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
cameraR.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5
x, y = np.meshgrid(range(box_r), range(box_c))
world_points = np.hstack((x.reshape(cxr, 1), y.reshape(cxr, 1), np.zeros((cxr, 1)))).astype(np.float32)
import os

file_counter = int(len(os.listdir("calibration_images")) / 2)
while True:
    retL, frameL = cameraL.read()
    retR, frameR = cameraR.read()

    if not retL and not retR:
        break
    frameR = cropHorizontal(frameR)
    frameL = cropHorizontal(frameL)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (box_r, box_c), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (box_r, box_c), None)
    img_shape = grayR.shape[::-1]

    # If found, add object points, image points (after refining them)
    if retL and retR:
        cv2.imwrite("calibration_images/" + "l" + str(file_counter) + ".jpg", frameL)
        cv2.imwrite("calibration_images/" + "r" + str(file_counter) + ".jpg", frameR)
        file_counter += 1
        print("Number of image:", file_counter)
        all_3d_points.append(world_points)
        corners2L = cv2.cornerSubPix(
            grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(corners2L)
        frameL = cv2.drawChessboardCorners(frameL, (box_r, box_c), corners2L, retL)
        corners2R = cv2.cornerSubPix(
            grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(corners2R)

        frameR = cv2.drawChessboardCorners(frameR, (box_r, box_c), corners2R, retR)

        cv2.imshow('imgL', frameL)
        cv2.imshow('imgR', frameR)
    else:
        cv2.imshow('imgL', frameL)
        cv2.imshow('imgR', frameR)

    k = cv2.waitKey(10)
    if k == 27:
        break

cameraL.release()
cameraR.release()
cv2.destroyAllWindows()
