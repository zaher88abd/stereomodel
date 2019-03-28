import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
all_3d_points = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.

# get cameras
cameraL = cv2.VideoCapture(1)
cameraR = cv2.VideoCapture(0)
cameraL.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
cameraR.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off

# rt=0 M1=0; d1=0 r1=0; t1=0
# rt=0 M2=0; d2=0 r2=0; t2=0
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
x, y = np.meshgrid(range(7), range(6))
world_points = np.hstack((x.reshape(42, 1), y.reshape(42, 1), np.zeros((42, 1)))).astype(np.float32)
outputFile="stereoCalibration.npz"
while True:
    retL, frameL = cameraL.read()
    retR, frameR = cameraR.read()
    if not retL and not retR:
        break
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (7, 6), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (7, 6), None)
    img_shape = grayR.shape[::-1]

    ret_L = False
    ret_R = False
    # If found, add object points, image points (after refining them)
    if retL and retR:
        all_3d_points.append(world_points)
        corners2L = cv2.cornerSubPix(
            grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(corners2L)
        frameL = cv2.drawChessboardCorners(frameL, (7, 6), corners2L, retL)
        cv2.imshow('imgL', frameL)
        corners2R = cv2.cornerSubPix(
            grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(corners2R)

        frameR = cv2.drawChessboardCorners(frameR, (7, 6), corners2R, retR)

        cv2.imshow('imgR', frameR)

        ret, mtxL, distL, rvecs, tvecs = cv2.calibrateCamera(all_3d_points, imgpointsL,
                                                             (grayL.shape[1], grayL.shape[0]), None, None)
        ret, mtxR, distR, rvecs, tvecs = cv2.calibrateCamera(all_3d_points, imgpointsR,
                                                             (grayR.shape[1], grayR.shape[0]), None, None)

        retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objectPoints=all_3d_points, imagePoints1=imgpointsL,
                                                             imagePoints2=imgpointsR, cameraMatrix1=mtxL,
                                                             distCoeffs1=distL, cameraMatrix2=mtxR, distCoeffs2=distR,
                                                             imageSize=(grayL.shape[1], grayL.shape[0]),
                                                             flags=cv2.CALIB_FIX_INTRINSIC)

        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        camera_model = dict([('R', R), ('T', T),
                             ('E', E), ('F', F)])
        print(camera_model)
        OPTIMIZE_ALPHA = 0.25
        (leftRectification, rightRectification, leftProjection, rightProjection,
         dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
            mtxL, distL, mtxR, distR,
            img_shape, R, T, None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            mtxL, distL, leftRectification,
            leftProjection, img_shape, cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            mtxR, distR, rightRectification,
            rightProjection, img_shape, cv2.CV_32FC1)

        np.savez_compressed(outputFile, imageSize=img_shape,
                            leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
                            rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)

    cv2.imshow('imgL', frameL)
    cv2.imshow('imgR', frameR)

    k = cv2.waitKey(1)
    if k == 27:
        break

cameraL.release()
cameraR.release()
cv2.destroyAllWindows()
