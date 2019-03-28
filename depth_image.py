import cv2
import numpy as np

depth = None


def click_and_crop(event, x, y, flags, param):
    if not depth is None and event == cv2.EVENT_LBUTTONDOWN:
        print(depth[x, y])


calibration = np.load(r"stereoCalibration.npz", allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

cv2.namedWindow("depth")
cv2.setMouseCallback("depth", click_and_crop)

stereoMatcher = cv2.StereoBM_create()
# get cameras
cameraL = cv2.VideoCapture(1)
cameraR = cv2.VideoCapture(0)
cameraL.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
cameraR.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
# cameraL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cameraR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images
stereoMatcher = cv2.StereoSGBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)
DEPTH_VISUALIZATION_SCALE = 1000
while True:
    retL, frameL = cameraL.read()
    retR, frameR = cameraR.read()
    if not retL and not retR:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    fixedLeft = cv2.remap(frameL, leftMapX, leftMapY, cv2.INTER_LINEAR)
    fixedRight = cv2.remap(frameR, rightMapX, rightMapY, cv2.INTER_LINEAR)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)
    # Normalised [0,255] as integer
    # depth = (depth - np.min(depth)) / np.ptp(depth)

    cv2.imshow('left', frameL)
    cv2.imshow('leftFix', fixedLeft)
    cv2.imshow('right', frameR)
    cv2.imshow('rightFix', fixedRight)
    cv2.imshow('depth', depth/1000)
    # print(np.max(depth), np.min(depth))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(depth.shape)
print(frameL.shape)
cameraL.release()
cameraR.release()
cv2.destroyAllWindows()
