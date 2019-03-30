import cv2
import numpy as np

depth = None


def click_and_crop(event, x, y, flags, param):
    if not depth is None and event == cv2.EVENT_LBUTTONDOWN:
        print(depth[x, y])


calibration = np.load(r"stereoCalibration1280x720.npz", allow_pickle=False)
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
cameraL.set(cv2.CAP_PROP_FRAME_WIDTH, 10000);
cameraL.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000);
cameraR.set(cv2.CAP_PROP_FRAME_WIDTH, 10000);
cameraR.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000);
cameraL.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
cameraR.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
# cameraL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cameraR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images
window_size = 3
min_disp = 16
num_disp = 112 - min_disp

stereoMatcher = cv2.StereoBM_create()


def set_stereo_parameter(x):
    stereoMatcher.setMinDisparity(cv2.getTrackbarPos('minDisparity', 'depth'))
    stereoMatcher.setNumDisparities(cv2.getTrackbarPos('numDisparities', 'depth') * 16)
    stereoMatcher.setBlockSize(cv2.getTrackbarPos('BlockSize', 'depth'))
    stereoMatcher.setSpeckleRange(cv2.getTrackbarPos('SpeckleRange', 'depth'))
    stereoMatcher.setSpeckleWindowSize(cv2.getTrackbarPos('SpeckleWindowSize', 'depth'))

    # # SGMB
    # stereoMatcher.setMinDisparity(cv2.getTrackbarPos('minDisparity', 'depth'))
    # stereoMatcher.setNumDisparities(cv2.getTrackbarPos('numDisparities', 'depth') * 16)
    # stereoMatcher.setSadWindowSize(cv2.getTrackbarPos('SADWindowSize', 'depth'))
    # stereoMatcher.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'depth'))
    # stereoMatcher.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'depth'))
    # stereoMatcher.setSpeckleRange(cv2.getTrackbarPos('speckleWindowSize', 'depth'))
    # stereoMatcher.setSpeckleWindowSize(cv2.getTrackbarPos('speckleRange', 'depth') * 16)
    pass


cv2.createTrackbar('minDisparity', 'depth', 1, 255, set_stereo_parameter)
cv2.createTrackbar('numDisparities', 'depth', 1, 100, set_stereo_parameter)
cv2.createTrackbar('BlockSize', 'depth', 3, 25, set_stereo_parameter)
cv2.createTrackbar('SpeckleRange', 'depth', 1, 200, set_stereo_parameter)
cv2.createTrackbar('SpeckleWindowSize', 'depth', 5, 15, set_stereo_parameter)
# cv2.createTrackbar('speckleWindowSize', 'depth', 50, 200, set_stereo_parameter)
# cv2.createTrackbar('speckleRange', 'depth', 1, 100, set_stereo_parameter)
CROP_WIDTH = 960


def cropHorizontal(image):
    CAMERA_WIDTH = image.shape[1]
    return image[:,
           int((CAMERA_WIDTH - CROP_WIDTH) / 2):
           int(CROP_WIDTH + (CAMERA_WIDTH - CROP_WIDTH) / 2)]


DEPTH_VISUALIZATION_SCALE = 1
while True:
    retL, frameL = cameraL.read()
    retR, frameR = cameraR.read()
    if not retL and not retR:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    frameL = cropHorizontal(frameL)
    frameR = cropHorizontal(frameR)
    fixedLeft = cv2.remap(frameL, leftMapX, leftMapY, cv2.INTER_LINEAR)
    fixedRight = cv2.remap(frameR, rightMapX, rightMapY, cv2.INTER_LINEAR)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)
    # Normalised [0,255] as integer
    # depth = (depth - np.min(depth)) / np.ptp(depth)

    # cv2.imshow('left', frameL)
    cv2.imshow('leftFix', fixedLeft)
    # cv2.imshow('right', frameR)
    cv2.imshow('rightFix', fixedRight)
    cv2.imshow('depth', (depth - np.min(depth) / num_disp))
    # print(np.max(depth), np.min(depth))
    key = cv2.waitKey(100)
    if key & 0xFF == ord('q'):
        break
    elif not key == -1:
        print(key)

print(depth.shape)
print(frameL.shape)
cameraL.release()
cameraR.release()
cv2.destroyAllWindows()
