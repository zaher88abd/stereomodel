import cv2
import numpy as np

depth = None
# Filtering
kernel= np.ones((3,3),np.uint8)

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')

def click_and_crop(event, x, y, flags, param):
    if not depth is None and event == cv2.EVENT_LBUTTONDOWN:
        print(depth[x, y])


calibration = np.load(r"stereoCalibration960x720.npz", allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

# cv2.namedWindow("depth")
# cv2.setMouseCallback("depth", click_and_crop)

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
# *******************************************
# ***** Parameters for the StereoVision *****
# *******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=5,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2)

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#
# def set_stereo_parameter(x):
#     stereoMatcher.setMinDisparity(cv2.getTrackbarPos('minDisparity', 'depth'))
#     stereoMatcher.setNumDisparities(cv2.getTrackbarPos('numDisparities', 'depth') * 16)
#     stereoMatcher.setBlockSize(cv2.getTrackbarPos('BlockSize', 'depth'))
#     stereoMatcher.setSpeckleRange(cv2.getTrackbarPos('SpeckleRange', 'depth'))
#     stereoMatcher.setSpeckleWindowSize(cv2.getTrackbarPos('SpeckleWindowSize', 'depth'))
#
#     # # SGMB
#     # stereoMatcher.setMinDisparity(cv2.getTrackbarPos('minDisparity', 'depth'))
#     # stereoMatcher.setNumDisparities(cv2.getTrackbarPos('numDisparities', 'depth') * 16)
#     # stereoMatcher.setSadWindowSize(cv2.getTrackbarPos('SADWindowSize', 'depth'))
#     # stereoMatcher.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'depth'))
#     # stereoMatcher.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'depth'))
#     # stereoMatcher.setSpeckleRange(cv2.getTrackbarPos('speckleWindowSize', 'depth'))
#     # stereoMatcher.setSpeckleWindowSize(cv2.getTrackbarPos('speckleRange', 'depth') * 16)
#     pass
#
#
# cv2.createTrackbar('minDisparity', 'depth', 1, 255, set_stereo_parameter)
# cv2.createTrackbar('numDisparities', 'depth', 1, 100, set_stereo_parameter)
# cv2.createTrackbar('BlockSize', 'depth', 3, 25, set_stereo_parameter)
# cv2.createTrackbar('SpeckleRange', 'depth', 1, 200, set_stereo_parameter)
# cv2.createTrackbar('SpeckleWindowSize', 'depth', 5, 15, set_stereo_parameter)
# # cv2.createTrackbar('speckleWindowSize', 'depth', 50, 200, set_stereo_parameter)
# # cv2.createTrackbar('speckleRange', 'depth', 1, 100, set_stereo_parameter)
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
    fixedLeft = cv2.remap(frameL, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    fixedRight = cv2.remap(frameR, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(grayLeft, grayRight)  # .astype(np.float32)/ 16
    dispL = disp
    dispR = stereoR.compute(grayRight, grayLeft)
    dispL = np.int16(dispL)
    dispR = np.int16(dispR)
    # Using the WLS filter
    filteredImg = wls_filter.filter(dispL, grayLeft, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    # cv2.imshow('Disparity Map', filteredImg)
    disp = ((disp.astype(
        np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect

    # Filtering the Results with a closing filter
    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,
                               kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

    # Colors map
    dispc = (closing - closing.min()) * 255
    dispC = dispc.astype(
        np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

    cv2.imshow('leftFix', fixedLeft)
    # cv2.imshow('right', frameR)
    cv2.imshow('rightFix', fixedRight)
    # cv2.imshow('depth1', (depth - np.min(depth) / num_disp))

    cv2.imshow('Filtered Color Depth',filt_Color)
    cv2.setMouseCallback("Filtered Color Depth", coords_mouse_disp, filt_Color)

    # print(np.max(depth), np.min(depth))
    key = cv2.waitKey(100)
    if key & 0xFF == ord('q'):
        break
    elif not key == -1:
        print(key)

print(frameL.shape)
cameraL.release()
cameraR.release()
cv2.destroyAllWindows()
