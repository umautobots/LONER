import numpy as np 
import cv2
from fusion_portable_calibration import FusionPortableCalibration
import argparse
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

parser = argparse.ArgumentParser("stereo to rgbd")
parser.add_argument("rosbag_path", type=str)
parser.add_argument("calib_path", type=str)

args = parser.parse_args()

bridge = CvBridge()

### Adapted from https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
calibration = FusionPortableCalibration(args.calib_path)

K_left = calibration.left_cam_intrinsic["K"]
distortion_left = calibration.left_cam_intrinsic["distortion_coeffs"]
rect_left = calibration.left_cam_intrinsic["rectification_matrix"]
size_left = (calibration.left_cam_intrinsic["width"], calibration.left_cam_intrinsic["height"])

xmap_left, ymap_left = cv2.initUndistortRectifyMap(K_left, distortion_left, rect_left, K_left, size_left, cv2.CV_32FC1)

K_right = calibration.right_cam_intrinsic["K"]
distortion_right = calibration.right_cam_intrinsic["distortion_coeffs"]
rect_right = calibration.right_cam_intrinsic["rectification_matrix"]
size_right = (calibration.right_cam_intrinsic["width"], calibration.right_cam_intrinsic["height"])

xmap_right, ymap_right = cv2.initUndistortRectifyMap(K_right, distortion_right, rect_right, K_right, size_right, cv2.CV_32FC1)

bag = rosbag.Bag(args.rosbag_path)

bag_it = bag.read_messages(topics=["/stereo/frame_left/image_raw", "/stereo/frame_right/image_raw"])

def get_next_image():
    topic1, msg1, timestamp1 = next(bag_it)
    _, msg2, timestamp2 = next(bag_it)
    
    assert (timestamp2 - timestamp1).to_sec() < 0.01, "Timestamps too far apart"

    if "left" in topic1:
        left_im = msg1
        right_im = msg2
    else:
        left_im = msg2
        right_im = msg1

    left_im_cv2 = bridge.imgmsg_to_cv2(left_im)
    right_im_cv2 = bridge.imgmsg_to_cv2(right_im)

    return left_im_cv2, right_im_cv2

def nothing(x):
    pass
 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
 
cv2.createTrackbar('minDisparity','disp',5,25,nothing)
cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('P1','disp',8,32,nothing)
cv2.createTrackbar('P2','disp',32,64,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createButton('Get Next Image', get_next_image)
 
# Creating an object of StereoBM algorithm
stereo: cv2.StereoSGBM = cv2.StereoSGBM_create()
 
imgL, imgR = get_next_image()

while True:
 
    imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
 
    # Applying stereo image rectification on the left image
    Left_nice= cv2.remap(imgL_gray,
              xmap_left,
              ymap_left,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
     
    # Applying stereo image rectification on the right image
    Right_nice = cv2.remap(imgR_gray,
              xmap_right,
              ymap_right,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
 
    # Updating the parameters based on the trackbar positions
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')
    p1 = cv2.getTrackbarPos('P1','disp')*3*blockSize**2
    p2 = cv2.getTrackbarPos('P2','disp')*3*blockSize**2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleWindowSize','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
     
    # Setting the updated parameters before computing disparity map
    stereo.setMinDisparity(minDisparity)
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setP1(p1)
    stereo.setP2(p2)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)


    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice,Right_nice)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    cv2.imshow("disp",disparity)
 
    # Close window using esc key
    if cv2.waitKey(1) == 27:
        break