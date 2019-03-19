#####################################################################

# Example :  contour edges for a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################
import numpy as np
import cv2
import argparse
import sys
import pandas as pd

#####################################################################

keep_processing = True;

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

#####################################################################

# this function is called as a call-back everytime the trackbar is moved
# (here we just do nothing)

def nothing(x):
    pass
def draw_flow(img, flow, step=18):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    # for (x1, y1), (x2, y2) in lines:
    #     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis
#####################################################################
def h_concatenate(img1, img2):

    # get size and channels for both images

    height1 = img1.shape[0];
    width1 = img1.shape[1];
    if (len(img1.shape) == 2):
        channels1 = 1;
    else:
        channels1 = img1.shape[2];

    height2 = img2.shape[0];
    width2 = img2.shape[1];
    if (len(img2.shape) == 2):
        channels2 = 1;
    else:
        channels2 = img2.shape[2];

    # make all images 3 channel, or assume all same channel

    if ((channels1 > channels2) and (channels1 == 3)):
        out2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR);
        out1 = img1;
    elif ((channels2 > channels1) and (channels2 == 3)):
        out1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR);
        out2 = img2;
    else: # both must be equal
        out1 = img1;
        out2 = img2;

    # height of first image is master height, width can remain unchanged

    if (height1 != height2):
        out2 = cv2.resize(out2, (height1, width2))

    return np.hstack((out1, out2));

# define video capture object

cap = cv2.VideoCapture();

# define display window name

windowName = "Largest Area Contour"; # window name
windowName2 = "All Contours"; # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL);

    # add some track bar controllers for settings

    lower_threshold = 118;
    cv2.createTrackbar("param1", windowName2, lower_threshold, 255, nothing);
    upper_threshold = 8;
    cv2.createTrackbar("param2", windowName2, upper_threshold, 255, nothing);
    smoothing_neighbourhood = 0;
    cv2.createTrackbar("minRadius", windowName2, smoothing_neighbourhood, 150, nothing);
    sobel_size = 13; # greater than 7 seems to crash
    cv2.createTrackbar("maxRadius", windowName2, sobel_size, 150, nothing);
    sthres = 157; # greater than 7 seems to crash
    cv2.createTrackbar("threshold", windowName2, sthres, 255, nothing);

    if (cap.isOpened):
        ret, frame = cap.read();
    print("frameshape",frame.shape)

    # convert image to grayscale to be previous frame

    prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read();

        # get parameters from track bars

        param1 = cv2.getTrackbarPos("param1", windowName2);
        param2 = cv2.getTrackbarPos("param2", windowName2);
        minRadius = cv2.getTrackbarPos("minRadius", windowName2);
        maxRadius = cv2.getTrackbarPos("maxRadius", windowName2);
        thres = cv2.getTrackbarPos("threshold", windowName2);

        cv2.imwrite('cir4.jpg', frame,(cv2.IMWRITE_JPEG_QUALITY, 80))

        img = cv2.imread('cir4.jpg',0)
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        ret1,cimg = cv2.threshold(img,thres,255,cv2.THRESH_BINARY)
        # cimg=cv2.adaptiveThreshold(img,155,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # cimg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,111,21)
        # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,5,param1=118,param2=8,minRadius=0,maxRadius=7)
        circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,5,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)

        circles = np.uint16(np.around(circles))
        no_c=circles.shape[1]
        print(no_c)


                        # cv2.putText(img, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (1,50)
        fontScale              = 2
        fontColor              = (255,255,0)
        lineType               = 4


        cv2.putText(img,'Bubble Statistics', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

                # cv2.putText(img, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (1,100)
        fontScale              = 2
        fontColor              = (255,255,0)
        lineType               = 4

        radius_all=[]
        for i in circles[0,:]:
        # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),1)
        # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(255,0,0),1)
            # print('radius',i[2])
            radius_all.append(i[2])
        # print('radius_all',radius_all)

        # df = pd.DataFrame(radius_all)
        df = pd.Series(radius_all)
        # s = pd.Series([radius_all])
        print("stats",df.describe())
        # [a1,b2,c3,d4,e5,f6, g7, h8]=df.describe()
        count, mean, std, min, t25,  t50, t75, max = df.describe()


        cv2.putText(img,'Number of bubbles :%d' % no_c, 
            (1,100), 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(img,'Min size of bubbles:%d' %  min, 
            (1,150), 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(img,'Max size of bubbles:%d' %  max, 
            (1,200), 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(img,'Mean size of bubbles :%d' % mean, 
            (1,250), 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(img,'Std of bubbles :%d' % std, 
            (1,300), 
            font, 
            fontScale,
            fontColor,
            lineType)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        print("mag",np.amax(mag))
        print("magshape",mag.shape)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mv=np.amax(mag)
        # min_mv=np.amean(mag)
        # mean_mv=np.amax(mag)



        cv2.putText(gray,'Maximum flow rate :%d' % mv, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)


        print("mag",mv)
        print("magshape",mag.shape)

        # display image with optic flow overlay


        # cv2.imshow(windowName, draw_flow(gray, flow))

        # display image
        cv2.imshow(windowName,draw_flow(gray, flow));
        cv2.imshow(windowName2, img)# h_concatenate(cimg, img));

        key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit  / press "f" for fullscreen display

        if (key == ord('x')):
            keep_processing = False;
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.");

#####################################################################
