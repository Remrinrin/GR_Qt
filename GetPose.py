# encoding=utf8
import cv2
import time
import numpy as np

protoFile = "preHand/hand/pose_deploy.prototxt"
weightsFile = "preHand/hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def GetPose(path_frame):
    frame = cv2.imread(path_frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    frame_new = np.zeros_like(frame)
    frame_new = np.full(frame_new.shape,255)

    aspect_ratio = frameWidth/frameHeight

    threshold = 0.1

    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if (points[partA] and points[partB]):
            cv2.line(frame_new, points[partA], points[partB], (0, 0, 0), 2)
            cv2.circle(frame_new, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame_new, points[partB], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

    #cv2.imwrite('image/Output-Keypoints.jpg', frame_new)
    return frame_new

#GetPose("image/123.png")
