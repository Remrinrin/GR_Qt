import os
from GetPose import *
import cv2

photo_dir = 'data/train/photo/'
pose_dir = 'data/train/pose/'
for file in os.listdir(photo_dir):
    photo = cv2.imread(photo_dir + file)
    pose = GetPose(photo)
    cv2.imwrite(pose_dir + file,pose)
    print(file + "____done")

