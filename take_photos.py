import cv2 as cv

img_roi_y = 30
img_roi_x = 200
img_roi_height = 310
img_roi_width = 300
capture = cv.VideoCapture(0)
index = 1
num = 0
while True:
    ret, frame = capture.read()
    if ret is True:
        img_roi = frame[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
        cv.imshow("frame", img_roi)
        index += 1
        if index % 5 == 0:
            num += 1
            cv.imwrite("data/train/photo/"
                       + "gesture_5."+str(num) + ".jpg", img_roi)
        c = cv.waitKey(50)
        if c == 27:
            break
        if index == 1000:
            break
    else:
        break

cv.destroyAllWindows()
capture.release()
