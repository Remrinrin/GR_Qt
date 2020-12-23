import sys
import cv2 as cv
import time
import shutil
import os
from PyQt5.QtWidgets import QMainWindow, QApplication
from GR_Qt_GUI import *
from GR_run import *

global camera
global Recognize


class main_win(QMainWindow,Ui_GR_Qt):
    def __init__(self,parent=None):
        super(main_win,self).__init__()
        self.setupUi(self)
        self.init_Win()
    def init_Win(self):
        self.Cam_Open.clicked.connect(self.CameraOpen)
        self.Cam_Close.clicked.connect(self.CameraClose)
        self.Reco_Start.clicked.connect(self.RecStart)
        self.Reco_Stop.clicked.connect(self.RecStop)
        self.Del_Log.clicked.connect(self.LogDelete)
        self.Log_Widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.Log_Widget.currentItemChanged.connect(self.show_Log)

    def CameraOpen(self):
        global camera
        global Recognize
        camera = True
        Recognize = False
        ROI_y = 30
        ROI_x = 200
        ROI_height = 310
        ROI_width = 300
        capture = cv.VideoCapture(0)
        success, frame = capture.read()
        index = 1
        while success and cv.waitKey(1) == -1:
            img_ROI = frame[ROI_y:(ROI_y + ROI_height), ROI_x:(ROI_x + ROI_width)]
            cv.imshow('Gesture', img_ROI)
            index += 1
            if camera == False:
                break
            if Recognize == True and index % 30 ==0:
                index = 1
                new_result = GRStart(img_ROI)
                new_result = str(new_result)
                self.HandPose.setPixmap(QtGui.QPixmap('data/testImage/pose/test_pose.jpg'))
                localtime = time.strftime("%H:%M:%S", time.localtime())
                _translate = QtCore.QCoreApplication.translate
                self.Result_Label.setText(_translate("GR_Qt", "当前结果编号："+new_result+"号"))
                self.Log_Widget.addItem(_translate("GR_Qt", "时间：[" + localtime +"]；结果：" + new_result + "号"))
                log_name = localtime[0:2] + localtime[3:5] + localtime[6:8]
                path_his = 'data/testImage/history/' + log_name + '.jpg'
                shutil.copyfile('data/testImage/pose/test_pose.jpg',path_his)
            success, frame = capture.read()
        capture.release()
        cv.destroyAllWindows()

    def CameraClose(self):
        global camera
        camera = False
    def RecStart(self):
        global Recognize
        Recognize = True
    def RecStop(self):
        global Recognize
        Recognize = False  
    def LogDelete(self):
        self.Log_Widget.clear()
        for file in os.listdir('data/testImage/history/'):
            os.remove('data/testImage/history/' + file)
    def show_Log(self):
        if self.Log_Widget.currentItem() == None:
            return
        else:
            log_name=self.Log_Widget.currentItem().text()
            print(log_name)
            log_name = log_name[4:6] + log_name[7:9] + log_name[10:12]
            path_his = 'data/testImage/history/' + log_name + '.jpg'
            self.HandPose.setPixmap(QtGui.QPixmap(path_his))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    run_win = main_win()
    run_win.show()
    sys.exit(app.exec())