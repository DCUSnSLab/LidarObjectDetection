import pyqtgraph as pg
import vispy
from vispy.scene import visuals, TurntableCamera

import ros_numpy
import sensor_msgs

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, Qt, QThread, QTimer
from PyQt5.QtGui import *
import rosbag
import rospy
import time, random
from threading import Thread
import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import pcl
from sklearn.linear_model import RANSACRegressor

import csv

read_topic = '/velodyne_points'  # 메시지 타입
count = 1
class ExMain(QWidget):
    def __init__(self):
        super().__init__()

        self.ALGO_FLAG = 2  # 1 : kdtree. 2 : dbscan
        self.clusterLabel = list()

        self.evaluation = {60: [[-19.80, 3.83, 4.7, 1.7], [-11.99, 3.33, 4.7, 1.7], [-5.14, 2.89, 4.7, 1.7], [2.44, 1.89, 4.7, 1.7], [11.23, 1.38, 4.7, 1.7], [-16.5, -0.65, 9, 3], [8.16, -2.15, 4.7, 1.7], [16.65, -2.66, 4.7, 1.7], [18.64, 0.04, 4.7, 1.7]],
                           120: [[-24.20, 3.47, 4.7, 1.7], [-21.09, -0.71, 9, 3], [-3.10, 2.22, 4.7, 1.7], [3.56, 8.16, 4.7, 1.7], [14.53, -2.45, 4.7, 1.7], [24.40, 0.44, 4.7, 1.7], [-16.04, 3.26, 4.7, 1.7]],
                           180: [[-31.15, -0.39, 9, 3], [-19.09, 3.45, 4.7, 1.7], [3.03, 3.47, 4.7, 1.7], [26.91, -9.94, 4.7, 1.7], [23.40, -2.17, 4.7, 1.7], [-33.08, 3.36, 4.7, 1.7], [16.53, 7.38, 4.7, 1.7]],
                           240: [[-9.78, 5.73, 4.7, 1.7], [14.08, 3.04, 4.7, 1.7], [22.73, -3.85, 4.7, 1.7], [-31.03, 7.8, 4.7, 1.7]],
                           300: [[26.08, -6.58, 4.7, 1.7]],
                           360: [[-33.00, 8.01, 4.7, 1.7], [24.36, -7.29, 4.7, 1.7]],
                           420: [[8.68, -5.05, 4.7, 1.7]],
                           480: [[-12.11, -4.57, 4.7, 1.7]],
                           540: [[-14.37, 5.20, 4.7, 1.7], [-27.89, -1.76, 4.7, 1.7], [23.94, 3.90, 4.7, 1.7], [-22.34, 9.32, 9, 3]],
                           600: [[-26.50, -2.54, 4.7, 1.7], [-27.10, 10.36, 4.7, 1.7], [11.96, 4.74, 4.7, 1.7]],
                           660: [[-22.00, -3.18, 4.7, 1.7]],
                           720: [[-28.77, -4.12, 4.7, 1.7], [16.39, -6.21, 4.7, 1.7]],
                           780: [[-0.22, -4.57, 4.7, 1.7]],
                           840: [[20.69, -5.07, 4.7, 1.7], [-14.46, -3.56, 4.7, 1.7]],
                           900: [[-8.50, 4.80, 4.7, 1.7], [-0.28, -4.50, 4.7, 1.7], [-30.75, -2.97, 4.7, 1.7]],
                           960: [[-25.58, -1.92, 4.7, 1.7]],
                           1020: [],
                           1080: [],
                           1140: [],
                           1200: [[-28.18, 1.60, 4.7, 1.7], [-19.98, 4.19, 4.7, 1.7]]
                           }

        self.frame = 180
        self.correct = list()

        self.getslider = None
        secs_list = []
        nsecs_list = []
        self.btn_Flag = None
        self.Stop_flag = False
        self.D_flag = False


        self.start_secs_time = 0
        self.start_nsecs_time = 0

        self.t = 0
        self.list1 = [0, 0]
        self.flag = False


        #self.setGeometry(300, 100, 1000, 1000)  # x, y, width, height

        # 3D
        self.canvas3D = vispy.scene.SceneCanvas(keys='interactive', bgcolor='#000d1a')
        self.view3D = self.canvas3D.central_widget.add_view()
        self.view3D.camera = 'arcball'
        self.view3D.camera = TurntableCamera(fov=30.0, elevation=90.0, azimuth=-90., distance=100, translate_speed=50.0)
        grid = visuals.GridLines(parent=self.view3D.scene, scale=(5, 5))

        self.scatter = visuals.Markers(edge_color=None, size=2)
        self.view3D.add(self.scatter)

        self.canvas = pg.GraphicsLayoutWidget()
        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.disableAutoRange()
        self.view.scaleBy(s=(20, 20))
        grid = pg.GridItem()
        self.view.addItem(grid)
        #self.geometry().setWidth(1000)
        #self.geometry().setHeight(1000)
        self.setWindowTitle("realtime")

        self.hbox = QGridLayout()
        self.hbox.addWidget(self.canvas)
        self.setLayout(self.hbox)

        #point cloud 출력용
        self.spt = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='o', size=2)
        self.view.addItem(self.spt)

        # global position to display graph
        self.pos = None

        #object 출력용
        self.objs = list() #for display to graph

        # object 출력용 position과 size
        self.objsPos = list()
        self.objsSize = list()

        # 정답 및 허용 오차 범위 출력용
        self.correct_objs = list()
        self.correct_borders = list()
        self.tolerance_borders = list()

        # #출력용 object를 미리 생성해둠
        # #생성된 object의 position값을 입력하여 그래프에 출력할 수 있도록 함
        numofobjs = 150
        for i in range(numofobjs):
            obj = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5) #obj 크기는 1m로 고정시킴
            obj.setPen(pg.mkPen('w'))
            self.view.addItem(obj)
            self.objs.append(obj)

            pos = [0, 0, 0] #x, y, z
            size = [0, 0, 0] #w, h, depth
            self.objsPos.append(pos)
            self.objsSize.append(size)

        #pyqtgraph에 Correct Object 생성
        for i in range(numofobjs):
            correct_obj = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5) #obj 크기는 1m로 고정시킴
            correct_obj.setBrush(QColor(18, 241, 246))
            correct_obj.setOpacity(0.2)
            self.view.addItem(correct_obj)
            self.correct_objs.append(correct_obj)

            tolerance_border = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5)  # obj 크기는 1m로 고정시킴
            tolerance_border.setPen(pg.mkPen(QColor(18, 241, 246)))
            self.view.addItem(tolerance_border)
            self.tolerance_borders.append(tolerance_border)

            correct_border = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5)  # obj 크기는 1m로 고정시킴
            correct_border.setPen(pg.mkPen(QColor(255, 255, 0)))
            self.view.addItem(correct_border)
            self.correct_borders.append(correct_border)

        #load bagfile
        test_bagfile = '/home/ros/rosbag/2022-02-10-19-54-31.bag'
        self.bag_file = rosbag.Bag(test_bagfile)

        self.slider = None
        self.slider = QSlider(Qt.Horizontal, self)

        # self.slider.sizePolicy().horizontalPolicy(QSizePolicy.Maximum)
        self.slider.sliderMoved.connect(self.change)
        self.slider.sliderPressed.connect(self.sliderPressed)
        self.slider.sliderReleased.connect(self.sliderReleased)
        # self.slider.valueChanged(self.count_base())
        # self.count_base.valueChanged(self.slider.setValue)

        self.Table = None
        self.Table = QTableWidget(self)
        table_column = ["정지", "입력", "클릭", "프레임", "화면전환"]
        self.Table.setFixedSize(600, 80)
        self.Table.setColumnCount(5)
        self.Table.setRowCount(1)
        self.Table.setHorizontalHeaderLabels(table_column)

        self.btn = QPushButton("||")
        self.Decreasebtn = QPushButton("->")
        self.MoveBtn = QPushButton("Move")
        self.btn3D = QPushButton("3D")
        self.btn.setCheckable(True)
        self.btn3D.setCheckable(True)

        self.label = QLabel('label', self)
        self.btn.clicked.connect(self.btn_event)
        self.btn3D.clicked.connect(self.btn_3D)
        self.MoveBtn.clicked.connect(self.input_time)
        self.MoveBtn.pressed.connect(self.input_btnPressed)
        self.MoveBtn.released.connect(self.input_btnReleased)
        # self.text = QLabel(self.)
        # self.text.setText(self,"aa")
        # self.btn.setCheckable(True)
        # self.btn.clicked.connect(self.getbagfile)

        self.line = QLineEdit(self)

        self.Table.setCellWidget(0,0,self.btn)
        self.Table.setCellWidget(0,1, self.line)
        self.Table.setCellWidget(0,2, self.MoveBtn)
        self.Table.setCellWidget(0,3,self.label)
        self.Table.setCellWidget(0,4,self.btn3D)

        self.btn.clicked.connect(self.btn_event)
        self.Decreasebtn.clicked.connect(self.DecreaseButton)
        self.hbox.addWidget(self.slider)
        self.hbox.addWidget(self.Table, 2, 0)
        self.btn.setMinimumHeight(35)
        self.hbox.addWidget(self.btn)
        self.hbox.addWidget(self.Decreasebtn)
        self.hbox.addWidget(self.label)


        for topic, msg, t in self.bag_file.read_messages(read_topic):
            secs_list.append(msg.header.stamp.secs)
            nsecs_list.append(msg.header.stamp.nsecs)
            self.secs_dict = dict(zip(range(len(secs_list)), secs_list))
            self.nsecs_dict = dict(zip(range(len(nsecs_list)), nsecs_list))
        self.sliderMax = len(secs_list)

        self.slider.setRange(0, self.sliderMax)
        self.slider.setSingleStep(1)

        self.Viewer2D = QWidget()
        self.Viewer3D = QWidget()

        self.Viewer2D_UI()
        self.Viewer3D_UI()

        self.Viewer = QStackedWidget(self)
        self.Viewer.addWidget(self.Viewer2D)
        self.Viewer.addWidget(self.Viewer3D)

        self.hbox.addWidget(self.Viewer,0,0)

        #ros thread
        self.bagthreadFlag = True
        self.bagthread = Thread(target=self.getbagfile)
        self.bagthread.start()
        #Graph Timer 시작
        self.mytimer = QTimer()
        self.mytimer.start(10)  # 1초마다 차트 갱신 위함...
        self.mytimer.timeout.connect(self.get_data)

        self.show()

    def change(self, val):
        self.start_secs_time = self.secs_dict[val]
        self.start_nsecs_time = self.nsecs_dict[val]
        global count
        self.slider.setValue(val)
        print(val)
        count = val
        self.flag = True

    def sliderPressed(self):
        if self.Stop_flag is True:
            self.btn_Flag = False

    def sliderReleased(self):
        if self.Stop_flag is True:
            self.btn_Flag = True

    def count_base(self):
        global count
        if count <= self.sliderMax:
            count += 1
            a= str(count)
            self.label.setText(a)
            self.slider.setValue(count)
            self.creat_correct_box()
        else:
            print("err")

    def creat_correct_box(self):
        #gui에 생성된 모든 correct object들의 표시상태를 false로 변경함. (초기화)
        for i in range(len(self.correct_objs)):
            self.correct_objs[i].setVisible(False)
            self.tolerance_borders[i].setVisible(False)
            self.correct_borders[i].setVisible(False)

        # 여기서 frame 부분을 현재 frame count와 동일하게 되어야 함.
        if count in list(self.evaluation.keys()):
            for i in range(len(self.evaluation[count])):
                # 허용 오차 범위 x, y값 설정
                self.tolerance_y = self.evaluation[count][i][1] - 0.5
                self.tolerance_y_size = self.evaluation[count][i][3] + 1.0  # 허용오차 범위를 0.5로 하기 위해서는 좌표값의 이동때문에 size 값을 1로 지정해야 함
                self.tolerance_x = self.evaluation[count][i][0] - 0.5
                self.tolerance_x_size = self.evaluation[count][i][2] + 1.0

                #표시할 object만 크기 변경 후 표시모드를 true로 변경
                # 허용 오차 범위 투명도 사각형 출력
                self.correct_objs[i].setRect(self.tolerance_x, self.tolerance_y, self.tolerance_x_size, self.tolerance_y_size)
                self.correct_objs[i].setVisible(True)

                # 허용 오차 범위 테두리 출력
                self.tolerance_borders[i].setRect(self.tolerance_x, self.tolerance_y, self.tolerance_x_size, self.tolerance_y_size)
                self.tolerance_borders[i].setVisible(True)

                # 정답 테두리 출력
                self.correct_borders[i].setRect(self.evaluation[count][i][0], self.evaluation[count][i][1], self.evaluation[count][i][2], self.evaluation[count][i][3])
                self.correct_borders[i].setVisible(True)

    def btn_event(self):
        self.btn_Flag = self.btn.isChecked()
        # print(self.btn_Flag)
        if self.btn_Flag == True:
            self.btn.setText("▶")
        else:
            self.btn.setText("||")

    def DecreaseButton(self,n):
        global count
        n = count+200
        self.start_secs_time = self.secs_dict[n]
        self.start_nsecs_time = self.nsecs_dict[n]
        # print(n)
        # print(count)
        self.list1.insert(1, n)
        count = n
        del (self.list1[2])
        if self.D_flag == False:
            self.btn_Flag= False
            self.D_flag = True

    @pyqtSlot()
    def get_data(self):
        if self.pos is not None:
            self.spt.setData(pos=self.pos)  # line chart 그리기

        #object 출력
        #50개 object중 position 값이 0,0이 아닌것만 출력
        for i, obj in enumerate(self.objs):
            objpos = self.objsPos[i]
            objsize = self.objsSize[i]
            if objpos[0] == 0 and objpos[1] == 0:
                obj.setVisible(False)
            else:
                obj.setVisible(True)
                # obj.setRect((objpos[0])-(objsize[0]/2), (objpos[1])-(objsize[1]/2), objsize[0], objsize[1])
                obj.setRect(objpos[0], objpos[1], objsize[0], objsize[1])
            # print(f"{i:d} : [{objpos[0]:.2f}, {objpos[1]:.2f}], [{objsize[0]:.2f}, {objsize[1]:.2f}]")

    #ros 파일에서 velodyne_points 메시지만 불러오는 부분
    def getbagfile(self):
        while True:
            for topic, msg, t in self.bag_file.read_messages(read_topic, start_time=rospy.Time(self.start_secs_time,self.start_nsecs_time)):

                if self.bagthreadFlag is False:
                    break

                if self.btn_Flag is True:# 일시정지
                    break


                #ros_numpy 데이터 타입 문제로 class를 강제로 변경
                msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2

                #get point cloud
                pc = ros_numpy.numpify(msg)
                points = np.zeros((pc.shape[0], 4)) #point배열 초기화 1번 컬럼부터 x, y, z, intensity 저장 예정

                # for ROS and vehicle, x axis is long direction, y axis is lat direction
                # ros 데이터는 x축이 정북 방향, y축이 서쪽 방향임, 좌표계 오른손 법칙을 따름
                points[:, 0] = pc['x']
                points[:, 1] = pc['y']
                points[:, 2] = pc['z']
                # points[:, 3] = pc['intensity']
                self.resetObjPos()
                self.doYourAlgorithm(points)
                self.count_base()
                time.sleep(0.1) #빨리 볼라면 주석처리 하면됨


                if self.flag == True:#슬라이더
                    self.flag = False
                    break

                if self.D_flag == True:
                    if self.btn_Flag == True:
                        break

    def downSampling(self, points):
        # <random downsampling>
        idx = np.random.randint(len(points), size=10000)
        points = points[idx, :]

    def kdtree(self, points):
        kdt = KDTree(points, leaf_size=40)
        # cluster_list = [[0 for j in range(0, )] for i in range(3)]
        cluster_list = [0 for i in range(len(points))]
        cluster = 1
        for i in range(3):
            cluster = cluster + 1
            random_point = random.randrange(len(points))
            # dist, ind = kdt.query(points[random_point:random_point+1], k=10)
            ind = kdt.query_radius(points[random_point:random_point + 1], r=1)[0]
            # print(ind)
            for j in ind:
                cluster_list[j] = cluster
        self.clusterLabel = np.asarray(cluster_list)

    def btn_3D(self):
        self.Flag_3D = self.btn3D.isChecked()
        if self.Flag_3D is True:
            self.btn3D.setText("2D")
            self.Viewer.setCurrentIndex(1)
        else:
            self.btn3D.setText("3D")
            self.Viewer.setCurrentIndex(0)

    def input_time(self):
        s=self.line.text()
        n= int(s)-1
        global count
        self.start_secs_time = self.secs_dict[n]
        self.start_nsecs_time = self.nsecs_dict[n]
        count = n
        self.flag = True

    def input_btnPressed(self):
        if self.Stop_flag is True:
            self.btn_Flag = False

    def input_btnReleased(self):
        if self.Stop_flag is True:
            self.btn_Flag = True

    def Viewer2D_UI(self):
        layout2D = QVBoxLayout()
        layout2D.addWidget(self.canvas)
        self.Viewer2D.setLayout(layout2D)

    def Viewer3D_UI(self):
        layout3D = QVBoxLayout()
        layout3D.addWidget(self.canvas3D.native)
        self.Viewer3D.setLayout(layout3D)

    def dbscan(self, points):  # dbscan eps = 1.5, min_size = 60
        dbscan = DBSCAN(eps=0.5, min_samples=10, algorithm='ball_tree').fit(points)
        self.clusterLabel = dbscan.labels_

    def doYourAlgorithm(self, points):
        box_cnt = list()
        # Filter_ROI
        roi = {"x": [-30, 30], "y": [-20, 20], "z": [-1.5, 5.0]}  # z값 수정

        x_range = np.logical_and(points[:, 0] >= roi["x"][0], points[:, 0] <= roi["x"][1])
        y_range = np.logical_and(points[:, 1] >= roi["y"][0], points[:, 1] <= roi["y"][1])
        z_range = np.logical_and(points[:, 2] >= roi["z"][0], points[:, 2] <= roi["z"][1])

        pass_through_filter = np.where(np.logical_and(x_range, np.logical_and(y_range, z_range)) == True)[0]
        points = points[pass_through_filter, :]

        # Downsampling
        # self.downSampling(points)

        # Clustering
        if self.ALGO_FLAG == 1:
            self.kdtree(points)
        elif self.ALGO_FLAG == 2:
            self.dbscan(points)

        # Bounding Box
        for i in range(1, max(self.clusterLabel)+1):
            tempobjPos = self.objsPos[i]
            tempobjSize = self.objsSize[i]

            index = np.asarray(np.where(self.clusterLabel == i))
            # print(i, 'cluster 개수 : ', len(index[0]))
            x = np.min(points[index, 0])
            y = np.min(points[index, 1])
            x_size = np.max(points[index, 0]) - np.min(points[index, 0])  # x_max 3
            y_size = np.max(points[index, 1]) - np.min(points[index, 1])  # y_max 1.3

            # car size bounding box
            carLength = 9 # 경차 : 3.6 소형 : 4.7 화물 차량 : 9
            carHeight = 3 # 경차 : 2 소형 : 2 화물 차량 : 9
            if (x_size <= carLength+1) and (y_size <= carHeight+1):
                box_cnt.append(i)
                tempobjPos[0] = x
                tempobjPos[1] = y
                tempobjSize[0] = x_size
                tempobjSize[1] = y_size

            # 정답지 좌표 출력
            # print("%d : [%.2f, %.2f, %.2f, %.2f]" % (i, tempobjPos[0], tempobjPos[1], tempobjSize[0], tempobjSize[1]))

        # 1번 실행 시 .csv 파일명 바꾸기
        f = open('test_2.csv', 'a', encoding="utf-8-sig")
        wr = csv.writer(f)
        if count == 1:
            wr.writerow(['N frame', '차량개수', '정답차량', '미검지/오검지', '실행시간', '정확도'])
        if count in list(self.evaluation.keys()):
            l = [0 for i in range(150)]
            correct_car = 0
            car_cnt = len(self.evaluation[count])
            for i in box_cnt:
                left_x = self.objsPos[i][0]  # left down x
                left_y = self.objsPos[i][1]  # left down y
                right_x = left_x + self.objsSize[i][0]  # right up x
                right_y = left_y + self.objsSize[i][1]  # right up y
                # print(i, ':', left_x, left_y, right_x, right_y)
                for j in range(len(self.evaluation[count])):
                    left_x_correct = self.evaluation[count][j][0] - 0.5
                    left_y_correct = self.evaluation[count][j][1] - 0.5
                    right_x_correct = left_x_correct + self.evaluation[count][j][2] + 1.0
                    right_y_correct = left_y_correct + self.evaluation[count][j][3] + 1.0
                    if(left_x > left_x_correct and right_x < right_x_correct and left_y > left_y_correct and right_y < right_y_correct):
                        l[j] = l[j] + 1
            # 정답 차량 카운팅
            for i in l:
                if(i > 0):
                    correct_car = correct_car + 1
            # 도로에 챠량이 없을 경우 예외 처리
            if len(self.evaluation[count]) == 0:
                self.correct.append('')
                print('pass\n')
                data = [count, car_cnt, correct_car, '미/오', 0.00, '']
            else:
                self.correct.append(correct_car/len(self.evaluation[count]))
                print('car_cnt : %d\ncorrect_car : %d개\n정확도 : %.2f\n' % (car_cnt, correct_car, correct_car/len(self.evaluation[count])))
                data = [count, car_cnt, correct_car, '미/오', 0.00, correct_car / len(self.evaluation[count])]
            wr.writerow(data)
        f.close()

        # print(self.correct)

        # obj detection
        # 그래프의 좌표 출력을 위해 pos 데이터에 최종 points 저장
        self.pos = points
        # print(self.pos)
        # print(self.pos[0])

    def resetObjPos(self):
        for i, pos in enumerate(self.objsPos):
            # reset pos, size
            pos[0] = 0
            pos[1] = 0
            os = self.objsSize[i]
            os[0] = 0
            os[1] = 0

    def closeEvent(self, event):
        print('closed')
        self.bagthreadFlag = False

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    ex = ExMain()

    sys.exit(app.exec_())