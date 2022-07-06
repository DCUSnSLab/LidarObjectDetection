import pyqtgraph as pg
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

read_topic = '/velodyne_points'  # 메시지 타입
count = 1
class ExMain(QWidget):
    def __init__(self):
        super().__init__()

        self.ALGO_FLAG = 2  # 1 : kdtree. 2 : dbscan
        self.clusterLabel = list()

        self.evaluation = {60: [[-15.10, 3.83, -4.7, 1.7], [-7.29, 3.33, -4.7, 1.7], [-0.44, 2.89, -4.7, 1.7], [2.44, 1.89, 4.7, 1.7], [11.23, 1.38, 4.7, 1.7], [-7.5, -0.65, -9, 3], [8.16, -2.15, 4.7, 1.7], [16.65, -2.66, 4.7, 1.7], [18.64, 0.04, 4.7, 1.7]],
                           120: [[-19.50, 3.47, -4.7, 1.7], [-12.09, -0.71, -9, 3], [-3.10, 2.22, -4.7, 1.7], [3.56, 8.16, 4.7, 1.7], [14.53, -2.45, 4.7, 1.7], [24.40, 0.44, 4.7, 1.7], [-11.34, 3.26, -4.7, 1.7]],
                           180: [[-22.15, -0.39, -9, 3], [-14.39, 3.45, -4.7, 1.7], [3.03, 3.47, 4.7, 1.7], [26.91, -9.94, 4.7, 1.7], [23.40, -2.17, 4.7, 1.7], [-28.38, 3.36, -4.7, 1.7], [16.53, 7.38, 4.7, 1.7]],
                           240: [[-5.08, 5.73, -4.7, 1.7], [14.08, 3.04, 4.7, 1.7], [22.73, -3.85, 4.7, 1.7], [-26.33, 4.70, -4.7, 1.7]],
                           300: [[26.08, -6.58, 4.7, 1.7]],
                           360: [[-28.30, 8.01, -4.7, 1.7], [24.36, -7.29, 4.7, 1.7]],
                           420: [[8.68, -5.05, 4.7, 1.7]],
                           480: [[-7.41, -4.57, -4.7, 1.7]],
                           540: [[-9.67, 5.20, -4.7, 1.7], [-23.19, -1.76, -4.7, 1.7], [23.94, 3.90, 4.7, 1.7], [-13.34, 9.32, -9, 3]],
                           600: [[-21.80, -2.54, -4.7, 1.7], [-22.40, 10.36, -4.7, 1.7], [11.96, 5.51, 4.7, 1.7]],
                           660: [[-17.30, -3.18, -4.7, 1.7]],
                           720: [[-24.07, -4.12, -4.7, 1.7], [16.39, -6.21, 4.7, 1.7]],
                           780: [[-0.22, -4.57, 4.7, 1.7]],
                           840: [[20.69, -5.07, 4.7, 1.7], [-9.76, -3.56, -4.7, 1.7]],
                           900: [[-3.8, 4.8, -4.7, 1.7], [-0.28, -4.50, 4.7, 1.7], [-26.05, -2.97, -4.7, 1.7]],
                           960: [[-20.88, -1.92, -4.7, 1.7]],
                           1020: [],
                           1080: [],
                           1140: [],
                           1200: [[-23.48, 1.60, -4.7, 1.7], [-15.28, 4.19, -4.7, 1.7]]
                           }

        self.frame = 180

        self.getslider = None
        secs_list = []
        nsecs_list = []
        self.btn_Flag = None
        self.D_flag = False


        self.start_secs_time = 0
        self.start_nsecs_time = 0

        self.t = 0
        self.list1 = [0, 0]
        self.flag = False


        hbox = QGridLayout()
        self.canvas = pg.GraphicsLayoutWidget()
        hbox.addWidget(self.canvas)
        self.setLayout(hbox)
        #self.setGeometry(300, 100, 1000, 1000)  # x, y, width, height

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.disableAutoRange()
        self.view.scaleBy(s=(20, 20))
        grid = pg.GridItem()
        self.view.addItem(grid)
        #self.geometry().setWidth(1000)
        #self.geometry().setHeight(1000)
        self.setWindowTitle("realtime")

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
        self.collect_objs = list()
        self.collect_borders = list()
        self.tolerance_borders = list()

        # #출력용 object를 미리 생성해둠
        # #생성된 object의 position값을 입력하여 그래프에 출력할 수 있도록 함
        numofobjs = 50
        for i in range(numofobjs):
            obj = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5) #obj 크기는 1m로 고정시킴
            obj.setPen(pg.mkPen('w'))
            self.view.addItem(obj)
            self.objs.append(obj)

            pos = [0, 0, 0] #x, y, z
            size = [0, 0, 0] #w, h, depth
            self.objsPos.append(pos)
            self.objsSize.append(size)

        #pyqtgraph에 Collect Object 생성
        for i in range(numofobjs):
            collect_obj = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5) #obj 크기는 1m로 고정시킴
            collect_obj.setBrush(QColor(18, 241, 246))
            collect_obj.setOpacity(0.2)
            self.view.addItem(collect_obj)
            self.collect_objs.append(collect_obj)

            tolerance_border = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5)  # obj 크기는 1m로 고정시킴
            tolerance_border.setPen(pg.mkPen(QColor(18, 241, 246)))
            self.view.addItem(tolerance_border)
            self.tolerance_borders.append(tolerance_border)

            collect_border = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5)  # obj 크기는 1m로 고정시킴
            collect_border.setPen(pg.mkPen(QColor(255, 255, 0)))
            self.view.addItem(collect_border)
            self.collect_borders.append(collect_border)

        #load bagfile
        test_bagfile = '/home/soobin/development/dataset/UrbanRoad/2022-02-10-19-54-31.bag'
        self.bag_file = rosbag.Bag(test_bagfile)

        self.slider = None
        self.slider = QSlider(Qt.Horizontal, self)

        # self.slider.sizePolicy().horizontalPolicy(QSizePolicy.Maximum)
        self.slider.sliderMoved.connect(self.change)
        # self.slider.valueChanged(self.count_base())
        # self.count_base.valueChanged(self.slider.setValue)

        self.btn = QPushButton("||")
        self.Decreasebtn = QPushButton("->")
        self.btn.setCheckable(True)
        self.label = QLabel('label',self)
        # self.text = QLabel(self.)
        # self.text.setText(self,"aa")
        # self.btn.setCheckable(True)
        # self.btn.clicked.connect(self.getbagfile)
        self.btn.clicked.connect(self.btn_event)
        self.Decreasebtn.clicked.connect(self.DecreaseButton)
        hbox.addWidget(self.slider)
        hbox.addWidget(self.btn)
        hbox.addWidget(self.Decreasebtn)
        hbox.addWidget(self.label)


        # sliderMin = self.secs_dict
        for topic, msg, t in self.bag_file.read_messages(read_topic):
            secs_list.append(msg.header.stamp.secs)
            nsecs_list.append(msg.header.stamp.nsecs)
            self.secs_dict = dict(zip(range(len(secs_list)), secs_list))
            self.nsecs_dict = dict(zip(range(len(nsecs_list)), nsecs_list))
        self.sliderMax = len(secs_list)

        self.slider.setRange(0, self.sliderMax)
        self.slider.setSingleStep(1)


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

        self.list1.insert(1, val)
        global count
        count = val

        del (self.list1[2])
        self.flag = True

    def count_base(self):
        global count
        if count <= self.sliderMax:
            count += 1
            a= str(count)
            self.label.setText(a)
            self.slider.setValue(count)
            self.creat_collect_box()
        else:
            print("err")

    def creat_collect_box(self):
        #gui에 생성된 모든 collect object들의 표시상태를 false로 변경함. (초기화)
        for i in range(len(self.collect_objs)):
            self.collect_objs[i].setVisible(False)
            self.tolerance_borders[i].setVisible(False)
            self.collect_borders[i].setVisible(False)

        # 여기서 frame 부분을 현재 frame count와 동일하게 되어야 함.
        if count in list(self.evaluation.keys()):
            print(len(self.evaluation[count]))
            for i in range(len(self.evaluation[count])):
                print(i)
                # 허용 오차 범위 y값 설
                self.tolerance_y = self.evaluation[count][i][1] - 0.5
                self.tolerance_y_size = self.evaluation[count][i][3] + 1.0  # 허용오차 범위를 0.5로 하기 위해서는 좌표값의 이동때문에 size 값을 1로 지정해야 함

                # 허용 오차 범위 x값 설정 -> 조건 추가해야 할 것 같은 느낌인데...
                if self.evaluation[count][i][0] < 0:
                    self.tolerance_x = self.evaluation[count][i][0] + 0.5
                    self.tolerance_x_size = self.evaluation[count][i][2] - 1.0
                else:
                    self.tolerance_x = self.evaluation[count][i][0] - 0.5
                    self.tolerance_x_size = self.evaluation[count][i][2] + 1.0
                # self.tolerance_x = 0
                # self.tolerance_y = 0
                # self.tolerance_x_size = 0
                # self.tolerance_y_size = 0
                # self.evaluation[count][i][0] = 0
                # self.evaluation[count][i][1] = 0
                # self.evaluation[count][i][2] = 0
                # self.evaluation[count][i][3] = 0

                #표시할 object만 크기 변경 후 표시모드를 true로 변경
                # 허용 오차 범위 투명도 사각형 출력
                self.collect_objs[i].setRect(self.tolerance_x, self.tolerance_y, self.tolerance_x_size, self.tolerance_y_size)
                self.collect_objs[i].setVisible(True)

                # 허용 오차 범위 테두리 출력
                self.tolerance_borders[i].setRect(self.tolerance_x, self.tolerance_y, self.tolerance_x_size, self.tolerance_y_size)
                self.tolerance_borders[i].setVisible(True)

                # 정답 테두리 출력
                self.collect_borders[i].setRect(self.evaluation[count][i][0], self.evaluation[count][i][1], self.evaluation[count][i][2], self.evaluation[count][i][3])
                self.collect_borders[i].setVisible(True)


    def btn_event(self):
        self.btn_Flag = self.btn.isChecked()
        print(self.btn_Flag)
        if self.btn_Flag == True:
            self.btn.setText("▶")
        else:
            self.btn.setText("||")

    def DecreaseButton(self,n):
        global count
        n = count+200
        self.start_secs_time = self.secs_dict[n]
        self.start_nsecs_time = self.nsecs_dict[n]
        print(n)
        print(count)
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


    def dbscan(self, points):  # dbscan eps = 1.5, min_size = 60
        dbscan = DBSCAN(eps=1, min_samples=20, algorithm='ball_tree').fit(points)
        self.clusterLabel = dbscan.labels_
        # print('DBSCAN(', len(self.clusterLabel), ') : ', self.clusterLabel)
        # print(self.clusterLabel)
        # for i in self.clusterLabel:
        #     print(i, end='')

        # 여기부터 object detection 알고리즘 적용해 보면 됨

    def doYourAlgorithm(self, points):
        box_cnt = list()
        # Filter_ROI
        roi = {"x": [-30, 30], "y": [-10, 20], "z": [-1.5, 5.0]}  # z값 수정

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
            if np.max(points[index, 0]) < 0.0:
                # 오른쪽 아래
                x = np.max(points[index, 0])
                y = np.min(points[index, 1])
                x_size = -(np.max(points[index, 0]) - np.min(points[index, 0]))  # x_max 3
                y_size = np.max(points[index, 1]) - np.min(points[index, 1])  # y_max 1.3
            else:
                # 왼쪽 아래
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
        print()

        # print(box_cnt)
        # print('car_cnt : ', len(self.evaluation[frame]))
        l = list()
        n = 0

        if count in list(self.evaluation.keys()):
            for i in box_cnt:
                collect_car = 0
                left_x = self.objsPos[i][0] - (self.objsSize[i][0] / 2)  # left down x
                left_y = self.objsPos[i][1] - (self.objsSize[i][1] / 2)  # left down y
                right_x = left_x + self.objsSize[i][0]  # right up x
                right_y = left_y + self.objsSize[i][1]  # right up y
                for j in range(len(self.evaluation[count])):
                    left_x_collect = self.evaluation[count][j][0] - (self.evaluation[count][j][2] / 2)
                    left_y_collect = self.evaluation[count][j][1] - (self.evaluation[count][j][3] / 2)
                    right_x_collect = left_x_collect + self.evaluation[count][j][2]
                    right_y_collect = left_y_collect + self.evaluation[count][j][3]
                    if (left_x <= left_x_collect and right_x <= right_x_collect) and (left_y <= left_y_collect and right_y <= right_y_collect):
                        collect_car = collect_car + 1
                        l.append(collect_car)

                    # print('collect')
                    # print(left_x, left_y, right_x, right_y)
                    # print(left_x_collect, left_y_collect, right_x_collect, right_y_collect)
            #         print(j, collect_car)

            # print(l)
            # print()
            # 수정 필요
            # print('car_cnt : ', len(self.evaluation[self.frame]))
            # if(len(l) > 0):
            #     print('collect_cnt : ', max(l))
            #     print(f'result : {max(l)/len(self.evaluation[self.frame]):.2f}\n')
            # else:
            #     print('collect_cnt : 0')
            #     print('result : 0.00\n')

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