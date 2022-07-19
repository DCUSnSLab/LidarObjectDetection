import pyqtgraph as pg
import ros_numpy
import sensor_msgs
import vispy.scene

from vispy.scene import visuals, TurntableCamera
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, Qt, QThread, QTimer
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
count = 0
class ExMain(QWidget):
    def __init__(self):
        super().__init__()

        self.ALGO_FLAG = 2  # 1 : kdtree. 2 : dbscan
        self.clusterLabel = list()

        self.getslider = None
        secs_list = []
        nsecs_list = []
        self.btn_Flag = None
        self.Stop_flag = False
        self.Flag_3D = None


        self.start_secs_time = 0
        self.start_nsecs_time = 0

        self.t = 0
        self.flag = False
        self.test = False


        self.hbox = QGridLayout()

        #3D
        self.canvas3D = vispy.scene.SceneCanvas(keys='interactive', bgcolor='#000d1a')
        self.view3D = self.canvas3D.central_widget.add_view()
        self.view3D.camera = 'arcball'
        self.view3D.camera = TurntableCamera(fov=30.0, elevation=90.0, azimuth=-90., distance=100, translate_speed=50.0)
        grid = visuals.GridLines(parent=self.view3D.scene, scale=(5, 5))

        self.scatter = visuals.Markers(edge_color=None, size=2)
        self.view3D.add(self.scatter)

        #2D
        self.canvas = pg.GraphicsLayoutWidget()
        self.view2D = self.canvas.addViewBox()
        self.view2D.setAspectLocked(True)
        self.view2D.disableAutoRange()
        self.view2D.scaleBy(s=(20, 20))
        grid = pg.GridItem()
        self.view2D.addItem(grid)
        self.spt = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='o', size=2)
        self.view2D.addItem(self.spt)

        self.setWindowTitle("realtime")

        # global position to display graph
        self.pos = None

        #object 출력용
        self.objs = list() #for display to graph

        # object 출력용 position과 size
        self.objsPos = list()
        self.objsSize = list()

        #출력용 object를 미리 생성해둠
        #생성된 object의 position값을 입력하여 그래프에 출력할 수 있도록 함
        numofobjs = 50
        for i in range(numofobjs):
            obj = pg.QtGui.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5) #obj 크기는 1m로 고정시킴
            obj.setPen(pg.mkPen('w'))
            self.view2D.addItem(obj)
            self.objs.append(obj)

            pos = [0, 0, 0] #x, y, z
            size = [0, 0, 0] #w, h, depth
            self.objsPos.append(pos)
            self.objsSize.append(size)


        #load bagfile
        test_bagfile = '/home/soju/snslab_ROS/test2.bag'
        self.bag_file = rosbag.Bag(test_bagfile)

        self.slider = None
        self.slider = QSlider(Qt.Horizontal, self)

        self.slider.sliderMoved.connect(self.change)
        self.slider.sliderPressed.connect(self.sliderPressed)
        self.slider.sliderReleased.connect(self.sliderReleased)

        self.Table = None
        self.Table = QTableWidget(self)
        table_column = ["정지", "입력", "클릭", "프레임","화면전환"]
        self.Table.setFixedSize(600,60)
        self.Table.setColumnCount(5)
        self.Table.setRowCount(1)
        self.Table.setHorizontalHeaderLabels(table_column)

        self.btn = QPushButton("||")
        self.MoveBtn = QPushButton("Move")

        self.btn3D = QPushButton("3D")
        self.btn.setCheckable(True)
        self.btn3D.setCheckable(True)

        self.label = QLabel('label',self)
        self.btn.clicked.connect(self.btn_event)
        self.btn3D.clicked.connect(self.btn_3D)
        self.MoveBtn.clicked.connect(self.input_time)
        self.MoveBtn.pressed.connect(self.input_btnPressed)
        self.MoveBtn.released.connect(self.input_btnReleased)

        self.line = QLineEdit(self)

        self.Table.setCellWidget(0,0,self.btn)
        self.Table.setCellWidget(0,1, self.line)
        self.Table.setCellWidget(0,2, self.MoveBtn)
        self.Table.setCellWidget(0,3,self.label)
        self.Table.setCellWidget(0,4,self.btn3D)


        self.hbox.addWidget(self.slider,1,0)
        self.hbox.addWidget(self.Table,2,0)

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


        self.setLayout(self.hbox)

        #ros thread
        self.bagthreadFlag = True
        self.bagthread = Thread(target=self.getbagfile)
        self.bagthread.start()
        #Graph Timer 시작
        self.mytimer = QTimer()
        self.mytimer.start(10)  # 1초마다 차트 갱신 위함...
        self.mytimer.timeout.connect(self.get_data)

        self.show()

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
            self.slider.setSliderPosition(count)
        else:
            print("err")

    def btn_event(self):
        self.btn_Flag = self.btn.isChecked()
        if self.btn_Flag == True:
            self.btn.setText("▶")
            self.start_secs_time = self.secs_dict[count]
            self.start_nsecs_time = self.nsecs_dict[count]
            self.Stop_flag = True
        else:
            self.btn.setText("||")
            self.start_secs_time = self.secs_dict[count]
            self.start_nsecs_time = self.nsecs_dict[count]
            self.Stop_flag = False

    def btn_3D(self):
        self.Flag_3D = self.btn3D.isChecked()
        if self.Flag_3D is True:
            self.btn3D.setText("2D")
            self.Viewer.setCurrentIndex(1)
        else:
            self.btn3D.setText("3D")
            self.Viewer.setCurrentIndex(0)

    #화면 전환 함수
    def Viewer2D_UI(self):
        layout2D = QVBoxLayout()
        layout2D.addWidget(self.canvas)
        self.Viewer2D.setLayout(layout2D)

    def Viewer3D_UI(self):
        layout3D = QVBoxLayout()
        layout3D.addWidget(self.canvas3D.native)
        self.Viewer3D.setLayout(layout3D)


    @pyqtSlot()
    def get_data(self):
        if self.pos is not None:
            if self.Flag_3D is True:
                self.scatter.set_data(pos=self.pos[:, :3], edge_color='white', size=1)
            else:
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
                obj.setRect((objpos[0])-(objsize[0]/2), (objpos[1])-(objsize[1]/2), objsize[0], objsize[1])
        #time.sleep(1)

    #ros 파일에서 velodyne_points 메시지만 불러오는 부분
    def getbagfile(self):
        while True:
            for topic, msg, t in self.bag_file.read_messages(read_topic, start_time=rospy.Time(self.start_secs_time,self.start_nsecs_time)):

                if self.bagthreadFlag is False:
                    break

                if self.btn_Flag is True:# 일시정지
                    break
                    # self.getbagfile

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


    def downSampling(self, points):
        # <random downsampling>
        idx = np.random.randint(len(points), size=10000)
        points = points[idx, :]

        # <voxel grid downsampling>
        # vox = points.make_voxel_grid_filter()
        # vox.set_leaf_size(0.01, 0.01, 0.01)
        # points = vox.filter()
        # print(points)

        # open3d.visualization.draw_geometries([points])
        # points.scale(1/points.get_max_bound()-points.get_min_bound())
        # voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(points, voxel_size=0.1)
        # open3d.visualization.draw_geometries([voxel_grid])

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

        # print('cluster_point : ', random_point, ', ', ind)

        # print('length : ', len(points), ', random point : ', random_point, ', random point list : ', points[random_point])
        # dist, ind = kdt.query(points[:], k=10)
        # print(points[:])

        # print('tree', kdt.get_tree_stats())
        # print('dist : ', dist, '\nind : ', ind)

        # print('count : ', kdt.query_radius(points[:1], r=0.3, count_only=True))
        # print(kdt.query_radius(points[:1], r=0.3))

    def dbscan(self, points):  # dbscan eps = 1.5, min_size = 60
        dbscan = DBSCAN(eps=1, min_samples=20, algorithm='ball_tree').fit(points)
        self.clusterLabel = dbscan.labels_
        # print('DBSCAN(', len(self.clusterLabel), ') : ', self.clusterLabel)
        # print(self.clusterLabel)
        # for i in self.clusterLabel:
        #     print(i, end='')

        # 여기부터 object detection 알고리즘 적용해 보면 됨

    def doYourAlgorithm(self, points):
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
        for i in range(1, max(self.clusterLabel) + 1):
            tempobjPos = self.objsPos[i]
            tempobjSize = self.objsSize[i]

            index = np.asarray(np.where(self.clusterLabel == i))
            # print(i, 'cluster 개수 : ', len(index[0]))
            cx = (np.max(points[index, 0]) + np.min(points[index, 0])) / 2  # x_min 1
            cy = (np.max(points[index, 1]) + np.min(points[index, 1])) / 2  # y_min 3
            x_size = np.max(points[index, 0]) - np.min(points[index, 0])  # x_max 3
            y_size = np.max(points[index, 1]) - np.min(points[index, 1])  # y_max 1.3

            # car size bounding box
            carLength = 4.7  # 경차 : 3.6 소형 : 4.7
            carHeight = 2  # 경차 : 2 소형 : 2
            if (x_size <= carLength) and (y_size <= carHeight):
                tempobjPos[0] = cx
                tempobjPos[1] = cy
                tempobjSize[0] = x_size
                tempobjSize[1] = y_size
            else:
                pass

            # print(i, 'cluster min : ', tempobjPos[0], tempobjPos[1])
            # print(i, 'cluster max : ', tempobjSize[0], tempobjSize[1])

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