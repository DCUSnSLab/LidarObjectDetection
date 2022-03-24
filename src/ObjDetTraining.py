import geomdl as geomdl
import open3d as open3d
import pyqtgraph as pg
import ros_numpy
import sensor_msgs

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

import pcl
# import pcl_helper # https://gist.github.com/adioshun/f35919c895631314394aa1762c24334c

class ExMain(QWidget):
    def __init__(self):
        super().__init__()
        self.ALGO_FLAG = 2 # 1 : kdtree. 2 : dbscan
        self.clusterLabel = list()
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

        #출력용 object를 미리 생성해둠
        #생성된 object의 position값을 입력하여 그래프에 출력할 수 있도록 함
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


        #load bagfile
        test_bagfile = '/home/hyewon/development/dataset/UrbanRoad/2022-02-10-19-54-31.bag'
        self.bag_file = rosbag.Bag(test_bagfile)

        #ros thread
        self.bagthreadFlag = True
        self.bagthread = Thread(target=self.getbagfile)
        self.bagthread.start()
        #Graph Timer 시작
        self.mytimer = QTimer()
        self.mytimer.start(10)  # 1초마다 차트 갱신 위함...
        self.mytimer.timeout.connect(self.get_data)

        self.show()


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
                obj.setRect((objpos[0])-(objsize[0]/2), (objpos[1])-(objsize[1]/2), objsize[0], objsize[1])
        #time.sleep(1)
        #print('test')

    #ros 파일에서 velodyne_points 메시지만 불러오는 부분
    def getbagfile(self):
        read_topic = '/velodyne_points' #메시지 타입

        for topic, msg, t in self.bag_file.read_messages(read_topic):
            if self.bagthreadFlag is False:
                break
            #ros_numpy 데이터 타입 문제로 class를 강제로 변경
            msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2

            #get point cloud
            pc = ros_numpy.numpify(msg)
            points = np.zeros((pc.shape[0], 3)) #point배열 초기화 1번 컬럼부터 x, y, z, intensity 저장 예정

            # for ROS and vehicle, x axis is long direction, y axis is lat direction
            # ros 데이터는 x축이 정북 방향, y축이 서쪽 방향임, 좌표계 오른손 법칙을 따름
            points[:, 0] = pc['x']
            points[:, 1] = pc['y']
            points[:, 2] = pc['z']
            # points[:, 3] = pc['intensity']

            self.resetObjPos()
            self.doYourAlgorithm(points)

            #print(points)
            time.sleep(0.1) #빨리 볼라면 주석처리 하면됨

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
        rng = np.random.RandomState(0)
        X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
        tree = KDTree(X, leaf_size=2)
        dist, ind = tree.query(X[:1], k=5)
        print(X)
        print(X[:1])
        print('dist', dist, 'index', ind)
        # kdt = KDTree(points, leaf_size=40)
        # dist, ind = kdt.query(points[:1], k=10)
        # print('dist : ', dist, '\nind : ', ind)
        # print('count : ', kdt.query_radius(points[:1], r=0.3, count_only=True))
        # print(kdt.query_radius(points[:1], r=0.3))

    def dbscan(self, points):
        dbscan = DBSCAN(eps=1, min_samples=50, algorithm='ball_tree').fit(points)
        self.clusterLabel = dbscan.labels_
        print('DBSCAN(', len(self.clusterLabel), ') : ', self.clusterLabel)

    #여기부터 object detection 알고리즘 적용해 보면 됨
    def doYourAlgorithm(self, points):
        # Filter_ROI
        roi = {"x":[-30, 30], "y":[-10, 20], "z":[-1.5, 5.0]} # z값 수정

        x_range = np.logical_and(points[:, 0] >= roi["x"][0], points[:, 0] <= roi["x"][1])
        y_range = np.logical_and(points[:, 1] >= roi["y"][0], points[:, 1] <= roi["y"][1])
        z_range = np.logical_and(points[:, 2] >= roi["z"][0], points[:, 2] <= roi["z"][1])

        pass_through_filter = np.where(np.logical_and(x_range, np.logical_and(y_range, z_range))==True)[0]
        points = points[pass_through_filter, :]

        # Downsampling
        # self.downSampling(points)

        # Clustering
        if self.ALGO_FLAG == 1:
            self.kdtree(points)
        elif self.ALGO_FLAG == 2:
            self.dbscan(points)

        # Bounding Box
        for i in range(max(self.clusterLabel)): # dbscan
            tempobjPos = self.objsPos[i]
            tempobjSize = self.objsSize[i]

            index = np.asarray(np.where(self.clusterLabel == i))
            print(i, 'cluster 개수 : ', len(index[0]))
            tempobjPos[0] = (np.max(points[index, 0]) + np.min(points[index, 0]))/2  # x_min 1
            tempobjPos[1] = (np.max(points[index, 1]) + np.min(points[index, 1]))/2 # y_min 3
            tempobjSize[0] = np.max(points[index, 0]) - np.min(points[index, 0]) # x_max 3
            tempobjSize[1] = np.max(points[index, 1]) - np.min(points[index, 1]) # y_max 1.3
            # print(i, 'cluster min : ', tempobjPos[0], tempobjPos[1])
            # print(i, 'cluster max : ', tempobjSize[0], tempobjSize[1])

        #obj detection
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