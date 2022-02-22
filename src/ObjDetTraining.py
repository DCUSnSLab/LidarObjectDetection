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

# pcl 라이브러리 import 문제
# import pcl
# import pcl_helper # https://gist.github.com/adioshun/f35919c895631314394aa1762c24334c

class ExMain(QWidget):
    def __init__(self):
        super().__init__()

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
            points = np.zeros((pc.shape[0], 4)) #point배열 초기화 1번 컬럼부터 x, y, z, intensity 저장 예정

            # for ROS and vehicle, x axis is long direction, y axis is lat direction
            # ros 데이터는 x축이 정북 방향, y축이 서쪽 방향임, 좌표계 오른손 법칙을 따름
            points[:, 0] = pc['x']
            points[:, 1] = pc['y']
            points[:, 2] = pc['z']
            points[:, 3] = pc['intensity']

            self.resetObjPos()
            self.doYourAlgorithm(points)

            #print(points)
            time.sleep(0.1) #빨리 볼라면 주석처리 하면됨

    #여기부터 object detection 알고리즘 적용해 보면 됨
    def doYourAlgorithm(self, points):
        #downsampling
        # pointcloud random downsampling
        idx = np.random.randint(len(points), size=7000)
        points = points[idx, :]

        # voxel grid downsampling
        # vox = points.make_voxel_grid_filter()
        # vox.set_leaf_size(0.01, 0.01, 0.01)
        # points = vox.filter()
        # print(points)

        #filter
        # points = pcl_helper.ros_to_pcl(points) # pcl_helper import 안됨
        # passthrough = points.make_passthrough_filter()
        # x축 영역 설정 부분
        # passthrough.set_filter_field_name('x')
        # passthrough.set_filter_limits(1.0, 20.0) # min, max 값 수정 필요
        # points = passthrough.filter()
        # points = pcl_helper.pcl_to_tos(points)


        # dic = {"x":[-10, 25],
        #        "y":[-30, 30],
        #        "z":[0.0, 10.0]}
        #
        # x_range = np.logical_and(points[:, 0] >= dic["x"][0], points[:, 0] <= dic["x"][1])
        # y_range = np.logical_and(points[:, 1] >= dic["y"][0], points[:, 1] <= dic["y"][1])
        # z_range = np.logical_and(points[:, 2] >= dic["z"][0], points[:, 2] <= dic["z"][1])
        #
        # pass_through_filter = np.logical_and(x_range, np.logical_and(y_range, z_range))
        # print(pass_through_filter)
        # open3d.utility.Vector3dVector(pass_through_filter)

        #obj detection

        # 그래프의 좌표 출력을 위해 pos 데이터에 최종 points 저장
        self.pos = points
        # print(self.pos)
        # print(self.pos[0])

        #테스트용 obj 생성, 임시로 0번째 obj에 x,y 좌표와 사이즈 입력
        tempobjPos = self.objsPos[0]
        tempobjSize = self.objsSize[0]
        tempobjPos[0] = 1
        tempobjPos[1] = 3
        tempobjSize[0] = 3
        tempobjSize[1] = 1.3

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