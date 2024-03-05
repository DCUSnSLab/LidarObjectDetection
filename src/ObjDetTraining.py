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
from sklearn.linear_model import RANSACRegressor
from scipy.ndimage import gaussian_filter
import pcl
import open3d
import pcl_ros


read_topic = '/velodyne_points'  # 메시지 타입
count = 0
class ExMain(QWidget):
    def __init__(self):
        super().__init__()

        self.clusterLabel = list()

        self.getslider = None
        secs_list = []
        nsecs_list = []
        self.btn_Flag = None
        self.Stop_flag = False
        self.Flag_3D = None
        self.Flag_ROI = None

        self.start_secs_time = 0
        self.start_nsecs_time = 0

        self.t = 0
        self.flag = False
        self.test = False


        self.hbox = QGridLayout()
        self.roi = {"x": [-30, 30], "y": [-10, 20], "z": [-0.4, 5.0]}

        #3D
        self.canvas3D = vispy.scene.SceneCanvas(keys='interactive', bgcolor='#000d1a') #SceneCanvas: 캔버스 종류, interactive: 캔버스 종료 및 전체 화면 모드로 전환(Esc, F11)
        self.view3D = self.canvas3D.central_widget.add_view() #기본 위젯 반환
        self.view3D.camera = 'arcball' #arcball: 마우스 드래그를 이용해서 뷰포트 카메라 위치 변경(보는 화면 전환)
        self.view3D.camera = TurntableCamera(fov=30.0, elevation=90.0, azimuth=-90., distance=100, translate_speed=50.0)
        #중심점에 뷰를 유지하면서 주위를 선회 / fov: 시야, elevation: 카메라 고도각(xy), azimuth: 카메라 방위각(yz), distance: 화전 지점에서 카메라까지의 거리, translate_speed: 카메라 중심점 이동 시 변환 속도에 대한 배율
        grid = visuals.GridLines(parent=self.view3D.scene, scale=(5, 5)) #그리드 표시

        self.scatter = visuals.Markers(edge_color=None, size=2) #edge_color: 외각선 색상
        self.view3D.add(self.scatter)

        #2D
        self.canvas = pg.GraphicsLayoutWidget()
        self.view2D = self.canvas.addViewBox()
        self.view2D.setAspectLocked(True) #가로 세로 비율 잠금
        self.view2D.disableAutoRange() #자동 범위 비활성화
        self.view2D.scaleBy(s=(20, 20)) #중심점(또는 시야 중심)을 기준으로 크기 조정
        grid = pg.GridItem() #그리드 표시
        self.view2D.addItem(grid) #grid 추가
        self.spt = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='o', size=2) #점(point cloud) 표시
        self.view2D.addItem(self.spt) #spt 추가

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
            self.objs.append(obj) #bbox새로고침


            pos = [0, 0, 0] #x, y, z
            size = [0, 0, 0] #w, h, depth
            self.objsPos.append(pos) #point cloud 출력(위치)
            self.objsSize.append(size) #point cloud 출력(크기)


        #load bagfile
        test_bagfile = '/home/yeon/lidar/2024-01-04-14-07-59.bag' #/home/yeon/lidar/2022-02-10-19-54-31.bag  2024-01-04-14-07-59.bag
        self.bag_file = rosbag.Bag(test_bagfile)

        self.slider = None
        self.slider = QSlider(Qt.Horizontal, self) #수평 슬라이더

        self.slider.sliderMoved.connect(self.change) #사용자가 슬라이더를 움직이면 event(self.change) 발생
        self.slider.sliderPressed.connect(self.sliderPressed) #슬라이더를 움직이기 시작할 때 발생
        self.slider.sliderReleased.connect(self.sliderReleased)

        self.Table = None
        self.Table = QTableWidget(self) #테이블 형태로 항목 배치
        table_column = ["정지", "입력", "클릭", "프레임","화면전환", "roi"]
        self.Table.setFixedSize(720,60) #테이블 크기
        self.Table.setColumnCount(6) #테이블 열 개수 지정
        self.Table.setRowCount(1) #테이블 행 개수 지정
        self.Table.setHorizontalHeaderLabels(table_column) #헤더 이름 설정

        self.btn = QPushButton("||") #푸시버튼
        self.MoveBtn = QPushButton("Move")
        self.btn3D = QPushButton("3D")
        self.roibtn = QPushButton("ON")

        self.btn.setCheckable(True) #누른 상태와 아닌 상태 구분(Ture인 경우, False인 경우는 버튼이 안먹힘)
        self.btn3D.setCheckable(True)
        self.roibtn.setCheckable(True)

        self.label = QLabel('label',self) #위젯(텍스트, 이미지 라벨 만들때 사용)
        self.btn.clicked.connect(self.btn_event)
        self.btn3D.clicked.connect(self.btn_3D)
        self.MoveBtn.clicked.connect(self.input_time) #clicked: 클릭 했을 때 시그널 발생
        self.MoveBtn.pressed.connect(self.input_btnPressed) #pressed: 버튼이 눌린상태일때 시그널 발생
        self.MoveBtn.released.connect(self.input_btnReleased) #released: 버튼에서 뗏을때 시그널 발생
        self.roibtn.clicked.connect(self.btn_roi)

        self.line = QLineEdit(self) #문자열 입력 및 수정할 수 있도록 하는 위젯

        self.Table.setCellWidget(0,0, self.btn)
        self.Table.setCellWidget(0,1, self.line)
        self.Table.setCellWidget(0,2, self.MoveBtn)
        self.Table.setCellWidget(0,3, self.label)
        self.Table.setCellWidget(0,4, self.btn3D)
        self.Table.setCellWidget(0, 5, self.roibtn)


        self.hbox.addWidget(self.slider,1,0) #(추가할 위젯, 행 번호, 열 번호) 순서대로 수직 배치
        self.hbox.addWidget(self.Table,2,0) #0: 위젯, 1: slider, 2: Table

        for topic, msg, t in self.bag_file.read_messages(read_topic): #bag 토픽 메시지 읽어오기
            secs_list.append(msg.header.stamp.secs) #msg 객체의 header 속성의 stamp 속성의 secs 값 추가 -> 타임스탬프에서 초 값 추가
            nsecs_list.append(msg.header.stamp.nsecs) #nsecs: nanosecond
            self.secs_dict = dict(zip(range(len(secs_list)), secs_list)) #secs_list값 딕셔너리 형태로 저장
            self.nsecs_dict = dict(zip(range(len(nsecs_list)), nsecs_list))
        self.sliderMax = len(secs_list)

        self.slider.setRange(0, self.sliderMax) #slider 범위 설정(0 ~ 타임스탬프 초 값)
        self.slider.setSingleStep(1) #조절 가능한 최소 단위 설정

        self.Viewer2D = QWidget() #2D 캔버스 설정
        self.Viewer3D = QWidget() #3D 캔버스 설정

        self.Viewer2D_UI()
        self.Viewer3D_UI()

        self.Viewer = QStackedWidget(self) #각 페이지를 전환해서 볼 수 있도록 하는 클래스(2D/3D 전환)
        self.Viewer.addWidget(self.Viewer2D)
        self.Viewer.addWidget(self.Viewer3D)

        self.hbox.addWidget(self.Viewer,0,0)


        self.setLayout(self.hbox) #위젯, slider, Table 지정 위치 반영

        #ros thread (쓰레드를 사용해서 속도 증가)
        self.bagthreadFlag = True #쓰레드 사용(T/F)
        self.bagthread = Thread(target=self.getbagfile)
        self.bagthread.start()
        #Graph Timer 시작
        self.mytimer = QTimer() #시간 경과 체크
        self.mytimer.start(10)  #해당 파라미터의 시간이 지난 후부터 시간 체크, 단위: ms / 1초마다 차트 갱신 위함
        self.mytimer.timeout.connect(self.get_data) #시간 간격마다 어떤 함수를 실행할지 결정

        self.show()

    def input_time(self):
        s=self.line.text() #입력받은 숫자 가져오기
        n= int(s)-1 #입력받은 숫자(입력받은 숫자 뒤의 프레임부터 나오기때문에 -1 필요)
        global count
        self.start_secs_time = self.secs_dict[n] #secs_list값에서 n값 저장 (147)
        self.start_nsecs_time = self.nsecs_dict[n]
        count = n
        self.flag = True #해당 초의 point cloud 반영

    def input_btnPressed(self): #클릭 버튼 누를 때
        if self.Stop_flag is True: #Stop_flag가 True면(False로 지정되어 있음) btn_Flag False
            self.btn_Flag = False

    def input_btnReleased(self): #클릭 버튼 뗏을 때
        if self.Stop_flag is True:
            self.btn_Flag = True

    def change(self, val): #slider가 움직였을 떼
        self.start_secs_time = self.secs_dict[val] #[key], ex: secs_dict[3]: 3번키의 값 리턴
        self.start_nsecs_time = self.nsecs_dict[val]
        global count
        self.slider.setValue(val) #slider값(val) 조정
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
            self.slider.setSliderPosition(count) #count 기반으로 동작
        else:
            print("err")

    def btn_event(self): #클릭했을 때
        self.btn_Flag = self.btn.isChecked() #상태 확인(T/F)
        if self.btn_Flag == True:
            self.btn.setText("▶")
            self.start_secs_time = self.secs_dict[count]
            self.start_nsecs_time = self.nsecs_dict[count]
            self.Stop_flag = True #멈춘 상태 유지
        else:
            self.btn.setText("||")
            self.start_secs_time = self.secs_dict[count]
            self.start_nsecs_time = self.nsecs_dict[count]
            self.Stop_flag = False #프레임 재생

    def btn_3D(self): #3D 화면 전환
        self.Flag_3D = self.btn3D.isChecked()
        if self.Flag_3D is True:
            self.btn3D.setText("2D") #2D 화면 전환 버튼
            self.Viewer.setCurrentIndex(1) #인덱스 값이나 위젯명을 사용해서 현재 페이지 변경 / 1 -> (인덱스)
        else:
            self.btn3D.setText("3D")
            self.Viewer.setCurrentIndex(0) #0: 2D, 1: 3D

    #화면 전환 함수
    def Viewer2D_UI(self):
        layout2D = QVBoxLayout() #위젯 수직 나열
        layout2D.addWidget(self.canvas)
        self.Viewer2D.setLayout(layout2D)

    def Viewer3D_UI(self):
        layout3D = QVBoxLayout()
        layout3D.addWidget(self.canvas3D.native) #native: 2D -> 3D 마우스를 이용한 화면전환때문으로 예상
        self.Viewer3D.setLayout(layout3D)

    def btn_roi(self):
        self.Flag_ROI = self.roibtn.isChecked()
        if self.Flag_ROI is True:
            self.roibtn.setText("OFF")
            self.roi = {"x": [-5, 7], "y": [-3.5, 4.5], "z": [-0.4, 5.0]}
        else:
            self.roibtn.setText("ON")
            self.roi = {"x": [-30, 30], "y": [-10, 20], "z": [-0.4, 5.0]}

    @pyqtSlot() #pyqt5.qtcore 모듈에 정의 된 데코레이터 / 데코레이터: 함수나 메서드에 적동되어, 해당 함수나 메서드의 기능을 확장하거나 변경하는 역할
    def get_data(self): #https://wikidocs.net/38522
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
                points = np.zeros((pc.shape[0], 3)) #point배열 초기화 1번 컬럼부터 x, y, z, intensity 저장 예정

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

    def clustering(self, points):
        dbscan = DBSCAN(eps=0.3, min_samples=20).fit(points)
        self.clusterLabel = dbscan.labels_  # 각 데이터가 어떤 클러스터에 속하는지

    def doYourAlgorithm(self, points):
        # filter_roi
        # 설정하지 않으면 도로 밖 구간에도 클러스터가 생성됨. 필요한 공간에만 생성될 수 있도록 roi 설정
        # roi: region of interest. 관심 영역 처리
        roi = self.roi
        # roi = {"x": [-5, 7], "y": [-3.5, 4.5], "z": [-0.4, 5.0]}  # z값 수정 #-1.5 (단위:m)
        # roi = {"x": [-30, 30], "y": [-10, 20], "z": [-1.5, 5.0]}

        x_range = np.logical_and(points[:, 0] >= roi["x"][0], points[:, 0] <= roi["x"][1])
        y_range = np.logical_and(points[:, 1] >= roi["y"][0], points[:, 1] <= roi["y"][1])
        z_range = np.logical_and(points[:, 2] >= roi["z"][0], points[:, 2] <= roi["z"][1])
        # np.logical_and: 모든 조건을 충족할 경우 True, 아닐 경우 False
        # roi영역안에 있는 값들 저장

        pass_through_filter = np.where(np.logical_and(x_range, np.logical_and(y_range, z_range)) == True)[0]
        # np.where: 조건 만족하는 위치 인덱스 찾기
        # 조건 두개만 가능하기 때문에 안에 logical_and를 한 번 더 사용
        points = points[pass_through_filter, :]

        # downsampling
        idx = np.random.randint(len(points), size=1000)  # points길이부터 size이하 범위의 정수 난수 생성
        points = points[idx, :]  # [:]: z처음부터 끝까지

        # clustering
        self.clustering(points)

        for i in range(1, max(self.clusterLabel) + 1):  # i: 클러스터 개수(클러스터 찾기 위함)
            tempobjPos = self.objsPos[i]
            tempobjSize = self.objsSize[i]

            index = np.asarray(np.where(self.clusterLabel == i))
            # asarry: 복사본 반환(서로 독립된 개체. 아무 영향을 주자 x), array: 참조본 반환(a=b, a를 고치면 b도 같이 수정)
            # whrere(): 조건에 맞는 입력 array 값의 인덱스 값을 알려줌(해당 클러스터의 값만 받기 위함)
            # print(i, 'cluster 개수 : ', index)

            # bounding box
            cx = (np.max(points[index, 0]) + np.min(points[index, 0])) / 2
            cy = (np.max(points[index, 1]) + np.min(points[index, 1])) / 2
            x_size = np.max(points[index, 0]) - np.min(points[index, 0])
            y_size = np.max(points[index, 1]) - np.min(points[index, 1])

            targetLength = 4.7  # 경차 : 3.6 소형 : 4.7 > 사람
            targetHeight = 1.5  # 경차 : 2 소형 : 2 > 사람 (바닥제거를 위해 1.1만큼 없앴으므로 약 1로 대체)
            if (x_size <= targetLength) and (y_size <= targetHeight):
                tempobjPos[0] = cx
                tempobjPos[1] = cy
                tempobjSize[0] = x_size
                tempobjSize[1] = y_size
            else:
                pass

        # 그래프의 좌표 출력을 위해 pos 데이터에 최종 points 저장
        self.pos = points

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