# LidarObjectDetection
3D Lidar Object Detection and Tracking

### Requirements
- pyqtgraph
- rospy
- rosbag
- numpy
- pyqt5

### 실험용 파일
- src/ObjDetTraining.py 사용
- 현재 단일 소스코드로 pyqtgraph에 pointcloud 및 object 데이터 출력되도록 셋팅 해놓음
- 소스코드 분석은 어렵지 않을 것으로 보임
- 아래 doYourAlgorithm() 함수에서 본격적으로 수행해 보면 됨
```
#여기부터 object detection 알고리즘 적용해 보면 됨
def doYourAlgorithm(self, points):
    #downsampling

    #filter

    #obj detection

    # 그래프의 좌표 출력을 위해 pos 데이터에 최종 points 저장
    self.pos = points

    #테스트용 obj 생성, 임시로 0번째 obj에 x,y 좌표와 사이즈 입력
    tempobjPos = self.objsPos[0]
    tempobjSize = self.objsSize[0]
    tempobjPos[0] = 1
    tempobjPos[1] = 3
    tempobjSize[0] = 3
    tempobjSize[1] = 1.3
```

### 데이터셋 위치
- 아래 ros 데이터 다운로드 후 사용
- 연구실 NAS/2021학년도/자율주행/rosbag/2022-02-10 DCU ~ 혁신도시/2022-02-10-19-54-31.bag

### 참고사항
- 현재 ros 데이터를 불러와서 처음부터 자동 재생 되고 있음
- 본격적으로 실험을 하려고 하면 정지되어 있는 데이터를 이용해서 확인 하는게 더 효율적일 수 있음
- 아래 코드 부분에서 for문을 이용하여 ros 메시지를 계속 불러 오는 것을 확인할 수 있으니, 잘 컨트롤 하면 정지된 데이터를 쓸 수 있음
- 예를들어 for 반복문을 1번 실행하고 for문 맨 아래에 break를 하면 ros파일을 1번만 읽고 우리 알고리즘을 수행하고 끝날 것임
- time.sleep(0.1)을 지우면 시간 상관없이 플레이 됨
```
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
        break #여기에 break를 입력하면 ros 데이터를 한번만 처리하고 thread 종료됨
```