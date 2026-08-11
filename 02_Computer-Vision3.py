# -*- coding: utf-8 -*-
"""
Part 3: 실시간 카메라 피드 / 객체 검출 / 객체 추적 / MediaPipe 신체 인식 / Section Project (J~M, Project)
원본: 02_Computer-Vision.ipynb (cell 213~279)
"""

# %% [markdown]
# ### J. 실시간 카메라 피드

# %% [markdown]
# #### $1)$ Jetson CSI 카메라 연결 및 동작 확인

# %% [markdown]
# 먼저 시스템에 인식된 카메라 장치 파일(/dev/video0, /dev/video1 등)을 확인합니다.

# %% [markdown]
# ```bash
# ls /dev/video*
# ```

# %% [markdown]
# 만약 `/dev/video*` 장치가 존재하지 않는다면, Jetson에서 카메라 인터페이스 설정이 활성화되어 있는지 확인해야 합니다. 아래 도구를 실행하여 CSI 카메라 설정을 활성화합니다.

# %% [markdown]
# ```bash
# sudo /opt/nvidia/jetson-io/jetson-io.py
# ```

# %% [markdown]
# 활성화가 되었다면 재부팅 후 카메라 장치를 다시 확인합니다.

# %% [markdown]
# ```bash
# reboot
# ls /dev/video*
# ```

# %% [markdown]
# 터미널에 `/dev/video0` 혹은 `/dev/video1`이 출력된다면 정상적으로 카메라 장치가 인식이 되었다는 의미입니다.
#
# 실시간 카메라 피드를 확인해봅시다.

# %% [markdown]
# ```bash
# gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvvidconv ! autovideosink
# ```

# %% [markdown]
# 에러가 발생하며 카메라 영상 출력이 되지 않는다면 아래 명령어를 실행하고 다시 카메라 피드를 확인합시다.

# %% [markdown]
# ```bash
# sudo systemctl restart nvargus-daemon
# ```

# %% [markdown]
# 카메라 영상이 확인이 된다면 다음으로 넘어가도록 합시다.

# %% [markdown]
# #### $2)$ 일반 Webcam의 실시간 카메라 피드 처리 방식

# %% [markdown]
# #### `cv2.VideoCapture()`:

# %% [markdown]
# ```python
# import cv2
#
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
#     cv2.imshow("VideoCapture", frame)
#
# cap.release()
# cv2.destroyAllWindows()
# ```

# %% [markdown]
# #### `imutils.video.VideoStream()`:

# %% [markdown]
# ```python
# import cv2
#
# import time
# from imutils.video import VideoStream
#
#
# vs = VideoStream(src=0).start()
# time.sleep(1.0)
#
# while True:
#     frame = vs.read()
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
#     cv2.imshow("VideoStream", frame)
#
# vs.stop()
# cv2.destroyAllWindows()
# ```

# %% [markdown]
# #### $3)$ Jetson CSI 카메라 GStreamer 파이프라인 테스트

# %% [markdown]
# #### `GStreamer`:

# %% [markdown]
# 우선, 카메라가 지원하는 포멧을 확인합니다.

# %% [markdown]
# ```bash
# sudo apt install v4l-utils
# v4l2-ctl --list-formats-ext -d /dev/video0
# ```

# %% [markdown]
# 일반적인 USB 카메라의 경우 출력은 다음과 같이 나타납니다.

# %% [markdown]
# ```text
# MJPG
#     1280x720
#     30 fps
# ```
#
# 또는
#
# ```text
# YUYV
#     640x480
#     30 fps
# ```

# %% [markdown]
# 출력에 따라 사용할 파이프라인을 선택합니다.

# %% [markdown]
# 카메라가 MJPEG를 지원할 경우:

# %% [markdown]
# ```python
# pipeline = (
#     "v4l2src device=/dev/video0 ! "
#     "image/jpeg,width=1280,height=720,framerate=30/1 ! "
#     "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
#     "queue leaky=downstream max-size-buffers=1 "
#     "max-size-bytes=0 max-size-time=0 ! "
#     "appsink drop=true max-buffers=1 sync=false"
# )
# ```

# %% [markdown]
# 카메라가 YUYV를 지원할 경우:

# %% [markdown]
# ```python
# pipeline = (
#     "v4l2src device=/dev/video0 ! "
#     "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
#     "videoconvert ! video/x-raw,format=BGR ! "
#     "queue leaky=downstream max-size-buffers=1 "
#     "max-size-bytes=0 max-size-time=0 ! "
#     "appsink drop=true max-buffers=1 sync=false"
# )
# ```

# %% [markdown]
# 하지만 Jetson의 CSI 카메라는 일반 USB 카메라와 동작 방식이 다릅니다.
#
# CSI 카메라는 센서에서 출력되는 원본 Bayer 데이터를 ISP(Image Signal Processor)를 통해 변환해야 하며, 일반적으로 다음과 같은 형태로 표시됩니다.

# %% [markdown]
# ```text
# 'RG10' (10-bit Bayer RGRG/GBGB)
#     Size: Discrete 3280x2464
#     Size: Discrete 1920x1080
#     Size: Discrete 1640x1232
#     Size: Discrete 1280x720
# ```

# %% [markdown]
# 이 경우 MJPEG 또는 YUYV 파이프라인을 사용하는 것이 아니라, NVIDIA Argus 카메라 인터페이스를 사용하는 `nvarguscamerasrc`를 사용해야 합니다.
#
# `nvarguscamerasrc`는 Jetson의 ISP를 이용하여 Bayer 센서 데이터를 자동으로 처리하고, OpenCV에서 사용할 수 있는 BGR 이미지로 변환합니다.

# %% [markdown]
# ```python
# pipeline = (
#     "nvarguscamerasrc sensor-id=0 ! "
#     "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
#     "nvvidconv ! "
#     "video/x-raw, format=BGRx ! "
#     "videoconvert ! "
#     "video/x-raw, format=BGR ! "
#     "appsink"
# )
#
# # pipeline = (
# #     "nvarguscamerasrc sensor-id=0 ! "
# #     "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
# #     "nvvidconv ! "
# #     "video/x-raw, format=BGRx ! "
# #     "videoconvert ! "
# #     "video/x-raw, format=BGR ! "
# #     "queue leaky=downstream max-size-buffers=1 ! "
# #     "appsink drop=true max-buffers=1 sync=false"
# # )
# ```

# %% [markdown]
# pipeline 선택 후 코드:

# %% [markdown]
# ```python
# cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
#     cv2.imshow("VideoCapture with GStreamer", frame)
#
# cap.release()
# cv2.destroyAllWindows()
# ```

# %% [markdown]
# ---

# %% [markdown]
# ### K. 실시간 객체 검출

# %% [markdown]
# 이제 앞선 실습에서 Color Segmentation을 활용해 이미지 속 원하는 색상 객체를 검출한 것과 동일하게 실시간 카메라 프레임 속 객체를 검출해봅시다.
#
# 기존과 동일한 방식을 사용하지만, 이제는 `while`문을 사용하여 매 프레임을 처리하여 출력합니다.

# %% [markdown]
# ```python
# # TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
#
# # 초록 LAB lower/upper range
# green_lower = np.array([30, 60, 90], dtype=np.uint8)
# green_upper = np.array([230, 115, 180], dtype=np.uint8)
#
# # TODO: 각 프레임 이미지를 전처리
# # TODO: Color Segmentation으로 Binary Mask 생성
# # TODO: Morphological Operation을 적용하여 객체 영역 추출
# # TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay
# ```

# %% [markdown]
# ---

# %% [markdown]
# ### L. 실시간 객체 추적

# %% [markdown]
# Kalman Filter (칼만 필터)란?
# * 노이즈가 포함된 측정값으로부터 실제 상태(위치, 속도 등)를 추정하는 알고리즘
# * 예측(Prediction)과 보정(Update) 과정을 반복하여 추정값을 지속적으로 개선
# * 센서 오차나 일시적인 측정 실패가 있어도 안정적인 추적 가능
# * 로봇 제어, 자율주행, 객체 추적 등 다양한 실시간 시스템에 활용

# %% [markdown]
# 한번 직접 Kalman Filter 함수를 만들어봅시다.
#
# 우선 역행렬 (inverse matrix)를 만들기 위해 NumPy의 `linalg.inv`를 import 합니다.
#
# Inverse는 `inv(_)`, transpose는 `_.transpose()`, dot product는 `_.dot(_)` 함수를 사용합니다.

# %% [markdown]
# ```python
# from numpy.linalg import inv
# ```

# %% [markdown]
# Kalman Filter 함수 및 변수 정의:

# %% [markdown]
# ```python
# def KalmanFilter(mu_prev, sigma_prev, z):
#     mu_bar = A_t.dot(mu_prev)
#     sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
#     if z is None:
#         return mu_bar, sigma_bar
#     else:
#         K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
#         mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
#         sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
#         return mu, sigma
#
#
# # Kalman filter 변수 정의
# A_t = np.array([[1, 1], [0, 1]])
# G = np.array([[0.5], [1]])
# R_t = G.dot(G.transpose())
# C_t = np.array([[1, 0]])
# Q_t = np.array([[1]])
# mu_t = np.array([[0, 0], [0, 0]])
# sigma_t = np.array([[0, 0], [0, 0]])
# ```

# %% [markdown]
# Kalman Filter 함수를 적용해봅시다.

# %% [markdown]
# ```python
# # TODO: Kalman Filter를 활용하여 객체 추적
#
# # TODO: 탐지된 객체의 중심과 반지름을 구해 프레임 위에 overlay
# # TODO: 예측된 객체의 중심과 반지름을 구해 프레임 위에 다른 색으로 overlay
#
# contour_lst, _ = cv2.findContours(...)
#
# # 객체 최초 검출 여부 확인용 boolean
# found = False
#
# while True:
#     frame = ...
#
#     if len(contour_lst) > 0:
#         # 가장 큰 contour 선택
#         contour = ...
#         # 최소 외접원 반지름
#         radius = ...
#         # 무게중심
#         center = ...
#
#         # 검출된 객체에 파란 원 overlay
#         cv2.circle(frame, center, int(radius), ...)
#         cv2.circle(frame, center, ...)
#
#         # 객체 최초 검출
#         if not found:
#             found = True
#
#     # 최초 검출 이후 Kalman Filter 적용
#     # 측정값 사용 (visible) -> Prediction & Update
#     if found and (len(contour_lst) > 0):
#         mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
#         x_bel, y_bel = mu_t[0][0], mu_t[0][1]
#
#     # 측정값 미사용 (occluded) -> Prediction
#     elif found and (len(contour_lst) <= 0):
#         mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
#         x_bel, y_bel = mu_t[0][0], mu_t[0][1]
#
#     # 예측한 객체 위치에 노란 원 overlay
#     cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius),...)
#     cv2.circle(frame, (int(x_bel), int(y_bel)), ...)
# ```

# %% [markdown]
# 직접 정의한 Kalman Filter 대신 OpenCV의 `cv2.KalmanFilter` 함수를 사용하여 더욱 부드러운 예측이 가능합니다.

# %% [markdown]
# ```python
# # TODO: OpenCV의 Kalman Filter 함수를 활용하여 객체 추적
#
# # --------------- OpenCV Kalman Filter 설정 ---------------
# # 상태 벡터: [x, y, vx, vy]
# # 측정 벡터: [x, y]
# kalman = cv2.KalmanFilter(4, 2)
#
# # Transition Matrix (상태 전이 행렬)
# kalman.transitionMatrix = np.array(
#     [
#         [1, 0, 1, 0],
#         [0, 1, 0, 1],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ],
#     dtype=np.float32,
# )
#
# # Measurement Matrix (측정 행렬)
# # 측정값으로부터 x, y만 관측
# kalman.measurementMatrix = np.array(
#     [
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#     ],
#     dtype=np.float32,
# )
#
# # Process Noise Covariance Matrix (프로세스 노이즈 공분산)
# # 값이 클수록 모델 예측보다 측정값 변화에 더 유연하게 반응
# kalman.processNoiseCov = np.array(
#     [
#         [1e-2, 0, 0, 0],
#         [0, 1e-2, 0, 0],
#         [0, 0, 5e-2, 0],
#         [0, 0, 0, 5e-2],
#     ],
#     dtype=np.float32,
# )
#
# # Measurement Noise Covariance Matrix (측정 노이즈 공분산)
# # 값이 클수록 측정값을 덜 신뢰하고 예측값을 더 신뢰
# kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
#
# # Posteriori Error Covariance Matrix (초기 추정 오차 공분산)
# kalman.errorCovPost = np.eye(4, dtype=np.float32)
#
# # -------------------------------------------------------
#
# contour_lst, _ = cv2.findContours(...)
#
# # 객체 최초 검출 여부 확인용 boolean
# found = False
#
# while True:
#     frame = ...
#
#     if len(contour_lst) > 0:
#         # 가장 큰 contour 선택
#         contour = ...
#         # 최소 외접원 반지름
#         radius = ...
#         # 무게중심
#         center = ...
#
#         # 검출된 객체에 파란 원 overlay
#         cv2.circle(frame, center, int(radius), ...)
#         cv2.circle(frame, center, ...)
#
#         # 객체 최초 검출 (최초 a priori state 저장)
#         if not found:
#             kalman.statePre = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
#             found = True
#
#     # 최초 검출 이후 Kalman Filter 적용
#         # 측정값 사용 (visible) -> Prediction & Update
#         if found and (len(contour_lst) > 0):
#             predicted_state = kalman.predict()   # prediction
#             corrected_state = kalman.correct(z)  # update
#             x_bel = corrected_state[0, 0]
#             y_bel = corrected_state[1, 0]
#
#         # 측정값 미사용 (occluded) -> Prediction
#         elif found and (len(contour_lst) <= 0):
#             predicted_state = kalman.predict()   # prediction (update X)
#             x_bel = predicted_state[0, 0]
#             y_bel = predicted_state[1, 0]
#
#     # 예측한 객체 위치에 노란 원 overlay
#     cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), ...)
#     cv2.circle(frame, (int(x_bel), int(y_bel)), ...)
# ```

# %% [markdown]
# 이로써 객체를 탐지와 함께 추적 및 예측도 할 수 있게 되었습니다.
#
# Kalman Filter를 사용하여, 선형적으로 객체의 위치를 예측하여 조명 오염, 센서 불량, 객체 가려짐 등 상황에서도 끊임 없는 추적을 할 수 있으며, 몇 초 뒤의 객체 위치도 대략적으로 예측이 가능합니다.

# %% [markdown]
# ---

# %% [markdown]
# ### M. MediaPipe 실시간 신체 인식

# %% [markdown]
# 신체(자세, 손, 얼굴) 인식/추적을 하기 위한 라이브러리인 MediaPipe를 우선 설치합니다.

# %% [markdown]
# ```bash
# pip install --no-deps mediapipe
# pip install absl-py flatbuffers sounddevice
# ```

# %% [markdown]
# `MediaPipe`에서 사용하는 모델도 설치해줍시다.

# %% [markdown]
# ```bash
# wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O src/models/MediaPipe/hand_landmarker.task
# wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -o src/models/MediaPipe/pose_landmarker_full.task
# ```

# %% [markdown]
# 이제 아래 코드를 한번 새 .py 파일에 복사하여 시도해봅시다.

# %% [markdown]
# ```python
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
#
#
# base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
# options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
# hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
# connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보
# finger_tips = (4, 8, 12, 16, 20)                                                                # 손가락 끝 landmark의 index
#
# pipeline = (
#     "nvarguscamerasrc sensor-id=0 ! "
#     "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
#     "nvvidconv ! "
#     "video/x-raw, format=BGRx ! "
#     "videoconvert ! "
#     "video/x-raw, format=BGR ! "
#     "queue leaky=downstream max-size-buffers=1 ! "
#     "appsink drop=true max-buffers=1 sync=false"
# )
#
# cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
#     # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
#     frame = cv2.flip(frame, 0)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # 프레임 내 손 탐지
#     result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
#
#     # 화면 좌측 상단에 텍스트 생성 (손 개수, 왼손/오른손/양손 여부)
#     labels = ["Left" if h[0].category_name == "Right" else "Right" for h in result.handedness]
#     cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
#     cv2.putText(frame, " / ".join(labels), (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
#
#     # 탐지 결과의 각 손마다 선과 점 그리기
#     for hand in result.hand_landmarks:
#         h, w = frame.shape[:2]  # 프레임 높이와 너비
#         points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표
#
#         # landmark를 연결하는 선 (skeleton) 그리기
#         for c in connections:
#             cv2.line(frame, points[c.start], points[c.end], (0,255,0), 2)
#
#         # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
#         for i, point in enumerate(points):
#             color = (0,0,255) if i in finger_tips else (255,0,0)
#             cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)
#
#     cv2.imshow("MediaPipe Hand Detection", frame)
#
# hand_detector.close()
# cap.release()
# cv2.destroyAllWindows()
# ```

# %% [markdown]
# Rule-based (규칙 기반) 시스템으로 펼쳐진 손가락 개수 계산:

# %% [markdown]
# ```python
# import cv2
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
#
#
# # 중간 점 기준으로 각도 계산하는 함수
# def calculate_angle(p1, p2, p3):
#     vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
#     vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
#
#     magnitude1 = np.linalg.norm(vector1)
#     magnitude2 = np.linalg.norm(vector2)
#
#     if magnitude1 == 0 or magnitude2 == 0:
#         return 0.0
#
#     cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
#     cosine = np.clip(cosine, -1.0, 1.0)
#
#     return np.degrees(np.arccos(cosine))
#
#
# base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
# options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
# hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
# connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보
#
# finger_tips = (4, 8, 12, 16, 20)  # 손가락 끝 landmark의 index
# angle_threshold = 160             # 손가락 펼쳐짐 여부 판단 각도 임계값
#
# # 각 손가락의 각도를 계산할 landmark index
# finger_angle_points = (
#     (1, 2, 3),      # 엄지: 2번 중심
#     (5, 6, 7),      # 검지: 6번 중심
#     (9, 10, 11),    # 중지: 10번 중심
#     (13, 14, 15),   # 약지: 14번 중심
#     (17, 18, 19),   # 소지: 18번 중심
# )
#
# pipeline = (
#     "nvarguscamerasrc sensor-id=0 ! "
#     "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
#     "nvvidconv ! "
#     "video/x-raw, format=BGRx ! "
#     "videoconvert ! "
#     "video/x-raw, format=BGR ! "
#     "queue leaky=downstream max-size-buffers=1 ! "
#     "appsink drop=true max-buffers=1 sync=false"
# )
#
# cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
#     # 프레임 높이와 너비
#     h, w = frame.shape[:2]
#
#     # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
#     frame = cv2.flip(frame, 0)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # 프레임 내 손 탐지
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#     result = hand_detector.detect(mp_image)
#
#     # 좌우 반전된 화면을 기준으로 왼손과 오른손 정보 변경
#     labels = ["Left" if handedness[0].category_name == "Right" else "Right" for handedness in result.handedness]
#
#     # 감지된 모든 손에서 펼쳐진 손가락 개수 계산
#     total_finger_count = 0
#     for hand in result.hand_landmarks:
#         for point1_idx, point2_idx, point3_idx in finger_angle_points:
#             angle = calculate_angle(
#                 hand[point1_idx],
#                 hand[point2_idx],
#                 hand[point3_idx],
#             )
#
#             if angle >= angle_threshold:
#                 total_finger_count += 1
#
#     # 화면 좌측 상단에 손 개수와 펼친 손가락 개수 표시
#     cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
#     cv2.putText(frame, f"Fingers: {total_finger_count}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
#
#     # 화면 우측 상단에 왼손/오른손/양손 여부 표시
#     handedness_text = " / ".join(labels)
#     text_size = cv2.getTextSize(handedness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
#     text_x = w - text_size[0] - 20
#     cv2.putText(frame, handedness_text, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
#
#     # 탐지 결과의 각 손마다 선과 점 그리기
#     for hand in result.hand_landmarks:
#         h, w = frame.shape[:2]  # 프레임 높이와 너비
#         points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표
#
#         # landmark를 연결하는 선 (skeleton) 그리기
#         for c in connections:
#             cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)
#
#         # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
#         for i, point in enumerate(points):
#             color = (0, 0, 255) if i in finger_tips else (255, 0, 0)
#             cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)
#
#     cv2.imshow("MediaPipe Hand Detection", frame)
#
# hand_detector.close()
# cap.release()
# cv2.destroyAllWindows()
# ```

# %% [markdown]
# 위 코드를 실행하여 신체 인식 및 규칙기반 시스템의 작동 원리를 이해했다면, 이제 MediaPipe를 활용해 본격적인 신체 인식 기능을 구현할 준비가 완료되었습니다.
#
# 이것으로 **"Computer Vision 기초"** 섹션을 마무리합니다.

# %% [markdown]
# ---

# %% [markdown]
# ## <center>< Section Project ></center>
#
# 본 섹션에서 배운 내용을 토대로 프로젝트를 진행합니다.<br>
# 모든 기술 (이미지 조정, 색공간 변환, Image Enhancement, 필터링, Color Segmentation, Contour Detection, 객체 추적, 신체 인식)을 활용하여, 자유롭게 객체를 인식하고 추적하며 신체와의 관계성을 통해 결과를 텍스트 혹은 이미지 overlay로 표현합시다 (결과 표현은 창의적으로 다른 방식으로도 허용).

# %%
# TODO: Section 2 "Computer Vision" Project

# %% [markdown]
# ---
# ---

# %% [markdown]
# <br><br><div style="text-align: right; color: gray; font-style: italic;">
# © 2026, 김규래 (Kyu Rae Kim), All rights reserved.&emsp;<br><br>
# This material is provided solely for the intended instructional purpose.&emsp;<br>
# Redistribution, reproduction, modification, adaptation, or reuse of this material in any form without prior written permission from the copyright holder is prohibited.&emsp;
# </div>
