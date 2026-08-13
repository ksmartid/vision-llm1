# -*- coding: utf-8 -*-
"""
Part J-2: 실시간 카메라 피드 - 일반 Webcam의 실시간 카메라 피드 처리 방식
원본: 02_Computer-Vision.ipynb (cell 226~230)
"""


# %% [markdown]
# #### $2)$ 일반 Webcam의 실시간 카메라 피드 처리 방식

# %% [markdown]
# #### 참고: 이 실습 장비에는 USB 웹캠 대신 Jetson CSI 카메라가 연결되어 있습니다.
#
# `cv2.VideoCapture(0)`처럼 인덱스로 직접 여는 방식은 일반 USB 웹캠(UVC 장치)에서만 동작합니다.
# CSI 카메라는 ISP(Argus)를 거쳐야 하므로 아래처럼 GStreamer 파이프라인을 통해 열어야 합니다.
# USB 웹캠이 연결된 환경이라면 `gstreamer_pipeline()` 대신 `0`(장치 인덱스)을 그대로 사용하면 됩니다.

# %%
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# %% [markdown]
# #### `cv2.VideoCapture()`:

# %%
import cv2

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. USB 웹캠이 연결되어 있는지 확인하세요.")
else:
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            cv2.imshow("VideoCapture", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()

# %% [markdown]
# #### `imutils.video.VideoStream()`:

# %%
import cv2
import time
from imutils.video import VideoStream

vs = VideoStream(src=gstreamer_pipeline()).start()
time.sleep(1.0)

try:
    while True:
        frame = vs.read()

        if frame is None:
            print("프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("VideoStream", frame)
finally:
    vs.stop()
    cv2.destroyAllWindows()
