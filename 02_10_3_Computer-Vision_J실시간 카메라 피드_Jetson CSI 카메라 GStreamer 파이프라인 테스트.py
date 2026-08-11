# -*- coding: utf-8 -*-
"""
Part J-3: 실시간 카메라 피드 - Jetson CSI 카메라 GStreamer 파이프라인 테스트
원본: 02_Computer-Vision.ipynb (cell 231~248)
"""


# %% [markdown]
# #### $3)$ Jetson CSI 카메라 GStreamer 파이프라인 테스트

# %% [markdown]
# #### `GStreamer`:

# %% [markdown]
# 우선, 카메라가 지원하는 포멧을 확인합니다.
#
# (`v4l-utils` 설치는 sudo 권한이 필요해 스크립트에서 자동 실행하지 않습니다. 미설치 시 `sudo apt install v4l-utils`를 먼저 실행하세요.)

# %%
import subprocess

try:
    result = subprocess.run(
        ["v4l2-ctl", "--list-formats-ext", "-d", "/dev/video0"],
        capture_output=True, text=True, timeout=5
    )
    print(result.stdout or result.stderr)
except FileNotFoundError:
    print("v4l2-ctl 명령을 찾을 수 없습니다. 설치: sudo apt install v4l-utils")
except Exception as e:
    print("포맷 확인 중 에러:", e)

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

# %%
pipeline_mjpeg = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# %% [markdown]
# 카메라가 YUYV를 지원할 경우:

# %%
pipeline_yuyv = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

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

# %%
pipeline_csi = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)

# 프레임 드랍을 허용해 지연을 줄이려면 아래 파이프라인을 대신 사용
# pipeline_csi = (
#     "nvarguscamerasrc sensor-id=0 ! "
#     "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
#     "nvvidconv ! "
#     "video/x-raw, format=BGRx ! "
#     "videoconvert ! "
#     "video/x-raw, format=BGR ! "
#     "queue leaky=downstream max-size-buffers=1 ! "
#     "appsink drop=true max-buffers=1 sync=false"
# )

# %% [markdown]
# pipeline 선택 후 코드:

# %%
import cv2

# 이번 실습은 Jetson CSI 카메라를 대상으로 하므로 pipeline_csi를 사용합니다.
# 일반 USB 웹캠이라면 위에서 확인한 포맷에 맞춰 pipeline_mjpeg 또는 pipeline_yuyv로 교체하세요.
pipeline = pipeline_csi

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("GStreamer 파이프라인으로 카메라를 열 수 없습니다. 파이프라인 문자열과 카메라 연결을 확인하세요.")
else:
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            cv2.imshow("VideoCapture with GStreamer", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()

# %% [markdown]
# ---
