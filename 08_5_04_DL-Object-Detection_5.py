# -*- coding: utf-8 -*-
"""
Step 5/10: E. YOLO 실시간 객체 탐지
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# ### E. YOLO 실시간 객체 탐지

# %% [markdown]
# 앞선 실습에서는 정적 이미지를 활용해 간단한 객체 탐지 과정을 경험해 보았습니다.
#
# 이번에는 CSI 카메라에서 들어오는 실시간 프레임에 YOLO 모델을 적용하여, 동적 영상 환경에서 객체를 실시간으로 탐지하는 실습을 진행해 보겠습니다.

# %% [markdown]
# 기본적으로 ***Computer Vision*** 섹션에서 OpenCV를 활용한 실시간 카메라 피드 처리 방식과 동일합니다.
#
# 동일하게 GStreamer 파이프라인을 통해 OpenCV `VideoCapture`로 영상 스트림을 받아오도록 합시다.

# %% [markdown]
# ```python
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
#     cv2.imshow("VideoCapture with GStreamer", frame)
#
# cap.release()
# cv2.destroyAllWindows()
# ```

# %% [markdown]
# GStreamer를 활용한 기본적인 실시간 카메라 프레임 처리 코드에 YOLO 모델을 불러와 실시간 객체 추적을 구현해봅시다.
#
# 지난 섹션에서 확인하셨다시피 Jupyter 환경에서는 OpenCV GUI 방식으로는 실행할 수 없는 관계로, 새로운 .py 파일에 복사하여 실행해봅시다.

# %%
from ultralytics import YOLO
import cv2


model = YOLO("src/models/YOLO/yolo11n.pt")
model.to("cuda")

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    results = model.predict(
        source=frame,   # source image
        conf=0.25,      # Confidence Threshold
        iou=0.5,        # IoU Threshold
        verbose=False,  # no output prints
        classes=None,   # selected class
    )

    output_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", output_frame)

cap.release()
cv2.destroyAllWindows()

# %% [markdown]
# ---
