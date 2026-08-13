# -*- coding: utf-8 -*-
"""
Step 6/10: F. 실시간 객체 탐지 성능 평가 (FPS)
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# ### F. 실시간 객체 탐지 성능 평가 (FPS)

# %% [markdown]
# YOLO 객체 탐지는 실시간성이 매우 뛰어난 모델이나, 하드웨어 성능, 모델의 크기, 입력 해상도 등, 많은 요소에 따라 처리 속도가 크게 변동됩니다.
#
# 특히 하드웨어 성능이 크게 제약되는 Edge 디바이스에서는 실시간 프레임 저하가 발생하기 쉽습니다.
#
# 따라서 목표하는 실시간 성능을 확보하기 위해 FPS를 정밀하게 측정하고 모니터링하여 성능을 최적화하는 과정이 필수적입니다.

# %% [markdown]
# FPS (Frames Per Second, 초당 프레임 수):
# * 1 / 프레임_처리_시간_s  =  1000 / 프레임_처리_시간_ms
# * 객체 탐지 모델이 초당 처리하는 이미지 프레임 수
# * 실시간 처리의 핵심 지표
# * 모델 크기(n, s, m, l, x)가 커질 수록 FPS 감소
# * 정확도와의 Trade-off
# * 엣지 디바이스(Edge Device)의 한계

# %% [markdown]
# 객체 탐지에서의 FPS:
# 1. 추론 FPS
#     * 모델 Forward 속도
# 2. End-to-End FPS
#     * “프레임 입력 + 프레임 전처리 + 추론 + NMS + BBox 시각화 + 화면 출력” 속도
#     * 체감 FPS
#     * 실제 성능 평가

# %% [markdown]
# 매 프레임 단위로 FPS를 즉시 계산하면 순간적인 연산 Bottleneck이나 프레임 드롭 때문에 수치의 변동 폭이 지나치게 커질 수 있습니다
#
# 이를 방지하고 안정적이고 부드러운 FPS 수치를 얻기 위해 지수 이동 평균(Exponential Moving Average, EMA) 방식을 활용합니다. EMA는 이전 프레임까지의 누적 FPS 평균값과 현재 측정된 FPS 값에 각각 가중치를 부여하여 결합함으로써, 최신 변경 사항을 반영하면서도 급격한 FPS 변화(Noise)를 완화해 줍니다.

# %% [markdown]
# 위의 실시간 객체 탐지 코드에 FPS 계산 및 모니터링 코드를 추가하여 실시간성을 확인해봅시다.

# %%
from ultralytics import YOLO
import cv2
import time


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

displayed_fps = 0.0

while True:
    start_time = time.perf_counter()

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

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    cv2.putText(output_frame, f"FPS: {displayed_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Detection with FPS", output_frame)

cap.release()
cv2.destroyAllWindows()

# %% [markdown]
# ---
