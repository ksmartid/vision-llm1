# -*- coding: utf-8 -*-
"""
Step 9/10: H. TensorRT 기반 YOLO 모델 최적화
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# ### H. TensorRT 기반 YOLO 모델 최적화

# %% [markdown]
# YOLO 모델 최적화:
# * 학습된 YOLO 모델을 배포 환경에서 더 빠르고 효율적으로 실행하기 위한 과정
# * 객체 탐지 성능을 유지하면서 추론 시간과 자원 사용량을 줄이는 것이 목적

# %% [markdown]
# YOLO 모델 최적화 필요성:
# * Jetson과 같은 Edge Device는 연산 성능, 메모리, 전력이 제한적
# * 실시간 영상 처리에서는 추론 속도가 낮으면 FPS가 감소하고 응답 지연이 발생
# * 실제 환경에서 안정적인 실시간 객체 탐지를 위해 모델 최적화가 중요

# %% [markdown]
# 우선, PyTorch 기반 YOLO 모델(`.pt`)을 TensorRT 엔진(.engine)으로 변환하기 위해서는 `tensorrt` 라이브러리가 필요합니다.
#
# 변환 과정에서 Python 코드로 직접 `tensorrt` 라이브러리를 import하여 사용하지는 않지만, 모델 변환 과정에서 내부적으로 사용되므로 정상적으로 설치되어 있는지 확인하도록 합시다.
#
# JetPack 6.2 환경에는 TensorRT 10.3이 기본적으로 포함되어 있습니다.

# %%
import tensorrt
print(tensorrt.__version__)

# %% [markdown]
# PyTorch와 CUDA 또한 정상적으로 설치되어 있는지 다시 확인해봅시다.

# %%
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# PyTorch, TensorRT 패키지와 CUDA가 정상적으로 확인되었다면 다음 단계로 넘어가도록 합시다.

# %% [markdown]
# #### $1)$ ONNX 변환

# %% [markdown]
# ONNX:
# * Open Neural Network Exchange
# * 딥러닝 모델을 서로 다른 프레임워크에서 사용할 수 있도록 만든 표준 모델 형식
# * PyTorch에 종속되지 않고 다양한 추론 환경에서 모델을 사용 가능
# * TensorRT가 모델을 최적화할 수 있도록 연결해주는 중간 형식으로 활용
#

# %% [markdown]
# 현재 모델:
# ```text
# src/models/YOLO/yolo11n.pt
# ```
# 을 ONNX로 변환해봅시다.

# %% [markdown]
# 우선, 기존 패키지들의 버전을 유지하면서 ONNX 변환에 필요한 패키지와 의존성들을 호환되는 버전으로 설치합시다.

# %% [markdown]
# ```bash
# pip install --no-deps onnx==1.16.0
# pip install --no-deps onnxruntime==1.23.2
# pip install --no-deps onnxslim==0.1.95
# pip install --no-deps ml_dtypes==0.5.4
# pip install --no-deps coloredlogs==15.0.1
# pip install --no-deps humanfriendly==10.0
# pip install --no-deps protobuf==3.20.3
# ```

# %% [markdown]
# 혹시 모를 버전 충돌을 방지하기 위해 패키지 자동 설치 기능을 비활성화합니다.

# %%
import os

os.environ["YOLO_AUTOINSTALL"] = "False"

# %%
from ultralytics import YOLO


model = YOLO("src/models/YOLO/yolo11n.pt")

onnx_path = model.export(
    format="onnx",  # 모델을 ONNX 형식으로 내보내기
    imgsz=640,      # 입력 이미지 크기: 640×640
    batch=1,        # 한 번에 1장의 프레임 처리
    dynamic=False,  # 입력 크기를 640×640으로 고정 (Static Shape)
    simplify=True,  # ONNX 연산 그래프 단순화
    opset=20,       # ONNX 연산 규격 버전을 20으로 설정
)

print("ONNX:", onnx_path)

# %% [markdown]
# 에러 없이 해당 코드가 완료되었다면, `src/models/YOLO/yolo11n.onnx` 파일이 생성되었음을 확인할 수 있습니다.

# %% [markdown]
# #### $2)$ 생성된 ONNX 검증

# %% [markdown]
# 생성된 `yolo11n.onnx` 파일이 ONNX 형식으로 정상적으로 변환되었는지 확인해봅시다.

# %%
import onnx


onnx_model = onnx.load(
    "src/models/YOLO/yolo11n.onnx"
)

onnx.checker.check_model(onnx_model)

print("ONNX model is valid.")

# %% [markdown]
# 입력 구조도 확인합시다.
#
# 일반적으로 YOLO export의 입력 Tensor 이름은 `images`입니다.

# %%
for input_info in onnx_model.graph.input:
    print(input_info.name)

# %% [markdown]
# #### $3)$ TensorRT 변환

# %% [markdown]
# TensorRT:
# * NVIDIA에서 제공하는 딥러닝 추론 최적화 라이브러리
# * 학습이 끝난 모델을 NVIDIA GPU에서 더 빠르게 실행하도록 최적화
# * 최적화된 모델은 보통 TensorRT Engine(.engine) 형태로 저장

# %% [markdown]
# TensorRT는 모델을 FP32, FP16, INT8 등의 정밀도로 최적화하여 TensorRT 엔진(`.engine`)으로 변환할 수 있습니다.
#
# 우선, TensorRT 모델 변환 및 성능 측정을 위한 CLI 도구인 `trtexec`을 사용하여 ONNX 모델을 FP32 TensorRT 엔진으로 변환하는 방법부터 살펴봅시다.

# %% [markdown]
# ##### FP32 Engine으로 변환:

# %% [markdown]
# ```bash
# /usr/src/tensorrt/bin/trtexec \
#     --onnx=$HOME/vision-llm/src/models/YOLO/yolo11n.onnx \
#     --saveEngine=$HOME/vision-llm/src/models/YOLO/yolo11n_fp32.engine
# ```

# %% [markdown]
# PyTorch 모델은 기본적으로 FP32 정밀도를 사용합니다. 따라서 FP32 TensorRT 엔진으로 변환하면 연산 그래프 최적화, 연산 결합, GPU 실행 최적화 등은 적용되지만, 정밀도 감소를 통한 최적화는 적용되지 않습니다.
#
# 모델을 더욱 최적화하기 위해서는 FP32보다 FP16 정밀도를 사용하는 것이 좋습니다.

# %% [markdown]
# ##### FP16 Engine으로 변환:

# %% [markdown]
# ```bash
# /usr/src/tensorrt/bin/trtexec \
#     --onnx=$HOME/vision-llm/src/models/YOLO/yolo11n.onnx \
#     --saveEngine=$HOME/vision-llm/src/models/YOLO/yolo11n_fp16.engine \
#     --fp16
# ```

# %% [markdown]
# 에러 없이 해당 명령어가 완료되었다면, `src/models/YOLO/yolo11n_fp16.engine` 파일이 생성되었음을 확인할 수 있습니다.

# %% [markdown]
# #### $4)$ 최적화 성능 확인

# %% [markdown]
# TensorRT 엔진으로 변환한 YOLO 모델의 추론 성능을 측정해봅시다.
#
# 아래 명령어를 사용하여 확인할 수 있습니다.

# %% [markdown]
# ```bash
# /usr/src/tensorrt/bin/trtexec \
#     --loadEngine=$HOME/vision-llm/src/models/YOLO/yolo11n_fp16.engine \
#     --warmUp=500 \
#     --duration=10
# ```

# %% [markdown]
# 출력 결과에서 확인해야 할 항목들은 다음과 같습니다.

# %% [markdown]
# | 항목 | 의미 |
# | --- | --- |
# | `Throughput` | 초당 처리 가능한 추론 횟수 |
# | `Latency` | 한 번의 추론에 걸리는 전체 시간 |
# | `GPU Compute Time` | GPU가 실제 모델 연산에 사용한 시간 |

# %% [markdown]
# #### $5)$ TensorRT 엔진으로 실시간 객체 탐지

# %% [markdown]
# 이제 TensorRT 엔진으로 최적화 변환된 YOLO 모델을 활용하여 실시간 객체 탐지를 진행해 봅시다.
#
# 최적화 이전 모델과 FPS를 비교하며 성능 향상 폭을 확인해 보도록 합시다.
#
# `.engine` 파일을 모델로 불러오는 경우에는, Engine 자체가 이미 NVIDIA GPU용으로 빌드된 파일이기 때문에 `model.to("cuda")`를 하지 않습니다.

# %%
from ultralytics import YOLO
import cv2
import time


model = YOLO("src/models/YOLO/yolo11n_fp16.engine")

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
