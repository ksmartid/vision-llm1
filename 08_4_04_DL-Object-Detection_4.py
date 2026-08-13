# -*- coding: utf-8 -*-
"""
Step 4/10: D. YOLO 개요
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# ### D. YOLO 개요

# %% [markdown]
# 지금까지 객체 탐지 시스템의 기초 원리와 주요 기법들을 살펴보았습니다.
#
# 이제 카메라 피드 기반의 실시간 객체 탐지를 위해, YOLO 모델을 알아보겠습니다.

# %% [markdown]
# **YOLO (You Only Look Once):**
# * CNN 기반 객체 탐지 모델
# * One-stage Detector
# * 이미지 전체를 신경망에 한 번 입력하여, 이미지 속 객체들을 전부 탐지
# * 객체들의 정보 동시 예측
#   * 위치: Bounding Box
#   * 종류: Class
#   * 신뢰도: Confidence
# * 실시간 처리에 적합
#   * 별도의 Region Proposal 단계가 없음
#   * Feature Map을 여러 객체가 공유함
#   * 여러 위치의 예측을 병렬로 계산함
#   * GPU의 병렬 연산을 효율적으로 활용
#   * 작은 모델부터 큰 모델까지 선택 가능

# %% [markdown]
# 우선, YOLO 모델을 활용하기 위해 공식 파이썬 라이브러리인 `ultralytics`를 설치해야합니다.
#
# `ultralytics` 패키지를 설치하기 위해서는 공식 문서에 명시된 `numpy>=1.23.0`, `opencv-python>=4.6.0` 버전 요구사항에 맞춰야 하므로, 설치 전 현재 버전을 먼저 확인해 보겠습니다.

# %%
import numpy as np
import cv2

print(np.__version__)
print(cv2.__version__)

# %% [markdown]
# 확인 결과 `OpenCV`는 버전 요구사항에 맞지만, `NumPy`는 업그레이드가 필요합니다.
#
# 따라서 `NumPy`만 권장 버전으로 업그레이드를 진행해 보겠습니다.
#
# 주의사항으로는 `NumPy`를 업그레이드할 때 버전 2.0 이상으로 올라가지 않도록 주의해야 합니다. `NumPy 2.x` 버전에서는 GStreamer가 연동된 `OpenCV`와 호환성 문제가 발생하여 정상적으로 동작하지 않습니다.

# %% [markdown]
# ```bash
# pip install "numpy==1.23.0"
# ```

# %%
import numpy as np
import cv2

print(np.__version__)
print(cv2.__version__)

# %% [markdown]
# 위 코드에서 `NumPy` 버전이 업데이트 되지 않는다면, "Ctrl+Shift+P"을 눌러 "Jupyter: Restart Kernel"을 선택하여 다시 실행해봅시다.

# %% [markdown]
# 다음으로, `ultralytics`를 의존성 패키지(Dependency) 없이 최소 설치 한 후, 필수 의존 패키지만 따로 추가 설치합시다.

# %% [markdown]
# ```bash
# pip install --no-deps ultralytics
# pip install --no-deps "typing_extensions>=4.10.0,<5"
# pip install --no-deps "filelock>=3.16.1,<4"
# ```

# %% [markdown]
# 정상적으로 설치가 되었는지 아래 코드로 확인해봅시다.

# %%
from ultralytics import YOLO

# %% [markdown]
# 에러 없이 `ultralytics` 라이브러리가 import 되었다면 다음으로 진행하면 되겠습니다.

# %% [markdown]
# #### $1)$ YOLO 모델 선택

# %% [markdown]
# YOLO는 사용 목적과 컴퓨터 환경에 맞게 선택할 수 있도록 매우 다양한 모델 라인업을 제공합니다.
#
# YOLO 모델의 이름은 크게 ***모델명 + 버전 + 모델 크기***의 구조로 이루어져 있습니다.

# %% [markdown]
# 예) YOLO11n
# * **YOLO**: **모델 이름** *(You Only Look Once)*
# * **숫자 (11)**: **버전 정보** (V1부터 지속적으로 발전하여 최근에는 **YOLO11**까지 출시)
# * **알파벳 (n)**: **모델 크기 및 파라미터 수** (n, s, m, l, x 등)

# %% [markdown]
# 모델 크기(알파벳) 구분
#
# | 알파벳 | 약자 | 특징 |
# | :---: | :---: | :--- |
# | **n** | Nano | 경량화 모델 (속도 최우선, 연산량 적음) |
# | **s** | Small | 소형 모델 (가벼우면서 적절한 성능) |
# | **m** | Medium | 중형 모델 (속도와 정확도의 균형) |
# | **l** | Large | 대형 모델 (높은 정확도) |
# | **x** | Extra Large | 초대형 모델 (최고 성능, 높은 연산량 필요) |

# %% [markdown]
# YOLO는 최신 버전인 `YOLO26`까지 출시되어 있지만, 본 수업에서는 아래와 같은 이유로 `YOLO11n` 모델을 선택하여 진행합니다.
#
# 1. **안정성과 호환성**: 최신 `YOLO26`에 비해 `YOLO11`은 라이브러리 및 실행 환경(PyTorch, OpenCV 등)과의 의존성 충돌 위험이 적고 매우 안정적
# 2. **실시간 가벼운 동작**: `n(Nano)` 모델은 연산량이 적어 한계가 많은 Jetson과 같은 엣지 디바이스에서도 실시간 카메라 피드 탐지를 무리 없이 수행
# 3. **빠르고 효율적인 실습**: 모델 가중치 파일 용량이 가벼워 로딩이 빠르며, 경량화 환경에서도 YOLO의 핵심 메커니즘을 학습하기에 최적화된 모델

# %% [markdown]
# "YOLO11n을 포함한 기본 YOLO 모델들은 수많은 이미지 데이터로 이미 사전 학습된 모델(Pre-trained Model)입니다.
#
# 이 모델은 일상적인 80가지 범주의 객체가 포함된 COCO(Common Objects in Context) 데이터셋으로 학습되어 있어, 별도의 추가 학습 없이도 바로 사람, 차량, 동물 등을 탐지할 수 있습니다.

# %% [markdown]
# #### $2)$ COCO 데이터셋

# %% [markdown]
# COCO 데이터셋:
# * **컴퓨터 비전 대표 데이터셋**
#   * Microsoft에서 제작한 대규모 객체 탐지(Object Detection) 및 분할(Segmentation) 데이터셋
# * **80가지 클래스 제공**
#   * 사람, 자동차, 강아지, 의자 등 일상에서 흔히 접하는 80가지 범주의 객체 정보가 포함되어 높은 활용도
# * **복잡한 배경과 맥락(Context)**
#   * 단일 물체만 깔끔하게 있는 이미지뿐만 아니라, 여러 객체가 어우러진 실제 환경 이미지가 다수 포함되어 있어 높은 실전 인식 성능

# %% [markdown]
# COCO 데이터셋의 80가지 클래스:
#
# ![](src/images/COCO_80_classes.png)

# %% [markdown]
# #### $3)$ YOLO 모델 최초 설치 및 불러오기

# %% [markdown]
# 이제 `YOLO11n` 모델을 불러와 보겠습니다.
#
# 앞서 다룬 MNIST 및 CIFAR-10 데이터셋처럼, 파일을 지정한 경로에서 불러오거나 파일이 존재하지 않는다면 자동으로 다운로드됩니다.
#
# 최초 사용하는 모델이므로, 경로를 지정하여 설치해봅시다.

# %%
model = YOLO("src/models/YOLO/yolo11n.pt")

# %% [markdown]
# Ultralytics는 GPU(CUDA)가 사용 가능한 환경이라면 기본적으로 모델을 `cuda:0` 디바이스에 자동으로 할당합니다.
#
# 만약 모델 실행 장치를 명시적으로 지정하고 싶다면 아래와 같이 디바이스 설정 코드를 추가해 줄 수 있습니다.

# %%
model.to("cuda")

print(model.device)

# %% [markdown]
# #### $4)$ 정적 이미지에서의 YOLO 객체 탐지

# %% [markdown]
# 이제 불러온 모델에 샘플 이미지를 적용하여 객체 탐지가 잘 동작하는지 확인해 볼까요?

# %%
import matplotlib.pyplot as plt

# %%
results = model.predict(
    source="src/images/city.png",
    conf=0.25,   # Confidence Threshold
    iou=0.5,     # IoU Threshold
    classes=None,
)

yolo_test_img = results[0].plot()  # 일반 .py 파일에서는 results[0].show()

yolo_test_img_rgb = cv2.cvtColor(yolo_test_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,8))
plt.imshow(yolo_test_img_rgb)
plt.axis("off");

# %% [markdown]
# 위 코드에서 `results[0]`와 같이 인덱싱을 한 이유는, YOLO가 여러 이미지를 동시에 처리(Batch processing)할 수 있어 결과가 리스트 형태로 반환되기 때문입니다.

# %%
results = model.predict(
    source=[
        "src/images/city.png",
        "src/images/apple.png",
    ],
    conf=0.25,   # Confidence Threshold
    iou=0.5,     # IoU Threshold
    classes=None,
)

yolo_test_img0 = results[0].plot()
yolo_test_img1 = results[1].plot()

yolo_test_img0_rgb = cv2.cvtColor(yolo_test_img0, cv2.COLOR_BGR2RGB)
yolo_test_img1_rgb = cv2.cvtColor(yolo_test_img1, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(18,10))

plt.subplot(1,2,1), plt.imshow(yolo_test_img0_rgb)
plt.subplot(1,2,2), plt.imshow(yolo_test_img1_rgb)

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# #### $5)$ 모델 출력 결과 확인

# %% [markdown]
# YOLO 모델이 계산한 최종 BBox 좌표, 클래스 인덱스, 신뢰도 점수는 아래와 같이 확인할 수 있습니다.

# %%
result = results[0]

print("* BBox 좌표:\n  ", result.boxes.xyxy)
print("\n* 클래스 Index:\n  ", result.boxes.cls)
print("\n* 객체별 Confidence Score:\n  ", result.boxes.conf)

# %% [markdown]
# #### $6)$ 클래스 지정 객체 탐지

# %% [markdown]
# YOLO 예측 함수인 `model.predict()`에는 원하는 클래스를 지정하여 객체를 탐지할 수 있는 기능이 있습니다.

# %%
results = model.predict(
    source="src/images/city.png",
    conf=0.25,   # Confidence Threshold
    iou=0.5,     # IoU Threshold
    classes=[2]  # 2번 클래스 = 자동차
)

yolo_test_img = results[0].plot()  # 일반 .py 파일에서는 results[0].show()

yolo_test_img_rgb = cv2.cvtColor(yolo_test_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,8))
plt.imshow(yolo_test_img_rgb)
plt.axis("off");

# %% [markdown]
# ---
