# -*- coding: utf-8 -*-
"""
Step 10/10: G. 양자화
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# 이전 단계(Step 9, H. TensorRT 기반 YOLO 모델 최적화)에서 사용한 라이브러리를 이어서 사용합니다.

# %%
from ultralytics import YOLO

# %% [markdown]
# ### G. 양자화

# %% [markdown]
# 양자화(Quantization)란?
# * 모델의 가중치와 연산에 사용하는 숫자의 정밀도를 낮추는 최적화 기법
# * 고정밀도 값을 더 적은 비트로 표현
# * 장점: 모델 크기, 메모리 사용량 감소 및 추론 속도 향상
# * 당점: 정확도 감소
# * Edge Device에서 중요
#

# %% [markdown]
# Jetson은 연산 성능과 메모리와 같이 사용 가능한 리소스가 제한된 Edge Device이므로 모델을 더욱 효율적으로 최적화할 필요가 있습니다.
#
# 따라서 FP16보다 한 단계 더 낮은 정밀도인 INT8 양자화까지 적용해봅시다.

# %% [markdown]
# #### $1)$ Calibration 데이터셋 준비

# %% [markdown]
# INT8은 FP32보다 표현할 수 있는 값의 범위가 작기 때문에, FP32 값을 어떤 INT8 값으로 변환할지 기준을 정하는 과정이 필요합니다. 이를 Calibration이라고 합니다.
#
# Calibration에서는 실제 추론 환경과 유사한 대표 이미지 데이터셋을 모델에 입력하여 각 Layer에서 발생하는 값의 분포를 분석하고, INT8 변환에 사용할 Quantization Scale을 계산합니다.
#
# 따라서 먼저 실제 사용 환경을 잘 대표할 수 있는 Calibration 데이터셋을 준비합니다.

# %% [markdown]
# Calibration 데이터셋은 일반적으로 수백 장 정도의 대표 이미지로 구성하며, 이번 실습에서는 약 500장의 이미지를 사용합니다.
#
# 실제 추론 환경과 최대한 유사한 데이터를 사용하기 위해 Jetson에 연결된 카메라의 프레임을 직접 캡처하여 Calibration 이미지로 저장합니다.
#
# 아래 코드를 별도 `.py` 파일에 복사하여 실행해봅시다.

# %%
from pathlib import Path
import yaml

import cv2
from ultralytics import YOLO


CALIBRATION_DIR = Path("src/datasets/calibration")
IMAGE_DIR = CALIBRATION_DIR / "images"
LABEL_DIR = CALIBRATION_DIR / "labels"
YAML_PATH = CALIBRATION_DIR / "calibration.yaml"

NUM_IMAGES = 500
SAVE_EVERY_N_FRAMES = 5

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


frame_count = 0
save_count = 0

while save_count < NUM_IMAGES:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Calibration: {save_count}/{NUM_IMAGES}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imshow("Calibration Image Collection", display_frame)

    if frame_count % SAVE_EVERY_N_FRAMES == 0:
        # Image (jpg) 파일 저장
        image_path = IMAGE_DIR / f"{save_count:04d}.jpg"
        cv2.imwrite(str(image_path), frame)

        # Label (txt) 파일 저장 (빈 파일)
        label_path = LABEL_DIR / f"{save_count:04d}.txt"
        label_path.touch()

        save_count += 1

        print(f"Saved: {save_count}/{NUM_IMAGES}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()


if save_count < NUM_IMAGES:
    raise RuntimeError(f"이미지가 {save_count}장만 저장되었습니다.")


model = YOLO("src/models/YOLO/yolo11n.pt")

# YAML 파일에 저장할 Dictionary
calibration_yaml = {
    "path": str(CALIBRATION_DIR.resolve()),
    "train": "images",
    "val": "images",
    "names": model.names,
}

# YAML 파일 저장
with open(YAML_PATH, "w", encoding="utf-8") as file:
    yaml.safe_dump(
        calibration_yaml,
        file,
        sort_keys=False,
        allow_unicode=True,
    )

print()
print("Calibration Dataset 생성 완료")
print(f"Images: {IMAGE_DIR}")
print(f"YAML:   {YAML_PATH}")

# %% [markdown]
# #### $2)$ INT8 양자화

# %% [markdown]
# 앞에서 저장한 Calibration 데이터셋을 사용하여 모델의 값 분포를 분석하고, INT8 변환에 필요한 Quantization Scale을 계산합니다.
#
# 계산된 Scale을 기반으로 모델을 INT8 정밀도로 양자화하여 TensorRT Engine으로 변환합니다.

# %%
from pathlib import Path


model = YOLO("src/models/YOLO/yolo11n.pt")

engine_path = model.export(
    format="engine",
    imgsz=640,
    quantize=8,
    data="src/datasets/calibration/calibration.yaml",
    batch=1,
    dynamic=False,
    device=0,
    nms=False,
)

print(f"TensorRT Engine 생성 완료: {engine_path}")

# 파일 이름 변경
int8_engine_path = Path(engine_path).replace("src/models/YOLO/yolo11n_int8.engine")

print(f"INT8 Engine: {int8_engine_path}")

# %% [markdown]
# 에러 없이 해당 코드가 완료되었다면, `src/models/YOLO/yolo11n_int8.engine` 파일이 생성되었음을 확인할 수 있습니다.

# %% [markdown]
# #### $3)$ 최적화 성능 확인

# %% [markdown]
# INT8 양자화를 통해 TensorRT 엔진으로 변환한 YOLO 모델의 추론 성능을 측정해봅시다.
#
# Ultralytics로 생성한 TensorRT Engine에는 모델 정보가 담긴 Ultralytics 전용 메타데이터가 앞부분에 추가됩니다.
#
# 하지만 trtexec은 순수 TensorRT Engine 형식만 읽기 때문에, 성능 측정 시에는 해당 메타데이터를 제거한 Raw Engine을 사용해야 합니다.
#
# 따라서 메타데이터를 제거한 INT8 엔진을 생성합시다.

# %%
import json
from pathlib import Path


engine_path = Path("src/models/YOLO/yolo11n_int8.engine")
raw_engine_path = Path("src/models/YOLO/yolo11n_int8_raw.engine")

with engine_path.open("rb") as f:
    # 앞의 4 bytes: metadata 길이
    metadata_length = int.from_bytes(f.read(4), byteorder="little")

    # 메타데이터 읽기
    metadata = json.loads(f.read(metadata_length).decode("utf-8"))

    # 나머지 실제 TensorRT 엔진
    raw_engine = f.read()

raw_engine_path.write_bytes(raw_engine)

print(f"Metadata: {metadata}")
print(f"Raw TensorRT Engine: {raw_engine_path}")

# %% [markdown]
# 이렇게 메타데이터를 제거하여 생성한 엔진 `yolo11n_int8_raw.engine`은 오직 `trtexec`으로 최적화 성능을 확인할 경우에만 사용합니다.
#
# 실제 객체 탐지에는 `yolo11n_int8_raw.engine`을 사용하지 않습니다.
#
# 이제 아래 명령어로 성능을 확인해봅시다.

# %% [markdown]
# ```bash
# /usr/src/tensorrt/bin/trtexec \
#     --loadEngine=$HOME/vision-llm/src/models/YOLO/yolo11n_int8_raw.engine \
#     --warmUp=500 \
#     --duration=10
# ```

# %% [markdown]
# 세 가지 최적화 모델들의 성능을 비교해봅시다.
#
# ```text
# yolo11n_fp32.engine
# yolo11n_fp16.engine
# yolo11n_int8.engine
# ```

# %% [markdown]
# #### $4)$ INT8 TensorRT 엔진으로 실시간 객체 탐지

# %% [markdown]
# 아래 코드를 `.py` 파일에 복사하여 INT8 TensorRT 엔진으로 실시간 객체 탐지를 시도해봅시다.

# %%
from ultralytics import YOLO
import cv2
import time


model = YOLO("src/models/YOLO/yolo11n_int8.engine")

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
# INT8 TensorRT 엔진을 사용한 실시간 객체 탐지에서도 FP16 엔진과 동일하게 약 30 FPS가 출력됩니다.
#
# 이는 모델의 최대 추론 속도가 30 FPS이기 때문이 아니라, 카메라 Pipeline에서 입력 프레임 속도를 30 FPS로 설정했기 때문입니다.
#
# INT8 TensorRT 엔진으로 객체 탐지가 정상적으로 수행되고 실시간으로 약 30 FPS가 유지된다면 최적화가 성공적으로 적용된 것입니다.
#
# 이것으로 **"딥러닝 기반 객체 탐지 시스템"** 섹션을 마무리합니다.
#
# 다음 섹션에서는 새로운 주제인 LLM에 관하여 배워보도록 하겠습니다.

# %% [markdown]
# ---
# ---

# %% [markdown]
# <br><br><div style="text-align: right; color: gray; font-style: italic;">
# © 2026, 김규래 (Kyu Rae Kim), All rights reserved.&emsp;<br><br>
# This material is provided solely for the intended instructional purpose.&emsp;<br>
# Redistribution, reproduction, modification, adaptation, or reuse of this material in any form without prior written permission from the copyright holder is prohibited.&emsp;
# </div>
