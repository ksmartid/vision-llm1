# -*- coding: utf-8 -*-
"""
Step 7/10: G. GPU 병렬 연산 (CuPy)
원본: 03_DL-and-GPU.ipynb (단계별로 재구성)
"""

# %%
import cv2
import matplotlib.pyplot as plt

# %% [markdown]
# ### G. GPU 병렬 연산 (CuPy)

# %% [markdown]
# GPU를 사용하여 기존 `NumPy` 배열 연산을 더욱 빠르게 수행할 수 있습니다. 그러기 위해서는 `NumPy`와 호환되는 GPU 배열 연산 라이브러리인 `CuPy`를 사용합니다.
#
# `CuPy`:
# * NumPy와 호환되는 GPU 배열 라이브러리
# * NumPy와 유사한 문법으로 GPU 배열 연산
# * NVIDIA CUDA를 활용하여 대규모 연산을 GPU에서 가속
#

# %% [markdown]
# 현재 개발 환경인 JetPack 6.2, CUDA 12.6, NumPy 1.21.5에 맞춰 CuPy를 설치해야 합니다.

# %% [markdown]
# ```bash
# python -m pip download --no-deps --only-binary=:all: "cupy-cuda12x==12.3.0" -d ~/Downloads
# python -m pip install --no-deps "fastrlock==0.8.3"
# python -m pip install --no-deps --only-binary=:all: ~/Downloads/cupy_cuda12x-12.3.0-*.whl
# ```

# %% [markdown]
# 설치가 제대로 되었는지 확인해봅시다.

# %%
import numpy as np
import cupy as cp

print("NumPy :", np.__version__)
print("CuPy  :", cp.__version__)

# %%
print("GPU count:", cp.cuda.runtime.getDeviceCount())
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())

x_cpu = np.arange(10)  # NumPy (host memory)
x_gpu = cp.arange(10)  # CuPy  (CUDA device memory)
y_cpu = x_cpu ** 2     # CPU에서 NumPy로 계산
y_gpu = x_gpu ** 2     # GPU에서 CuPy로 계산

cp.cuda.Stream.null.synchronize()

print("NumPy result:", y_cpu)
print("CuPy result:", y_gpu)

print("NumPy type:", type(y_cpu))
print("CuPy type:", type(y_gpu))

# %% [markdown]
# 에러 없이 아래와 같은 결과가 출력되었다면 다음으로 넘어가도록 합시다.

# %% [markdown]
# ```text
# NumPy : 1.21.5
# CuPy  : 12.3.0
#
# GPU count: 1
# CUDA runtime: 12060
# NumPy result: [ 0  1  4  9 16 25 36 49 64 81]
# CuPy result: [ 0  1  4  9 16 25 36 49 64 81]
# NumPy type: <class 'numpy.ndarray'>
# CuPy type: <class 'cupy.ndarray'>
# ```

# %% [markdown]
# #### $1)$ CuPy 개요

# %% [markdown]
# #### `NumPy` 배열:

# %%
x_cpu = np.array([1, 2, 3, 4])

print(type(x_cpu))
print(x_cpu)

# %% [markdown]
# `NumPy`를 사용하여 정의한 데이터는 기본적으로 host(CPU) memory인 시스템 메모리(RAM)에 저장됩니다.

# %% [markdown]
# #### `CuPy` 배열:

# %%
x_gpu = cp.array([1, 2, 3, 4])

print(type(x_gpu))
print(x_gpu)

# %% [markdown]
# #### $2)$ CPU(NumPy) vs. GPU(CuPy) 메모리

# %% [markdown]
# `CuPy`를 사용하여 정의한 데이터는 device(GPU) memory인 그래픽 메모리(VRAM)에 저장됩니다.

# %% [markdown]
# 일반적인 경우에는 RAM과 VRAM이 분리되어 있지만, Jetson Orin Nano 같은 경우에는 메모리 칩을 공유하는 통합 메모리 아키텍쳐 (Unified Memory Architecture, UMA)를 사용합니다.
#
# 비록 물리적으로 나뉘어 있지는 않더라도, 통합 메모리 환경에서도 서로 다른 영역에 분리되어 있어 CPU와 GPU가 각자 할당된 메모리를 관리하며 `NumPy` 배열과 `CuPy` 배열을 함께 연산하려면 어느 한쪽의 메모리 공간으로 데이터를 먼저 복사해야 합니다.

# %% [markdown]
# #### CPU에서 GPU로 복사 (`NumPy` → `CuPy`):

# %%
x_from_cpu_to_gpu = cp.asarray(x_cpu)

print(type(x_from_cpu_to_gpu))
print(x_from_cpu_to_gpu)

# %% [markdown]
# #### GPU에서 CPU로 복사 `CuPy` → `NumPy`:

# %%
x_from_gpu_to_cpu = cp.asnumpy(x_gpu)

print(type(x_from_gpu_to_cpu))
print(x_from_gpu_to_cpu)

# %% [markdown]
# #### 잘못된 연산 예시:

# %%
try:
    y = x_cpu + x_gpu
except TypeError as e:
    print(f"잘못된 연산: {e}")

# %% [markdown]
# #### $3)$ NumPy와 CuPy 문법 비교:

# %% [markdown]
# `NumPy`:

# %%
x_cpu = np.array([1, 2, 3, 4])

total = np.sum(x_cpu)
maximum = np.max(x_cpu)
minimum = np.min(x_cpu)
mean = np.mean(x_cpu)
index = np.argmax(x_cpu)

print(total, maximum, minimum, mean, index)
print(type(total), type(maximum), type(minimum), type(mean), type(index))

# %% [markdown]
# `CuPy`:

# %%
x_gpu = cp.array([1, 2, 3, 4])

total = np.sum(x_gpu)
maximum = np.max(x_gpu)
minimum = np.min(x_gpu)
mean = np.mean(x_gpu)
index = np.argmax(x_gpu)

print(total, maximum, minimum, mean, index)
print(type(total), type(maximum), type(minimum), type(mean), type(index))

total = cp.sum(x_gpu).item()
maximum = cp.max(x_gpu).item()
minimum = cp.min(x_gpu).item()
mean = cp.mean(x_gpu).item()
index = cp.argmax(x_gpu).item()

print(total, maximum, minimum, mean, index)
print(type(total), type(maximum), type(minimum), type(mean), type(index))

# %% [markdown]
# `NumPy`:

# %%
a_cpu = np.random.rand(1000, 1000).astype(np.float32)
b_cpu = np.random.rand(1000, 1000,).astype(np.float32)

c_cpu = a_cpu @ b_cpu

print(c_cpu)

# %% [markdown]
# `CuPy`:

# %%
a_gpu = cp.random.rand(1000, 1000, dtype=cp.float32)
b_gpu = cp.random.rand(1000, 1000, dtype=cp.float32)

c_gpu = a_gpu @ b_gpu

print(c_gpu)

# %% [markdown]
# `NumPy`:

# %%
import time


start = time.perf_counter()

y_cpu = x_cpu ** 2

end = time.perf_counter()

print(end - start)

# %% [markdown]
# `CuPy`:

# %%
import time


cp.cuda.Stream.null.synchronize()
start = time.perf_counter()

y_gpu = x_gpu ** 2

cp.cuda.Stream.null.synchronize()
end = time.perf_counter()

print(end - start)

# %% [markdown]
# `OpenCV` 및 `Matplotlib`:

# %% [markdown]
# ```python
# ret, frame = cap.read()              # NumPy, CPU
#
# frame_gpu = cp.asarray(frame)        # CPU → GPU
# result_gpu = frame_gpu ** 2          # GPU 연산
# result_cpu = cp.asnumpy(result_gpu)  # GPU → CPU
#
# cv2.imshow("Result", result_cpu)     # OpenCV (NumPy)
# plt.imshow(result_cpu)               # Matplotlib (NumPy)
# ```

# %% [markdown]
# 문법은 매우 비슷하지만 물리적으로 연산이 실행되는 장치가 다릅니다. `NumPy`는 CPU 코어, `CuPy`는 GPU 코어에서 연산이 일어납니다.
#
# Jetson과 같이 CPU와 GPU가 물리적인 메모리를 공유하는 환경에서도, 두 장치의 연산 코어 수와 성능, 그리고 사용 목적은 엄연히 다릅니다.
#
# Jetson Orin Nano에서는 `NumPy` 배열을 6개의 고성능 CPU 코어가 복잡하고 정교한 제어 흐름과 함께 순차적으로 연산을 처리하는 반면, `CuPy` 함수를 실행하면 1024개의 GPU 코어가 일제히 대규모 병렬 연산을 수행합니다.

# %% [markdown]
# #### $4)$ CuPy를 활용한 배열 연산 GPU 가속 1 – 이미지 잔상 효과 구현

# %% [markdown]
# `CuPy`를 사용하여 대규모 배열 연산이 GPU에서 어떻게 가속되는지 알아보겠습니다.
#
# 이번 실습에서는 이미지 배열을 여러 픽셀만큼 이동한 뒤, 각 이미지에 서로 다른 가중치를 곱하여 누적하는 이미지 잔상(Image Trail) 효과를 구현합니다.
#
# #### 이미지 잔상 효과의 원리
#
# 원본 이미지를 오른쪽으로 조금씩 이동한 복사본들을 만들고, 각 복사본에 가중치를 곱한 뒤 모두 더합니다.
#
# ```text
# 원본 이미지 × weight[0]
# 오른쪽으로 1픽셀 이동한 이미지 × weight[1]
# 오른쪽으로 2픽셀 이동한 이미지 × weight[2]
# 오른쪽으로 3픽셀 이동한 이미지 × weight[3]
# ...
# ```
#
# 가중치는 이동 거리가 멀어질수록 작아지도록 설정합니다. 따라서 원본 이미지에 가까운 잔상은 선명하게 나타나고, 멀리 떨어진 잔상은 점점 흐리게 나타납니다.
#
# 예를 들어 입력 배열과 가중치가 다음과 같다고 가정합니다.
#
# ```text
# 입력 배열:  [10, 20, 30, 40, 50]
# 가중치:     [0.6, 0.3, 0.1]
# ```
#
# 각 이동 결과에 가중치를 곱하면 다음과 같습니다.
#
# ```text
# 이동 없음:      [10, 20, 30, 40, 50] × 0.6
# 오른쪽 1칸:     [ 0, 10, 20, 30, 40] × 0.3
# 오른쪽 2칸:     [ 0,  0, 10, 20, 30] × 0.1
# ```
#
# 이 결과들을 같은 위치끼리 더하면 오른쪽 방향으로 이어지는 잔상 효과가 만들어집니다.
#
# #### CPU와 GPU의 처리 방식
#
# NumPy를 사용하는 CPU 코드에서는 출력 이미지의 픽셀을 하나씩 순차적으로 계산합니다.
#
# 반면 CuPy를 사용하는 GPU 코드에서는 이동 횟수를 순차적으로 반복하되, 각 반복에서 이미지 배열 전체에 대한 곱셈과 덧셈을 GPU에서 병렬로 처리합니다.
#
# * CPU
#   * 출력 픽셀을 하나씩 순차적으로 계산
# * GPU
#   * 이동 횟수는 순차적으로 반복
#   * 각 이동 단계의 전체 픽셀 연산은 병렬로 처리

# %% [markdown]
# **글로벌 변수 정의:**

# %%
IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

# %% [markdown]
# **이미지 불러오기:**

# %%
img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

plt.imshow(input_np, cmap="gray")
plt.axis("off");

# %% [markdown]
# **잔상 효과 함수 (CPU):**

# %%
def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """

    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):

            for shift in range(len(weights)):
                # x값 shift (오른쪽으로 이동했으므로 입력은 그만큼 왼쪽)
                input_x = output_x - shift

                if input_x < 0:
                    continue

                # shift한 픽셀을 누적하여 출력 배열에 저장
                output[output_y, output_x] += image[output_y, input_x] * weights[shift]

    return output

# %% [markdown]
# **잔상 효과 함수 (GPU):**

# %%
def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """

    height, width = image.shape
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(len(weights)):
        # 배열 slicing: 출력의 [shift:] 열은 입력의 [:width-shift] 열에서 이동해온 값
        input_region = image[:, :width - shift]
        output_region = output[:, shift:]

        # GPU가 병렬 연산할 output_region 계산 코드
        output_region += input_region * weights[shift]

    return output

# %% [markdown]
# **가중치 설정:**

# %%
# 이동 거리가 멀어질수록 가중치를 작게 설정
weights_np = np.linspace(
    1.0,
    0.05,
    TRAIL_LENGTH,
    dtype=np.float32
)

# 전체 가중치의 합을 1로 정규화
weights_np /= np.sum(weights_np)

# %% [markdown]
# **NumPy CPU 처리:**

# %%
cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start

# %% [markdown]
# **CuPy 배열 준비:**

# %%
upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start

# %% [markdown]
# **GPU 워밍업:**

# %%
warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(
    image=warmup_image,
    weights=warmup_weights
)

cp.cuda.Stream.null.synchronize()

# %% [markdown]
# **CuPy GPU 처리:**

# %%
cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start

# %% [markdown]
# **GPU 결과를 CPU로 가져오기:**

# %%
download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start

# %% [markdown]
# **결과 비교:**

# %%
compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

# %% [markdown]
# **결과 출력:**

# %%
print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")

# %% [markdown]
# **이미지 출력:**

# %%
plt.figure(figsize=(18,12))

plt.subplot(1,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

# %%
plt.figure(figsize=(12,8))
plt.imshow(output_cp_np, cmap="gray")
plt.axis("off");

# %% [markdown]
# #### $5)$ CuPy를 활용한 배열 연산 GPU 가속 2 – Convolution 연산

# %% [markdown]
# 이전 실습에서는 이미지 배열을 여러 픽셀만큼 이동하고, 각 배열에 서로 다른 가중치를 곱한 뒤 출력 배열에 누적하여 이미지 잔상 효과를 구현했습니다.
#
# 이 과정에서 이동 거리를 제어하는 for문은 Python에서 순차적으로 실행되지만, 각 반복에서 수행되는 이미지 배열 전체의 곱셈과 덧셈은 GPU에서 병렬로 처리된다는 점을 확인했습니다.
#
# 이번 실습에서는 이러한 배열 이동과 가중치 누적 방식을 2차원으로 확장하여 convolution 연산을 구현합니다.
#
# 이전에 정의한 `convolution2d()` 함수는 NumPy를 사용하여 출력 이미지의 픽셀을 하나씩 순차적으로 계산합니다.
#
# 각 출력 위치에서 커널 크기만큼 이미지 영역을 잘라낸 뒤, 이미지 영역과 커널을 원소별로 곱하고 그 결과를 모두 더하여 하나의 출력 픽셀을 생성합니다.
#
# ```text
# 출력 위치 하나 선택
# → 커널 크기의 이미지 영역 추출
# → 이미지 영역과 커널의 가중합 계산
# → 출력 픽셀 하나 저장
# ```
#
# 이 방식은 convolution의 원리를 직접 확인하기에는 적합하지만, 이미지의 높이와 너비만큼 Python 반복문을 실행해야 하므로 큰 이미지에서는 처리 시간이 오래 걸립니다.
#
# <br>
#
# 이번에는 이전에 정의한 `convolution2d()` 함수와 동일한 결과를 계산하는 CuPy 기반의 `convolution2d_cp()` 함수를 작성합니다.
#
# CuPy 버전에서는 출력 픽셀을 하나씩 반복하지 않고, 커널의 각 원소를 순차적으로 선택합니다. 그리고 모든 출력 위치에서 현재 커널 원소와 대응되는 입력 픽셀을 배열로 선택한 뒤, 해당 커널 가중치를 곱하여 출력 배열 전체에 누적합니다.
#
# ```text
# 커널 원소 하나 선택
# → 모든 출력 위치에서 대응되는 입력 픽셀 선택
# → 배열 전체에 현재 커널 가중치 적용
# → 출력 배열 전체에 누적
# ```
#
# 두 함수는 계산 순서가 다르지만, 최종적으로 각 출력 픽셀에는 이미지 영역과 커널의 동일한 가중합이 저장됩니다.
#
# <br>

# %% [markdown]
# **함수의 입력으로는 이전 실습과 동일한 이미지를 사용합니다:**

# %%
plt.imshow(input_np, cmap="gray")
plt.axis("off");

# %% [markdown]
# **Convolution 함수 정의 (CuPy):**

# %%
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """

    kernel_height, kernel_width = kernel.shape

    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    for ky in range(kernel_height):
        for kx in range(kernel_width):
            # 커널 원소 하나 선택
            weight = kernel[ky, kx]

            # 모든 출력 위치에서 대응되는 입력 픽셀 선택
            input_region = padded_image[
                ky : ky + output_height * stride : stride,
                kx : kx + output_width * stride : stride,
            ]

            # 배열 전체에 현재 커널 가중치 적용, 출력 배열 전체에 누적
            output += input_region * weight

    output += bias

    return output

# %% [markdown]
# **NumPy CPU 처리:**

# %%
# 이전 단계(E. CNN 개요)에서 정의한 함수와 필터를 이어서 사용합니다.
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width), dtype=np.float32)

    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride

            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            weighted_sum = np.sum(image_region * kernel)

            output[output_y, output_x] = weighted_sum + bias

    return output


def relu(x):
    return np.maximum(0, x)


# x축/y축 edge 검출 필터 (E 단계와 동일)
dx_edge_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
dy_edge_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
dx_bias = -0.2
dy_bias = -0.2

# 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
cpu_start = time.perf_counter()

output_dx_np = convolution2d(image=input_np, kernel=dx_edge_filter, bias=dx_bias, stride=1, padding=1)
output_dy_np = convolution2d(image=input_np, kernel=dy_edge_filter, bias=dy_bias, stride=1, padding=1)

cpu_end = time.perf_counter()
cpu_time = cpu_end - cpu_start

relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)

# %% [markdown]
# **CuPy GPU 처리:**

# %%
# 변환: NumPy 배열을 CuPy 배열로 변환
input_conv_cp = cp.asarray(input_np)
dx_edge_filter_cp = cp.asarray(dx_edge_filter)
dy_edge_filter_cp = cp.asarray(dy_edge_filter)

# 워밍업: 작은 배열로 CUDA Context/Kernel 준비
warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_kernel = cp.asarray(dx_edge_filter)

_ = convolution2d_cp(image=warmup_image, kernel=warmup_kernel, bias=dx_bias, stride=1, padding=1)

cp.cuda.Stream.null.synchronize()

# 처리: convolution을 GPU로 처리
cp.cuda.Stream.null.synchronize()
gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(image=input_conv_cp, kernel=dx_edge_filter_cp, bias=dx_bias, stride=1, padding=1)
output_dy_cp = convolution2d_cp(image=input_conv_cp, kernel=dy_edge_filter_cp, bias=dy_bias, stride=1, padding=1)

cp.cuda.Stream.null.synchronize()
gpu_end = time.perf_counter()
gpu_time = gpu_end - gpu_start

relu_dx_cp_np = cp.asnumpy(relu(output_dx_cp))
relu_dy_cp_np = cp.asnumpy(relu(output_dy_cp))

# %% [markdown]
# **결과 출력:**

# %%
# CPU/GPU 결과 비교 및 출력
compute_speedup = cpu_time / gpu_time
max_error_dx = float(np.max(np.abs(relu_dx_np - relu_dx_cp_np)))
max_error_dy = float(np.max(np.abs(relu_dy_np - relu_dy_cp_np)))

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"Processing time: {gpu_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Speedup: {compute_speedup:.2f}x")
print(f"Max error (dx): {max_error_dx:.6f}")
print(f"Max error (dy): {max_error_dy:.6f}")

# %% [markdown]
# **이미지 출력:**

# %%
plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# ---
