# -*- coding: utf-8 -*-
"""
Step 5/10: E. Convolutional Neural Network (CNN) 개요
원본: 03_DL-and-GPU.ipynb (단계별로 재구성)
"""

# %%
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)

# %% [markdown]
# ### E. Convolutional Neural Network (CNN) 개요

# %% [markdown]
# 합성곱 신경망 (CNN)이란?
# * 이미지처럼 공간적인 구조를 가진 데이터를 처리하기 위해 설계된 신경망
# * 이미지를 2차원 또는 3차원 형태로 유지하면서 주변 픽셀 사이의 패턴을 추출
# * 커널(Kernel)을 활용한 합성곱(Convolution) 연산을 통해 이미지의 특징을 추출
# * 합성곱의 결과로 특징 맵 (Feature Map)을 생성
# * 풀링(Pooling)을 통해 중요한 특징은 유지하면서 데이터 크기를 줄여 연산 효율을 높임
# * 추출된 특징을 바탕으로 이미지 분류, 객체 탐지, 얼굴 인식 등 다양한 컴퓨터 비전 분야에서 활용됨

# %% [markdown]
# 특징 맵 (Feature Map):
# * 합성곱(Convolution) 연산의 결과로 생성되는 특징 표현
# * 입력 이미지의 특정 특징을 강조하여 나타냄
# * 하나의 Kernel당 하나의 Feature Map이 생성됨
# * 생성된 Feature Map은 다음 층의 입력으로 사용되어 더 복잡한 특징을 학습
#
# 특징 맵 (Feature Map)의 값:
# * 해당 위치에서 Filter가 찾는 패턴이 얼마나 강하게 나타나는지를 의미
#   * 큰 양수  →  Filter 패턴과 유사한 특징이 강하게 존재
#   * 0에 가까운 값  →  특징이 거의 없음
#   * 큰 음수  →  Filter 패턴과 반대 방향의 특징이 존재
# * CNN에서는 합성곱 결과에 ReLU를 적용하는 경우가 많음
#   * 음수 값은 0으로 변환, 양수 값은 유지

# %% [markdown]
# ####  Feature Map 생성

# %% [markdown]
# 이전 세션에서 다룬 이미지 전처리 과정인 필터링(Filtering)과 마찬가지로, CNN에서도 이미지 위에서 격자 형태의 커널(Kernel)을 움직이며 합성곱을 계산합니다. 그 결과, 이미지의 핵심 특징이 담긴 Feature Map이 생성됩니다.
#
# 연산의 첫 과정은 기존 필터링과 동일합니다. 원하는 특징을 추출하기 위해 적절한 크기의 커널을 정의한 뒤, 이미지 위를 이동하며 합성곱을 계산합니다. 그 이후 CNN에서는 활성화 함수인 ReLU를 적용하여, 원치 않는 특성(음수 값)은 0으로 처리하고 유의미한 특성(양수 값)은 그대로 유지합니다.

# %% [markdown]
# 우선 필수 라이브러리를 import 합니다. `NumPy`와 `matplotlib`는 위 과정에서 이미 import 했으니, 이미지를 불러오기 위한 `OpenCV` 라이브러리를 import 합시다.

# %%
import os

import cv2

# %% [markdown]
# Feature Map 실습을 위한 이미지를 불러옵니다.

# %%
# 스크립트 파일 위치를 기준으로 절대 경로를 구성하여, 어느 작업 디렉터리에서
# 실행하더라도 이미지를 정상적으로 찾을 수 있도록 한다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(SCRIPT_DIR, "src/images/seagull.jpg")

img_bgr = cv2.imread(img_path)

if img_bgr is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 새 figure를 명시적으로 생성하여 이후에 그릴 흑백 이미지와 창이 겹치지 않도록 한다.
plt.figure()
plt.imshow(img_rgb)
plt.title("Original Image (RGB)")
plt.axis("off")

# %% [markdown]
# 오늘 추출할 특징은 edge이므로 색상 정보는 불필요합니다. 그러므로 이미지를 흑백으로 변환합시다.

# %%
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img_gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# %% [markdown]
# 이제, 2차원 이미지에 하나의 CNN 필터를 적용하는 함수를 정의합니다.

# %% [markdown]
# 필터를 적용하게 되면 입력 이미지의 크기, 커널의 크기, stride와 padding 값에 따라 최종 출력 이미지의 크기가 결정됩니다.
#
# $H: \hspace{4pt}$ 입력 이미지 높이
# $W: \hspace{4pt}$ 입력 이미지 너비
# $P: \hspace{4pt}$ Padding 픽셀 값
# $K: \hspace{4pt}$ 커널 크기
# $S: \hspace{4pt}$ Stride 픽셀 값
#
# $$
# height \hspace{4pt} = \hspace{4pt}
# \lfloor \frac{H+2P-K}{S} \rfloor \hspace{2pt} + \hspace{2pt} 1
# $$
#
# $$
# width \hspace{4pt} = \hspace{4pt}
# \lfloor \frac{W+2P-K}{S} \rfloor \hspace{2pt} + \hspace{2pt} 1
# $$

# %%
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용.

    Parameters
    ----------
    image : np.ndarray
        입력 이미지, shape = (height, width)
    kernel : np.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : np.ndarray
        편향까지 적용된 convolution 결과
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이 및 너비, P: Padding 픽셀 값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산한 크기로 출력 이미지 준비
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 커널을 움직이며 필터 적용
    for output_y in range(output_height):
        for output_x in range(output_width):
            # 커널을 적용할 이미지 영역의 좌측상단
            start_y = output_y * stride
            start_x = output_x * stride

            # 커널을 적용할 이미지 영역을 crop
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            # 필터를 적용하여 가중합 계산
            weighted_sum = np.sum(image_region * kernel)

            # 편향 적용
            output[output_y, output_x] = weighted_sum + bias

    return output

# %% [markdown]
# 이제 CNN의 가중치와 편향을 구해야 합니다. 하지만 이번 실습에서는 모델을 직접 학습시키는 대신, 커널을 통해 Edge를 추출한 Feature Map을 생성하고 확인해 보는 것이 목적이므로 가중치와 편향을 직접 설정하겠습니다.
#
# CNN에서의 가중치와 편향이 무엇을 의미할까요?
# * CNN의 가중치: 적용할 필터(커널) 각 칸의 값
# * CNN의 편향: 약한 특징을 무시하거나 증폭시키기 위한 값

# %% [markdown]
# 우선 CNN의 가중치를 설정합시다.
#
# 이전 세션의 Sobel Filter 과정에서 배운 것처럼, 가로와 세로 방향의 edge를 검출하기 위한 커널을 각각 정의해봅시다.

# %%
# x축 방향 edge (세로 선) 검출 필터
dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

# y축 방향 edge (가로 선) 검출 필터
dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

# %% [markdown]
# 그 다음, CNN의 편향도 설정해봅시다.
#
# 가로/세로 선 검출 시, 너무 약한 특징은 무시하도록 합시다.

# %%
# 너무 약한 특징 (0.2 이하)은 무시
dx_bias = -0.2
dy_bias = -0.2

# %% [markdown]
# 최종적으로, 이미지에 가중치(필터)와 편향을 적용하는 `convolution2d()` 함수와 앞서 정의한 활성화 함수(ReLU)를 적용하는 `relu()` 함수를 사용하여 Feature Map을 생성합니다.

# %%
# 딥러닝 모델에서 더 안정적인 계산을 위해 이미지를 실수(float32)로 변환
input_image = img_gray.astype(np.float32) / 255.

# 가중치와 편향을 적용하여 세로 edge를 검출한 Feature Map 생성
vertical_output = convolution2d(
    image=input_image,
    kernel=dx_edge_filter,
    bias=dx_bias,
    stride=1,
    padding=1
)

# 가중치와 편향을 적용하여 가로 edge를 검출한 Feature Map 생성
horizontal_output = convolution2d(
    image=input_image,
    kernel=dy_edge_filter,
    bias=dy_bias,
    stride=1,
    padding=1
)

# 활성화 함수 (ReLU) 적용
vertical_feature_map = relu(vertical_output)
horizontal_feature_map = relu(horizontal_output)

# %%
plt.figure(figsize=(15,8))

plt.subplot(2,3,1), plt.imshow(input_image, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(vertical_output, cmap="gray"), plt.title("Vertical: Convolution + Bias")
plt.subplot(2,3,3), plt.imshow(vertical_feature_map, cmap="gray"), plt.title("Vertical Feature Map + ReLU")
plt.subplot(2,3,4), plt.imshow(input_image, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(horizontal_output, cmap="gray"), plt.title("Horizontal: Convolution + Bias")
plt.subplot(2,3,6), plt.imshow(horizontal_feature_map, cmap="gray"), plt.title("Horizontal Feature Map + ReLU")

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# ReLU를 사용하면 필터가 적용된 결과에서 음수 값을 무시하게 됩니다.
# 가로/세로 edge 필터를 적용한 결과에서 부호(+/-)는 edge의 방향 정보를 담고 있습니다.
#
# 현재 적용한 필터를 예로 들어봅시다.
#
# $$
# \begin{bmatrix}
# -1 & 0 & 1 \\
# -1 & 0 & 1 \\
# -1 & 0 & 1
# \end{bmatrix}
# $$
#
# 이와 같은 경우, 양수 결과 값은 왼쪽이 어둡고 오른쪽이 밝다는 의미입니다. 음수 결과 값은 반대로 왼쪽이 밝고 오른쪽이 어둡다는 정보를 갖고 있습니다.
# ReLU 함수를 적용하면 오직 오른쪽이 왼쪽보다 급격히 밝아지는 영역만 검출하게 됩니다. 이런 결과는 손글씨 인식과 같이 정확한 획의 형태를 파악하는 경우에 유용하게 사용됩니다.

# %% [markdown]
# 반대 방향 edge를 추출한 Feature Map도 시각화해봅시다.

# %%
# 딥러닝 모델에서 더 안정적인 계산을 위해 이미지를 실수(float32)로 변환
input_image = img_gray.astype(np.float32) / 255.

# 필터의 방향을 반대로 설정
dx_reverse_filter = -dx_edge_filter
dy_reverse_filter = -dy_edge_filter

# 가중치와 편향을 적용하여 세로 edge를 검출한 Feature Map 생성
vertical_reverse_output = convolution2d(
    image=input_image,
    kernel=dx_reverse_filter,
    bias=dx_bias,
    stride=1,
    padding=1
)

# 가중치와 편향을 적용하여 가로 edge를 검출한 Feature Map 생성
horizontal_reverse_output = convolution2d(
    image=input_image,
    kernel=dy_reverse_filter,
    bias=dy_bias,
    stride=1,
    padding=1
)

# 활성화 함수 (ReLU) 적용
vertical_reverse_feature_map = relu(vertical_reverse_output)
horizontal_reverse_feature_map = relu(horizontal_reverse_output)

# %%
plt.figure(figsize=(15,8))

plt.subplot(2,3,1), plt.imshow(input_image, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(vertical_output, cmap="gray"), plt.title("Vertical: Convolution + Bias")
plt.subplot(2,3,3), plt.imshow(vertical_reverse_feature_map, cmap="gray"), plt.title("Vertical Feature Map + ReLU")
plt.subplot(2,3,4), plt.imshow(input_image, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(horizontal_output, cmap="gray"), plt.title("Horizontal: Convolution + Bias")
plt.subplot(2,3,6), plt.imshow(horizontal_reverse_feature_map, cmap="gray"), plt.title("Horizontal Feature Map + ReLU")

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# 비교:

# %%
plt.figure(figsize=(15,8))

plt.subplot(2,3,1), plt.imshow(input_image, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(vertical_feature_map, cmap="gray"), plt.title("Vertical Feature Map")
plt.subplot(2,3,3), plt.imshow(vertical_reverse_feature_map, cmap="gray"), plt.title("Vertical Feature Map (Reversed)")
plt.subplot(2,3,4), plt.imshow(input_image, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(horizontal_feature_map, cmap="gray"), plt.title("Horizontal Feature Map")
plt.subplot(2,3,6), plt.imshow(horizontal_reverse_feature_map, cmap="gray"), plt.title("Horizontal Feature Map (Reversed)")

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# 앞선 실습에서 확인한 방법으로, Smoothing 필터와 Sharpening 필터를 적용하여 Feature Map을 생성해봅시다.

# %%
# Smoothing/Sharpening 필터 정의 (이전 세션의 Box Filter, Sharpening Kernel과 동일)
smoothing_filter = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
], dtype=np.float32)

sharpening_filter = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
], dtype=np.float32)

# Smoothing/Sharpening 필터 정규화 (필터 합이 1이 되도록)
# Sharpening filter는 합이 이미 1이므로 정규화해도 값이 그대로 유지된다.
smoothing_filter = smoothing_filter / smoothing_filter.sum()
sharpening_filter = sharpening_filter / sharpening_filter.sum()

# Smoothing/Sharpening 편향 정의 (밝기 보정이 필요하지 않으므로 0)
smoothing_bias = 0.0
sharpening_bias = 0.0

# convolution2d 함수를 사용하여 가중치와 편향 적용
smoothing_output = convolution2d(
    image=input_image,
    kernel=smoothing_filter,
    bias=smoothing_bias,
    stride=1,
    padding=1,
)
sharpening_output = convolution2d(
    image=input_image,
    kernel=sharpening_filter,
    bias=sharpening_bias,
    stride=1,
    padding=1,
)

# 활성화 함수 적용 (음수 값 제거)
smoothing_feature_map = relu(smoothing_output)
sharpening_feature_map = relu(sharpening_output)


# 결과 시각화
plt.figure(figsize=(12,8))

plt.subplot(2,2,1), plt.imshow(input_image, cmap="gray"), plt.title("Input Image")
plt.subplot(2,2,2), plt.imshow(smoothing_output, cmap="gray", vmin=0, vmax=1), plt.title("Smoothing: Convolution + Bias")
plt.subplot(2,2,3), plt.imshow(input_image, cmap="gray"), plt.title("Input Image")
plt.subplot(2,2,4), plt.imshow(sharpening_output, cmap="gray", vmin=0, vmax=1), plt.title("Sharpening: Convolution + Bias")

for ax in plt.gcf().axes:
    ax.axis("off")

# %%
# 지금까지 생성한 모든 figure를 화면에 표시한다.
# (Jupyter 셀 단위 실행에서는 각 셀이 자동으로 그림을 표시하므로 생략해도 되지만,
#  일반 스크립트로 실행할 때는 plt.show()를 호출해야 창이 나타난다.)
plt.show()

# %% [markdown]
# ---
