# -*- coding: utf-8 -*-
"""
Part E: Color Segmentation
원본: 02_Computer-Vision.ipynb (cell 119~130)
"""


# %% [markdown]
# (아래 셀들은 이전 섹션에서 정의된 상태를 이 파일 단독 실행을 위해 재구성한 것입니다.)


# %%
# --- 이전 section 상태 재구성: 원본 cell 4 ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data

# %%
# --- 이전 section 상태 재구성: 원본 cell 57 ---
image = skimage.data.astronaut()
image = image.astype(np.float32) / 255.

# TODO: 이미지를 HSV 색공간으로 변환
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# TODO: HSV 이미지의 각 채널 분리
h, s, v = cv2.split(image_hsv)

# TODO: V 값에 배율을 적용하여 명도 증폭
v_amplified = v * 1.5

# TODO: np.clip(v, 0, 1)을 사용하여 최소/최대값 설정
v_amplified = np.clip(v_amplified, 0, 1)

# TODO: 각 채널을 cv2.merge([h, s, v])로 결합
image_hsv_amplified = cv2.merge([h, s, v_amplified])

# TODO: 색공간을 다시 변환하여 이미지 출력
image_rgb_amplified = cv2.cvtColor(image_hsv_amplified, cv2.COLOR_HSV2RGB)

plt.imshow(image_rgb_amplified)
plt.axis("off");


# %%
# --- 이전 section 상태 재구성: 원본 cell 74 ---
from ipywidgets import interact

def change_kernel(kernel_size):
    # 홀수 kernel만 사용
    if kernel_size % 2 == 0:
        kernel_size += 1

    b_blur = cv2.blur(image, (kernel_size, kernel_size))
    g_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1), plt.imshow(b_blur), plt.title(f"Box Filter ({kernel_size}x{kernel_size})")
    plt.subplot(1,2,2), plt.imshow(g_blur), plt.title(f"Gaussian Filter ({kernel_size}x{kernel_size})")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.show()

interact(change_kernel, kernel_size=(1,101,2));

# %% [markdown]
# ### E. Color Segmentation

# %% [markdown]
# 색상 기반 분할 (Color Segmentation)이란?
# * 기법 :   색상 기반 분할 (Color Segmentation)
# * 설명 :   원하는 특정 색상만 분리/추출
# * 방법 :   색상 임계값 처리  (Color Thresholding)
# * 결과 :   이진 마스크  (Binary Mask)

# %% [markdown]
# 이미지 내 빨간색 color thresholding을 통해 이진 마스크 (binary mask)를 생성해봅시다.

# %%
motor = skimage.data.stereo_motorcycle()[0]

plt.imshow(motor)
plt.axis("off");
plt.show()

# %% [markdown]
# 우선 color thresholding을 위해 이미지를 HSV 색공간으로 변환해줍시다.
#
# 그 다음, `cv2.inRange(hsv_이미지, color_lower, color_upper)` 함수를 사용하여 mask를 생성합니다.
#
# 빨강은 $0^\circ$에 가깝기에 H값 $0^\circ$ ~ ($0$ + $\theta)^\circ$와 $(180-\theta)^\circ$ ~ $180^\circ$ 두 가지 mask를 bitwise OR로 연산합니다.

# %%
def update_mask(h_upper1=10, h_lower2=170):
    lower1 = np.array([0, 100, 50])
    upper1 = np.array([h_upper1, 255, 255])
    lower2 = np.array([h_lower2, 100, 50])
    upper2 = np.array([180, 255, 255])

    # 이미지를 HSV 색공간으로 변환
    motor_hsv = cv2.cvtColor(motor, cv2.COLOR_RGB2HSV)

    # Mask 생성
    mask1 = cv2.inRange(motor_hsv, lower1, upper1)
    mask2 = cv2.inRange(motor_hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    plt.figure(figsize=(16,8))

    plt.subplot(1,2,1), plt.imshow(motor), plt.title("Original")
    plt.subplot(1,2,2), plt.imshow(mask, cmap="gray"), plt.title(f"H: 0~{h_upper1}, {h_lower2}~180")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.show()

# 슬라이더로 upper/lower 값 추측
interact(update_mask, h_upper1=(0, 30, 1), h_lower2=(150, 180, 1));
plt.show()

# %% [markdown]
# 선택한 값을 아래 코드에 적용

# %%
# TODO: 빈칸에 선택한 값 지정
red_lower1 = np.array([0, 100, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 100, 50])
red_upper2 = np.array([180, 255, 255])

# TODO: HSV로 색공간 변환
motor_hsv = cv2.cvtColor(motor, cv2.COLOR_RGB2HSV)

# TODO: Mask 생성
mask_hsv1 = cv2.inRange(motor_hsv, red_lower1, red_upper1)
mask_hsv2 = cv2.inRange(motor_hsv, red_lower2, red_upper2)
mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(motor), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# Binary Mask에 노이즈가 너무 많습니다.
#
# 노이즈를 제거하기 위해 Low-pass Filter (blur)를 적용합시다.

# %%
# TODO: Gaussian Filter를 적용하여 binary mask 생성 (커널 사이즈: 7x7)
motor_blur = cv2.GaussianBlur(motor, (7, 7), 0)
mask_blur_hsv = cv2.GaussianBlur(mask_hsv, (7, 7), 0)

plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1,2,2), plt.imshow(mask_blur_hsv, cmap="gray"), plt.title("Binary Mask (Blurred)")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# 너무 강하게 smoothing하면 정보 유실이 크기에 약하게만 smoothing filter를 적용합니다.

# %% [markdown]
# ---
