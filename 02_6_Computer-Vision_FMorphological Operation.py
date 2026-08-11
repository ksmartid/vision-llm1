# -*- coding: utf-8 -*-
"""
Part F: Morphological Operation
원본: 02_Computer-Vision.ipynb (cell 131~141)
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
# --- 이전 section 상태 재구성: 원본 cell 122 ---
motor = skimage.data.stereo_motorcycle()[0]

plt.imshow(motor)
plt.axis("off");

# %%
# --- 이전 section 상태 재구성: 원본 cell 126 ---
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


# %%
# --- 이전 section 상태 재구성: 원본 cell 128 ---
# TODO: Gaussian Filter를 적용하여 binary mask 생성 (커널 사이즈: 7x7)
motor_blur = cv2.GaussianBlur(motor, (7, 7), 0)
mask_blur_hsv = cv2.GaussianBlur(mask_hsv, (7, 7), 0)

plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1,2,2), plt.imshow(mask_blur_hsv, cmap="gray"), plt.title("Binary Mask (Blurred)")

for ax in plt.gcf().axes:
    ax.axis("off")


# %% [markdown]
# ### F. Morphological Operation

# %% [markdown]
# 그럼 남아있는 노이즈는 어떻게 처리할까요?

# %% [markdown]
# 모폴로지 연산 (Morphological Operation)이란?
# * 형태학적 연산
# * Mask의 영역을 확대/축소
# * Erosion(침식), Dilation(팽창), Opening(열기), Closing(닫기)

# %% [markdown]
# Erosion (침식)  vs.  Dilation (팽창):
# * Erosion (침식)
#   * Kernel 내 모든 픽셀이 1 (흰색)일 때만 중심 픽셀을 1 (흰색)으로 설정
#   * AND
#   * 흰색 영역 감소
#   * 배경의 노이즈 제거
# * Dilation (팽창)
#   * Kernel 내 픽셀 하나라도 1 (흰색)이면 중심 픽셀을 1 (흰색)으로 설정
#   * OR
#   * 흰색 영역 확장
#   * 내부의 검정 구멍과 내부의 끊어진 윤곽선 연결

# %% [markdown]
# Erosion (침식) vs. Dilation (팽창)

# %%
binmask_crop = cv2.imread('src/images/binmask_crop.PNG', cv2.IMREAD_GRAYSCALE)

binmask_crop_erode = cv2.erode(binmask_crop, None, iterations=2)
binmask_crop_dilate = cv2.dilate(binmask_crop, None, iterations=2)

plt.figure(figsize=(16,8))

plt.subplot(1,3,1), plt.imshow(binmask_crop, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(binmask_crop_erode, cmap="gray"), plt.title("Eroded")
plt.subplot(1,3,3), plt.imshow(binmask_crop_dilate, cmap="gray"), plt.title("Dilated")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# Opening (열기) vs. Closing (닫기)

# %%
# kernel = np.ones((3, 3))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

binmask_crop_opened = cv2.morphologyEx(binmask_crop, cv2.MORPH_OPEN, kernel)
binmask_crop_closed = cv2.morphologyEx(binmask_crop, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(16,8))

plt.subplot(1,3,1), plt.imshow(binmask_crop, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(binmask_crop_opened, cmap="gray"), plt.title("Opened")
plt.subplot(1,3,3), plt.imshow(binmask_crop_closed, cmap="gray"), plt.title("Closed")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# 위에서 확인한 방식들을 활용하여 `skimage.data.stereo_motorcycle[0]` 이미지에서 깔끔하게 빨강 영역 검출을 시도해봅시다.

# %%
# TODO: 적절한 morphological opertaion을 사용하여 최적의 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_hsv_opened = cv2.morphologyEx(mask_blur_hsv, cv2.MORPH_OPEN, kernel)
mask_hsv_final = cv2.morphologyEx(mask_hsv_opened, cv2.MORPH_CLOSE, kernel)

# TODO: subplot 3개 (원본 이진 마스크, blur 처리된 이진 마스크, 최종 마스크) 생성
plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1,3,2), plt.imshow(mask_blur_hsv, cmap="gray"), plt.title("Binary Mask (Blurred)")
plt.subplot(1,3,3), plt.imshow(mask_hsv_final, cmap="gray"), plt.title("Final Binary Mask (Opened + Closed)")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# ---
