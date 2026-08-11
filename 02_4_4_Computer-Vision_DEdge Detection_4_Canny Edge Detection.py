# -*- coding: utf-8 -*-
"""
Part D-4: Edge Detection - Canny Edge Detection
원본: 02_Computer-Vision.ipynb (cell 109~118)
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
# --- 이전 section 상태 재구성: 원본 cell 102 ---
image = skimage.data.astronaut()

# TODO: RGB2GRAY/BGR2GRAY 둘 중 선택 고려하여 image를 grayscale로 변환 (skimage.data.astronaut은 RGB, cv2.imread는 BGR)
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# TODO: image_gray를 float32로 변환
image_gray = image_gray.astype(np.float32) / 255.

# TODO: image_gray에 Gaussian Filter 적용 (Kernel 크기 11x11)
image_blur = cv2.GaussianBlur(image_gray, (11, 11), 0)

# TODO: dx, dy 생성
sobel_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3)

# TODO: subplot 3개 (원본 흑백 이미지, dx, dy) 생성
plt.figure(figsize=(18,8))
plt.subplot(1,3,1), plt.imshow(image_gray, cmap="gray"), plt.title("Original (grayscale)")
plt.subplot(1,3,2), plt.imshow(sobel_x, cmap="gray"), plt.title("dx")
plt.subplot(1,3,3), plt.imshow(sobel_y, cmap="gray"), plt.title("dy")

for ax in plt.gcf().axes:
    ax.axis("off")


# %%
# --- 이전 section 상태 재구성: 원본 cell 105 ---
sobel_mag = cv2.magnitude(sobel_x, sobel_y)

plt.imshow(sobel_mag, cmap="gray")
plt.axis("off");

# %%
# --- 이전 section 상태 재구성: 원본 cell 107 ---
sobel_mag_abs = cv2.convertScaleAbs(sobel_mag)

plt.imshow(sobel_mag_abs, cmap="gray")
plt.axis("off");

# %% [markdown]
# #### $4)$ Canny Edge Detection

# %% [markdown]
# Sobel Filter의 단점을 보완하는 "Canny Edge Detection"
#
# 기존 Sobel Filter의 두꺼운 edge를 매우 얇고 정확한 edge로 검출하는 알고리즘/기법

# %% [markdown]
# Canny Edge Detection의 과정:
# 1. Gaussian Smoothing
#    * 노이즈 제거  (edge 검출에 치명적)
# 2. Sobel Filter
#    * Edge 후보 검출  (아직 너무 굵은 edge)
# 3. Non-maximum Suppression (비최대 억제, NMS)
#    * Gradient 방향 기준으로 local maximum만 유지  (나머지 0으로 설정)
# 4. Hysteresis Thresholding (히스테레시스 임계값 처리)
#    * Local maximum들을 이중 임계값으로 처리
#    * 중간값 (보류된 픽셀)은 “연결성” 추적

# %% [markdown]
# Canny Edge Detection을 구현해봅시다.

# %%
image = skimage.data.astronaut()

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image_gray = image_gray.astype(np.float32) / 255.

# 아래 셀은 원본 노트북에서 "일부러 에러를 발생시켜 보여주는" 예시입니다.
# 노트북에서는 이 셀만 에러난 채로 두고 다음 셀로 넘어가지만, 스크립트에서는
# 그대로 두면 여기서 전체 실행이 멈추므로 try/except로 감싸 에러 메시지만 보여주고 계속 진행합니다.
try:
    canny_edge = cv2.Canny(image_gray, 50, 150)
    plt.imshow(canny_edge, cmap="gray")
    plt.axis("off");
    plt.show()
except cv2.error as e:
    print("예상된 에러 (아래 markdown에서 설명):", e)

# %% [markdown]
# Sobel Filter와 동일한 방식으로 흑백 이미지를 float32로 변환하였더니 에러가 발생합니다.
#
# "`_src.depth() == CV_8U in function 'cv::Canny'`"
#
# `cv::Canny` 함수는 `CV_8U` 데이터 타입을 필요로 한다고 명시되어 있습니다.
#
# 그러므로 float32로 데이터 변환을 하면 안됩니다.

# %%
image = skimage.data.astronaut()

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# 해당 코드 제외: image_gray = image_gray.astype(np.float32) / 255.

canny_edge = cv2.Canny(image_gray, 50, 150)

plt.imshow(canny_edge, cmap="gray")
plt.axis("off");
plt.show()

# %% [markdown]
# 이제는 문제 없이 이미지가 렌더링됩니다.
#
# 확연히 Sobel Filter보다 edge가 더 얇고 정확해진 것을 확인할 수 있습니다.

# %%
plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(sobel_mag_abs, cmap="gray"), plt.title('Sobel Filter')
plt.subplot(1,2,2), plt.imshow(canny_edge, cmap="gray"), plt.title('Canny Edge Detection')

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# ---
