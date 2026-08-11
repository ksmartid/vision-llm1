# -*- coding: utf-8 -*-
"""
Part C-1: 필터링 - Box Filter
원본: 02_Computer-Vision.ipynb (cell 60~65)
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


# %% [markdown]
# ### C. 필터링

# %% [markdown]
# 필터링 (Filtering)이란?
# * 불필요한 성분 (노이즈)을 제거하고 원하는 특징만 선별하는 이미지 전처리 과정
# * Kernel을 통한 합성곱 (Convolution) 연산 수행

# %% [markdown]
# 필터링의 종류
# * Smoothing: 이미지에서 노이즈를 제거하여 부드럽게 만드는 과정 (blur)
#   * Box Filter (박스 블러)
#   * Gaussian Filter (가우시안 블러)
# * Sharpening: 이미지에서 세부 디테일을 강조하여 선명하게 만드는 과정

# %% [markdown]
# #### $1)$ Box Filter

# %% [markdown]
# OpenCV의 `cv2.blur` 함수를 사용해 Box Filter를 적용해봅시다.

# %%
box_blur = cv2.blur(image, (11,11))

plt.imshow(box_blur)
plt.axis("off");
plt.show()
