# -*- coding: utf-8 -*-
"""
Part C-3: 필터링 - Sharpening Filter
원본: 02_Computer-Vision.ipynb (cell 75~80)
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
# #### $3)$ Sharpening Filter

# %% [markdown]
# 다음으로는 sharpening filter를 적용해봅시다.
#
# 안타깝게도 OpenCV에 `cv2.blur`, `cv2.GaussianBlur`와 같은 sharpening 함수가 존재하지 않습니다.
#
# 그러므로 NumPy 배열을 사용하여 kernel을 직접 만들어 `cv2.filter2D` 함수를 통해 Sharpening Filter를 적용해봅시다.

# %% [markdown]
# `cv2.filter2D(src, ddepth, kernel)`
# * `src`: 원본 이미지
# * `ddepth`: 데이터 타입 (-1은 원본 이미지와 동일한 데이터 타입)
# * `kernel`: 커널 배열

# %%
sharp_kernel = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32)

sharpened = cv2.filter2D(image, cv2.CV_32F, sharp_kernel)
sharpened = np.clip(sharpened, 0, 1)

plt.imshow(sharpened)
plt.axis("off");
plt.show()

# %% [markdown]
# Sharpening Kernel을 변경하여 이미지를 더욱 선명하게 만들어봅시다.

# %%
# TODO: 기존 sharpening kernel 포함하여 총 3가지 kernel을 적용한 결과를 원본 이미지와 함께 비교하는 subplot 생성
sharp_kernel_weak = np.array([[0, -1, 0],
                               [-1, 3, -1],
                               [0, -1, 0]], dtype=np.float32)

sharp_kernel_strong = np.array([[-1, -1, -1],
                                 [-1, 9, -1],
                                 [-1, -1, -1]], dtype=np.float32)

sharpened_weak = np.clip(cv2.filter2D(image, cv2.CV_32F, sharp_kernel_weak), 0, 1)
sharpened_strong = np.clip(cv2.filter2D(image, cv2.CV_32F, sharp_kernel_strong), 0, 1)

plt.figure(figsize=(18,5))
plt.subplot(1,4,1), plt.imshow(image), plt.title("Original")
plt.subplot(1,4,2), plt.imshow(sharpened_weak), plt.title("Weak Sharpen")
plt.subplot(1,4,3), plt.imshow(sharpened), plt.title("Sharpen")
plt.subplot(1,4,4), plt.imshow(sharpened_strong), plt.title("Strong Sharpen")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()
