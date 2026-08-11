# -*- coding: utf-8 -*-
"""
Part C-2: 필터링 - Gaussian Filter
원본: 02_Computer-Vision.ipynb (cell 66~74)
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
# --- 이전 section 상태 재구성: 원본 cell 65 ---
box_blur = cv2.blur(image, (11,11))

plt.imshow(box_blur)
plt.axis("off");

# %% [markdown]
# #### $2)$ Gaussian Filter

# %% [markdown]
# OpenCV의 `cv2.GaussianBlur` 함수를 사용해 Gaussian Filter를 적용해봅시다.

# %%
gaussian_blur = cv2.GaussianBlur(image, (11,11), 0)

plt.imshow(gaussian_blur)
plt.axis("off");
plt.show()

# %% [markdown]
# Box Filter vs. Gaussian Filter 비교

# %%
plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(box_blur), plt.title('Box Filter')
plt.subplot(1,2,2), plt.imshow(gaussian_blur), plt.title('Gaussian Filter')

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# 이번에는 Kernel 크기에 따른 변화를 확인해봅시다.
#
# 우선 Jupyter Notebook에서 위젯 (슬라이더)를 구현하기 위하여 `ipywidgets`를 설치해야 합니다.

# %% [markdown]
# ```bash
# pip install ipywidgets
# ```

# %% [markdown]
# 설치가 완료되었다면 아래 코드를 실행해봅시다.

# %%
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
plt.show()
