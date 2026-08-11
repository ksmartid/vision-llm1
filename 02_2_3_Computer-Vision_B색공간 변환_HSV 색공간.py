# -*- coding: utf-8 -*-
"""
Part B-3: 색공간 변환 - HSV 색공간
원본: 02_Computer-Vision.ipynb (cell 47~59)
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
# --- 이전 section 상태 재구성: 원본 cell 9 ---
image = skimage.data.astronaut()

plt.imshow(image)
plt.axis("off");

# %% [markdown]
# #### $3)$ HSV 색공간

# %% [markdown]
# HSV 색공간:
# * H (Hue): 색상
#   * 일반: $0^\circ$ ~ $360^\circ$
#   * OpenCV: $0^\circ$ ~ $179^\circ$ (uint8), $0.0^\circ$ ~ $360.0^\circ$ (float32)
# * S (Saturation): 채도
#   * 일반: 0 ~ 100%
#   * OpenCV: 0 ~ 255 (uint8), 0.0 ~ 1.0 (float32)
# * V (Value): 명도
#   * 일반: 0 ~ 100%
#   * OpenCV: 0 ~ 255 (uint8), 0.0 ~ 1.0 (float32)

# %% [markdown]
# 이제 HSV 색공간으로 변환하여 각 channel을 시각화해 봅시다.
#
# `cv2.split(이미지)` 함수를 사용하면 이미지의 각 channel을 분리할 수 있습니다.

# %%
# TODO: astronaut 이미지를 HSV 색공간으로 변환
# TODO: HSV 각 채널을 분리한 뒤 subplot에 시각화

image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(image_hsv)

plt.figure(figsize=(15,8))

plt.subplot(1,3,1), plt.imshow(h, cmap="gray"), plt.title('H Channel (Hue)')
plt.subplot(1,3,2), plt.imshow(s, cmap="gray"), plt.title('S Channel (Saturation)')
plt.subplot(1,3,3), plt.imshow(v, cmap="gray"), plt.title('V Channel (Value)')

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# HSV 각 채널을 분리하였다면 각 색상/채도/명도를 조절하여 이미지를 수정할 수 있습니다.

# %% [markdown]
# H (색상) 각도를 shift하여 색상을 변경해 봅시다.
#
# H의 범위가 (float32 기준) $0.0^\circ$ ~ $360.0^\circ$이기에 360도가 넘어가면 0으로 순환하도록 코드를 작성합시다.

# %%
image = skimage.data.astronaut()
image = image.astype(np.float32) / 255.

# TODO: 이미지를 HSV 색공간으로 변환
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# TODO: HSV 이미지의 각 채널 분리
h, s, v = cv2.split(image_hsv)

# TODO: H 각도를 증가시켜 색상 변경 (360도 초과하지 않도록 유의)
h_shifted = (h + 60.0) % 360.0

# TODO: 각 채널을 cv2.merge([h, s, v])로 결합
image_hsv_shifted = cv2.merge([h_shifted, s, v])

# TODO: 색공간을 다시 변환하여 이미지 출력
image_rgb_shifted = cv2.cvtColor(image_hsv_shifted, cv2.COLOR_HSV2RGB)

plt.imshow(image_rgb_shifted)
plt.axis("off");
plt.show()

# %% [markdown]
# S (채도) 값을 감쇠시켜 채도를 낮춰봅시다.
#
# S의 범위가 (float32 기준) 0.0 ~ 1.0 이기에 범위에서 벗어나지 않도록 코드를 작성합시다.

# %%
image = skimage.data.astronaut()
image = image.astype(np.float32) / 255.

# TODO: 이미지를 HSV 색공간으로 변환
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# TODO: HSV 이미지의 각 채널 분리
h, s, v = cv2.split(image_hsv)

# TODO: S 값에 배율을 적용하여 채도 감쇠
s_attenuated = s * 0.3

# TODO: np.clip(s, 0, 1)을 사용하여 최소/최대값 설정
s_attenuated = np.clip(s_attenuated, 0, 1)

# TODO: 각 채널을 cv2.merge([h, s, v])로 결합
image_hsv_attenuated = cv2.merge([h, s_attenuated, v])

# TODO: 색공간을 다시 변환하여 이미지 출력
image_rgb_attenuated = cv2.cvtColor(image_hsv_attenuated, cv2.COLOR_HSV2RGB)

plt.imshow(image_rgb_attenuated)
plt.axis("off");
plt.show()

# %% [markdown]
# V (명도) 값을 증폭시켜 명도를 높여봅시다.
#
# V의 범위가 (float32 기준) 0.0 ~ 1.0 이기에 범위에서 벗어나지 않도록 코드를 작성합시다.

# %%
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
plt.show()

# %% [markdown]
# 앞선 실습에서 확인했듯이, HSV 색공간은 색상/채도/명도가 독립적으로 분리되어 있습니다.
#
# 따라서 각 채널을 개별적으로 조절하여 이미지를 원하는 대로 변형할 수 있습니다.

# %% [markdown]
# ---
