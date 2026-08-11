# -*- coding: utf-8 -*-
"""
Part C-4: 필터링 - Border Type
원본: 02_Computer-Vision.ipynb (cell 81~85)
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
# #### $4)$ Border Type

# %% [markdown]
# 필터링을 적용할 때 borderType을 지정하여 차이를 확인해봅시다.

# %%
# TODO: 각 borderType을 적용한 결과 이미지와 제목이 적혀있는 subplot 생성

kernel_size = (101, 101)

border_types = [
    (cv2.BORDER_CONSTANT, "CONSTANT"),
    (cv2.BORDER_REFLECT, "REFLECT"),
    (cv2.BORDER_REFLECT_101, "REFLECT_101"),
    (cv2.BORDER_REPLICATE, "REPLICATE"),
]

plt.figure(figsize=(18,8))

# TODO: enumerate과 for loop을 사용하여 효율적으로 subplot 생성
for idx, (border_type, label) in enumerate(border_types):
    blurred = cv2.GaussianBlur(image, kernel_size, 0, borderType=border_type)
    plt.subplot(2, 2, idx + 1)
    plt.imshow(blurred)
    plt.title(label)

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# 테두리 부근에만 영향이 미치므로 전반적인 차이는 크지 않지만, `cv2.BORDER_CONSTANT` 방식은 결과가 명확하게 다른 것을 확인할 수 있습니다.

# %% [markdown]
# ---
