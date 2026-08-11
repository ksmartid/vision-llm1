# -*- coding: utf-8 -*-
"""
Part A: 이미지 조정
원본: 02_Computer-Vision.ipynb (cell 7~31)
"""


# %% [markdown]
# (아래 셀들은 이전 섹션에서 정의된 상태를 이 파일 단독 실행을 위해 재구성한 것입니다.)


# %%
# --- 이전 section 상태 재구성: 원본 cell 4 ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data

# %% [markdown]
# ### A. 이미지 조정

# %% [markdown]
# 이어서 실습에 사용될 테스트 이미지를 불러와 확인해봅시다.

# %%
image = skimage.data.astronaut()

plt.imshow(image)
plt.axis("off");
plt.show()

# %% [markdown]
# 이미지는 결국 픽셀 '배열'입니다.
#
# 배열의 크기는 "H x W x C"이며, [row, col, channel]이라고도 표현합니다.
#
# 위 이미지는 RGB 이미지이기에 channel의 개수는 3입니다.

# %%
image.shape

# %% [markdown]
# 저희가 사용하는 astronaut 이미지는 세로와 가로가 512 픽셀이며 각 픽셀마다 3가지 값 (RGB)가 저장되어 있습니다.
#
# `image.shape`에서 알 수 있듯이, 이미지는 NumPy 배열입니다.
#
# 그럼 NumPy 방식으로 indexing 하여 픽셀 하나만 골라서 확인해볼까요?

# %%
image[0, 0]

# %% [markdown]
# 위 결과에서 볼 수 있듯이, 픽셀 하나에는 R,G,B 값이 들어있습니다.
#
# 각 픽셀의 데이터 타입은 `uint8` 입니다.
#
# 색상을 확인해봅시다.

# %%
rgb = np.array([154, 147, 151])

color = np.zeros((1, 1, 3), dtype=np.uint8)
color[0, 0] = rgb

plt.imshow(color)
plt.axis("off");
plt.show()

# %% [markdown]
# 이미지는 원점 (origin)이 좌측 상단입니다. row가 증가하면 아래로, column이 증가하면 우측으로 이동합니다.

# %%
image[50, 0]

# %%
rgb = np.array([45, 34, 93])

color = np.zeros((1, 1, 3), dtype=np.uint8)
color[0, 0] = rgb

plt.imshow(color)
plt.axis("off");
plt.show()

# %% [markdown]
# 이번에는 NumPy 방식으로 배열을 slicing 해보면 어떻게 될까요?

# %%
sliced = image[0:100, 0:50]

plt.imshow(sliced)
plt.axis("off");
plt.show()

# %% [markdown]
# NumPy slcing을 한다면 이미지를 crop하는 효과를 낼 수 있습니다.
#
# 다음으로는 OpenCV 기능을 사용하여 이미지를 회전/반전시켜 봅시다.

# %%
plt.figure(figsize=(15,6))

plt.subplot(1,4,1), plt.imshow(image)
plt.subplot(1,4,2), plt.imshow(cv2.rotate(image.copy(), cv2.ROTATE_90_CLOCKWISE))
plt.subplot(1,4,3), plt.imshow(cv2.rotate(image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE))
plt.subplot(1,4,4), plt.imshow(cv2.rotate(image.copy(), cv2.ROTATE_180))

for ax in plt.gcf().axes:
    ax.axis("off")

print(f"ROTATE_90_CLOCKWISE = {cv2.ROTATE_90_CLOCKWISE}")
print(f"ROTATE_90_COUNTERCLOCKWISE = {cv2.ROTATE_90_COUNTERCLOCKWISE}")
print(f"ROTATE_180 = {cv2.ROTATE_180}")
plt.show()

# %%
plt.figure(figsize=(15,6))

plt.subplot(1,4,1), plt.imshow(image)
plt.subplot(1,4,2), plt.imshow(cv2.flip(image.copy(), 1))
plt.subplot(1,4,3), plt.imshow(cv2.flip(image.copy(), 0))
plt.subplot(1,4,4), plt.imshow(cv2.flip(image.copy(), -1))

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# 이미지는 기본적으로 픽셀 값을 [0,255]으로 저장하며, 데이터 타입은 `uint8`입니다.

# %%
image.dtype

# %% [markdown]
# 데이터 타입을 [0,255] 8비트 정수에서 [0,1] 32비트 실수로 변환해봅시다.

# %%
plt.imshow(image.astype(np.float32))
plt.axis("off");
plt.show()

# %% [markdown]
# Matplotlib이 실수 값은 [0,1]인데 왜 그 밖의 숫자를 건네주는지 불평하는군요.
#
# 각 픽셀 값을 255로 나눠줍시다.

# %%
plt.imshow(image.astype(np.float32) / 255.)
plt.axis("off");
plt.show()

# %% [markdown]
# 이제 32비트 실수를 사용하여 이전과 같은 이미지를 볼 수 있습니다.

# %% [markdown]
# ---
