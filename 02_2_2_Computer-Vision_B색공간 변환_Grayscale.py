# -*- coding: utf-8 -*-
"""
Part B-2: 색공간 변환 - Grayscale
원본: 02_Computer-Vision.ipynb (cell 41~46)
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
# --- 이전 section 상태 재구성: 원본 cell 36 ---
img_bgr = cv2.imread("src/images/1665_Girl_with_a_Pearl_Earring.jpg")
img_bgr = img_bgr.astype(np.float32) / 255.

# %%
# --- 이전 section 상태 재구성: 원본 cell 9 ---
image = skimage.data.astronaut()

plt.imshow(image)
plt.axis("off");

# %% [markdown]
# #### $2)$ Grayscale

# %% [markdown]
# 동일한 방식으로 흑백 (grayscale)로 변환할 수 있습니다.

# %%
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray, cmap="gray")
plt.axis("off");
plt.show()

# %% [markdown]
# 우리의 우주비행사도 흑백으로 변환해 볼까요?

# %%
image_gray_astronaut = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(image_gray_astronaut, cmap="gray")
plt.axis("off");
plt.show()

# %% [markdown]
# 두 결과물이 다른 이유:
#
# $$
# \text{Gray} = 0.299R + 0.587G + 0.114B
# $$
