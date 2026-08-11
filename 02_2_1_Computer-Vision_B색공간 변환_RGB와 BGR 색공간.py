# -*- coding: utf-8 -*-
"""
Part B-1: 색공간 변환 - RGB와 BGR 색공간
원본: 02_Computer-Vision.ipynb (cell 32~40)
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
# ### B. 색공간 변환

# %% [markdown]
# 색공간 (Color Space)이란?
# * 색을 일정한 기준으로 표현하는 방식
# * 색공간마다 표현 가능한 색의 범위가 다름
# * 일관된 색 표현을 위해 색공간 통일
# * 색공간 변환을 통해 이미지 전처리 효율을 극대화

# %% [markdown]
# #### $1)$ RGB와 BGR 색공간

# %% [markdown]
# 기본적으로 Matplotlib은 RGB, OpenCV는 BGR 색공간을 사용합니다.
#
# 이번에는 OpenCV로 이미지를 불러옵시다.

# %%
img_bgr = cv2.imread("src/images/1665_Girl_with_a_Pearl_Earring.jpg")
img_bgr = img_bgr.astype(np.float32) / 255.

# %% [markdown]
# OpenCV는 `cv2.imread` 함수를 사용하여 로컬에 저장되어 있는 이미지를 불러올 수 있습니다.
#
# 이제 Matplotlib으로 이미지를 확인해봅시다.

# %%
plt.imshow(img_bgr)
plt.axis("off");
plt.show()

# %% [markdown]
# 무언가 확실히 잘못됐죠?
#
# 그 이유는 OpenCV는 기본적으로 BGR 순서로 이미지를 읽어오지만, Matplotlib은 RGB 순서 기준으로 이미지를 해석하기 때문입니다.
#
# 따라서 최종적으로 이미지를 렌더링할 시각화 도구의 색공간 기준으로 변환해 주어야 합니다.
#
# OpenCV의 `cv2.cvtColor` 함수를 사용하여 변환해봅시다.

# %%
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off");
plt.show()
