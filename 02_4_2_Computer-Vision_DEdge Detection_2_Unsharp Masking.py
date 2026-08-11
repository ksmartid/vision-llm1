# -*- coding: utf-8 -*-
"""
Part D-2: Edge Detection - Unsharp Masking
원본: 02_Computer-Vision.ipynb (cell 93~95)
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
# --- 이전 section 상태 재구성: 원본 cell 91 ---
img_path = 'src/images/sky.png'
sky_img = cv2.imread(img_path)

plt.figure(figsize=(5,10))

# TODO: 원본 이미지 확인
plt.imshow(cv2.cvtColor(sky_img, cv2.COLOR_BGR2RGB))
plt.axis("off");


# %%
# --- 이전 section 상태 재구성: 원본 cell 92 ---
# TODO: 이미지를 grayscale로 변환
sky_img_gray = cv2.cvtColor(sky_img, cv2.COLOR_BGR2GRAY)

# TODO: 이미지를 float32로 변환
sky_img_gray = sky_img_gray.astype(np.float32) / 255.

# TODO: Gaussian Filter 적용 (Kernel 사이즈는 55x55)
sky_blur = cv2.GaussianBlur(sky_img_gray, (55, 55), 0)

# TODO: cv2.subtract를 사용하여 원본에서 저주파 성분 차감 (High-pass Filter)
sky_hf = cv2.subtract(sky_img_gray, sky_blur)

# TODO: 고주파 이미지 확인 (cmap 설정)
plt.figure(figsize=(5,10))
plt.imshow(sky_hf, cmap="gray")
plt.axis("off");


# %% [markdown]
# #### $2)$ Unsharp Masking

# %% [markdown]
# Unsharp Masking을 적용하여 기존 이미지의 edge를 강조해봅시다.
#
# 위 과정에서 만든 고주파 이미지를 원본 흑백 이미지와 합칩니다.
#
# `cv2.add(원본_이미지, 고주파_이미지)` 함수를 사용하면 됩니다.

# %%
# TODO: cv2.add를 사용하여 원본과 고주파 이미지를 합산
unsharp_mask_applied = cv2.add(sky_img_gray, sky_hf)

# TODO: subplot 3개 (원본 흑백 이미지, 고주파 이미지, unsharp mask 적용된 이미지) 생성
plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(sky_img_gray, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(sky_hf, cmap="gray"), plt.title("High Frequency")
plt.subplot(1,3,3), plt.imshow(unsharp_mask_applied, cmap="gray"), plt.title("Unsharp Mask Applied")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()
