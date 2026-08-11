# -*- coding: utf-8 -*-
"""
Part D-1: Edge Detection - Residual Extraction
원본: 02_Computer-Vision.ipynb (cell 86~92)
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
# ### D. Edge Detection

# %% [markdown]
# 이미지 주파수 (Image Frequency):
# * 저주파 (Low Frequency)
#   * 하늘, 피부, 벽면, 부드러운 음영, 기본 형태
#   * 없애면 윤곽선 스케치 (음영/형태 사라짐)
# * 고주파 (High Frequency)
#   * 경계선, 무늬, 질감, 거친 표면, 노이즈
#   * 없애면 흐릿한 이미지 (질감/노이즈 사라짐)

# %% [markdown]
# 저역통과 필터 (Low-pass Filter) vs. 고역통과 필터 (High-pass Filter):
# * LPF
#   * 저주파만 통과 (고주파는 억제)  →  흐릿한 이미지
#   * Box Filter, Gaussian Filter
# * HPF
#   * 고주파만 통과 (저주파는 억제)  →  윤곽선 스케치
#   * Residual Extraction (Unsharp Masking), Sobel Filter, Canny Edge Detection

# %% [markdown]
# #### $1)$ Residual Extraction

# %% [markdown]
# Residual Extraction을 적용하여 High-pass Filter의 결과물인 윤곽선 스케치 이미지를 만들어봅시다.
#
# 우선 'sky.png' 이미지를 `cv2.imread` 함수로 불러와 grayscale로 만듭니다.
#
# 그 다음, Gaussian Filter를 통해 저주파 이미지를 생성합니다.
#
# 기존 이미지에서 저주파 이미지를 감산 (subtract)하여 고주파 이미지를 생성 (High-pass Filter 적용)합니다.
#
# 이미지 감산은 `cv2.subtract(이미지, 저주파_이미지)` 함수를 사용하면 됩니다.

# %%
img_path = 'src/images/sky.png'
sky_img = cv2.imread(img_path)

plt.figure(figsize=(5,10))

# TODO: 원본 이미지 확인
plt.imshow(cv2.cvtColor(sky_img, cv2.COLOR_BGR2RGB))
plt.axis("off");
plt.show()

# %%
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
plt.show()
