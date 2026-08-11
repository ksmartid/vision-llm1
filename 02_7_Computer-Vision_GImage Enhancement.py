# -*- coding: utf-8 -*-
"""
Part G: Image Enhancement
원본: 02_Computer-Vision.ipynb (cell 142~169)
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
# --- 이전 section 상태 재구성: 원본 cell 74 ---
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

# %% [markdown]
# ### G. Image Enhancement

# %% [markdown]
# Image Enhancement란?
# * Color segmentation은 조명 오염에 취약
# * 대비를 강화하여 원하는 영역 추출
# * LAB 색공간 사용

# %% [markdown]
# 우선 조명 오염에 영향을 받은 이미지를 확인해봅시다.

# %%
basketball = cv2.imread('src/images/basketball_crop.jpg')
basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

plt.imshow(basketball_rgb)
plt.axis("off");
plt.show()

# %% [markdown]
# 해당 이미지에 이전과 동일한 방식으로 color segmentation을 적용하여 binary mask를 생성해봅시다.

# %%
# TODO: 이미지를 HSV 색공간으로 변환
basketball_hsv = cv2.cvtColor(basketball_rgb, cv2.COLOR_RGB2HSV)

# TODO: lower_orange_hsv, upper_orange_hsv 배열 생성 (농구공 Hue는 대략 5~25, S와 V 범위는 시도)
lower_orange_hsv = np.array([5, 100, 100])
upper_orange_hsv = np.array([25, 255, 255])

# TODO: HSV mask 생성
mask_hsv = cv2.inRange(basketball_hsv, lower_orange_hsv, upper_orange_hsv)

plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")
plt.axis("off");
plt.show()

# %% [markdown]
# 정확한 HSV 범위를 알 수 없으니, 슬라이더를 사용하여 시도해봅시다.

# %%
def update_mask(
    h_low, h_high,
    s_low, s_high,
    v_low, v_high
):
    lower_orange_hsv = np.array([h_low, s_low, v_low])
    upper_orange_hsv = np.array([h_high, s_high, v_high])

    mask_hsv = cv2.inRange(basketball_hsv, lower_orange_hsv, upper_orange_hsv)

    plt.figure(figsize=(12,8))

    plt.subplot(1,2,1), plt.imshow(basketball), plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2), plt.imshow(mask_hsv, cmap="gray")
    plt.title(f"H:{h_low}-{h_high}, S:{s_low}-{s_high}, V:{v_low}-{v_high}")
    plt.axis("off")

    plt.show()


interact(
    update_mask,
    h_low=(0, 179, 1),
    h_high=(0, 179, 1),
    s_low=(0, 255, 5),
    s_high=(0, 255, 5),
    v_low=(0, 255, 5),
    v_high=(0, 255, 5)
);
plt.show()

# %% [markdown]
# 조명 오염이 매우 심하여, 아무리 시도해도 농구공 영역을 추출하기가 어렵습니다.
#
# 그러므로 HSV 색공간 대신 LAB 색공간으로 변환하여 시도해봅시다.

# %% [markdown]
# LAB 색공간:
# * 대립색 이론 (Opponent-Process Theory)를 기반으로 한 색공간
# * L* (Lightness)
#   * 인간의 눈이 체감하는 밝기 (지각 밝기)
#   * 0 ~ 100 (어두움 ~ 밝음)
# * a*
#   * -128 ~ 127 (초록 ~ 빨강)
# * b*
#   * -128 ~ 127 (파랑 ~ 노랑)

# %%
lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

lower_orange_lab = np.array([ 70, 130,  70])
upper_orange_lab = np.array([180, 230, 185])

mask_lab = cv2.inRange(lab, lower_orange_lab, upper_orange_lab)

plt.imshow(mask_lab, cmap="gray")
plt.axis("off");
plt.show()

# %% [markdown]
# LAB 기반 color segmentation은 조명 오염에도 불구하고 농구공 영역이 어느 정도 추출되는 것을 확인했습니다.
#
# 원본 사진과 합쳐 확인해봅시다.

# %%
result_hsv = cv2.bitwise_and(basketball_rgb, basketball_rgb, mask=mask_hsv)
result_lab = cv2.bitwise_and(basketball_rgb, basketball_rgb, mask=mask_lab)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(basketball_rgb), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(result_hsv), plt.title("HSV Mask Result")
plt.subplot(1,3,3), plt.imshow(result_lab), plt.title("LAB Mask Result")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# 원하는 영역을 부분적으로 추출하였으나, 조금만 더 욕심내 볼까요?
#
# LAB 기반 color segmentation과 더불어 CLAHE 기법을 적용해봅시다.

# %% [markdown]
# CLAHE (Contrast Limited Adaptive Histogram Equalization):
# * “대비 제한 적응형 히스토그램 평활화”
# * 이미지를 작은 영역으로 나누어 국소적인 대비 향상
# * 노이즈가 과도하게 증폭되지 않도록 대비 상한선을 제한

# %% [markdown]
# CLAHE (Contrast Limited Adaptive Histogram Equalization) 발전 과정:
# 1. **Histogram Equalization**
#    * 이미지 전체 밝기 분포 (히스토그램) 분석
#    * 최소/최대 밝기에 맞춰 재분포
#    * 이미지 전체를 일관적으로 처리
#    * 특정 영역만 너무 밝거나 어두워지는 현상
# 2. **Adaptive** Histogram Equalization
#    * 이미지를 격자로 분리 후 개별적으로 평탄화
#    * 단색 구역의 노이즈 대비가 극대화되는 현상
# 3. **Contrast Limited** Adaptive Histogram Equalization
#    * 히스토그램의 높이 (빈도수)가 제한을 초과하면 잘라냄
#    * 단색 영역에서도 과도하지 않은 명암비

# %% [markdown]
# 이제 CLAHE를 적용하여 명암 강화를 해봅시다.

# %%
cat = skimage.data.chelsea()

plt.imshow(cat)
plt.axis("off");
plt.show()

# %% [markdown]
# 이 고양이의 주황색 털을 추출해 보려고 합니다.
#
# 하지만 문제가 있습니다. 옅은 주황색과 진한 주황색 털이 저희가 원하는 "주황색"과 너무나도 비슷합니다.
#
# 우선 이미지 명암 강화 없이 color segmentation을 확인해봅시다.

# %%
lower_orange_fur_hsv = np.array([12,  50,  50])
upper_orange_fur_hsv = np.array([38, 230, 230])

# TODO: 이미지를 HSV 색공간으로 변환
cat_hsv = cv2.cvtColor(cat, cv2.COLOR_RGB2HSV)

# TODO: 이진 마스크 생성
cat_mask = cv2.inRange(cat_hsv, lower_orange_fur_hsv, upper_orange_fur_hsv)

# TODO: 이미지 출력
plt.imshow(cat_mask, cmap="gray")
plt.axis("off");
plt.show()

# %% [markdown]
# 뭔가 디테일이 많이 부족한 모습입니다.
#
# 이번엔 CLAHE를 사용하여 대비를 강화해봅시다.

# %%
# TODO: 이미지를 LAB 색공간으로 변환
cat_lab = cv2.cvtColor(cat, cv2.COLOR_RGB2LAB)

# TODO: 각 채널을 분리
l, a, b = cv2.split(cat_lab)

# CLAHE 생성 (clip 제한 2.0, 타일 크기 8x8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# L (Lightness) 강화
enhanced_l = clahe.apply(l)
enhanced_lab = cv2.merge((enhanced_l, a, b))

# TODO: LAB에서 HSV 색공간으로 바로 변환을 할 수 없으므로, RGB로 변환 후 HSV로 변환
enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
enhanced_hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)

# TODO: 강화된 이미지를 사용하여 이진 마스크 생성 (강화 전과 동일한 lower/upper HSV range 사용)
enhanced_cat_mask = cv2.inRange(enhanced_hsv, lower_orange_fur_hsv, upper_orange_fur_hsv)

plt.figure(figsize=(16,10))
plt.subplot(2,2,1), plt.imshow(cat), plt.title("Original Image")
plt.subplot(2,2,2), plt.imshow(enhanced_rgb), plt.title("LAB (CLAHE) Enhanced Image")
plt.subplot(2,2,3), plt.imshow(cat_mask, cmap="gray"), plt.title("Binary Mask")
plt.subplot(2,2,4), plt.imshow(enhanced_cat_mask, cmap="gray"), plt.title("LAB (CLAHE) Enhanced Binary Mask")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# CLAHE 적용으로 명도 대비를 높이면, binary mask 생성 시 원하는 색상을 훨씬 더 명확하게 추출할 수 있습니다.
#
# 보다 더 정교한 결과를 위해서는 `skimage`의 `exposure.equalize_adapthist()` 함수를 사용하여 CLAHE를 적용할 수 있습니다.

# %%
from skimage import color, exposure

cat = skimage.data.chelsea()

lower_orange_fur_hsv = np.array([12,  50,  50])
upper_orange_fur_hsv = np.array([38, 230, 230])
lower_orange_fur_hsv = lower_orange_fur_hsv / 255.
upper_orange_fur_hsv = upper_orange_fur_hsv / 255.

hsv_cat = color.rgb2hsv(cat)

mask_normal = (
    (hsv_cat[:, :, 0] >= lower_orange_fur_hsv[0]) & (hsv_cat[:, :, 0] <= upper_orange_fur_hsv[0]) &
    (hsv_cat[:, :, 1] >= lower_orange_fur_hsv[1]) & (hsv_cat[:, :, 1] <= upper_orange_fur_hsv[1]) &
    (hsv_cat[:, :, 2] >= lower_orange_fur_hsv[2]) & (hsv_cat[:, :, 2] <= upper_orange_fur_hsv[2])
)

lab = color.rgb2lab(cat)

l_channel = lab[:, :, 0] / 100.0
l_enhanced = exposure.equalize_adapthist(
    l_channel,
    kernel_size=32,
    clip_limit=0.02
)
lab[:, :, 0] = l_enhanced * 100.0

rgb_enhanced = (color.lab2rgb(lab) * 255).astype(np.uint8)
hsv_enhanced = color.rgb2hsv(rgb_enhanced)

mask_enhanced = (
    (hsv_enhanced[:, :, 0] >= lower_orange_fur_hsv[0]) & (hsv_enhanced[:, :, 0] <= upper_orange_fur_hsv[0]) &
    (hsv_enhanced[:, :, 1] >= lower_orange_fur_hsv[1]) & (hsv_enhanced[:, :, 1] <= upper_orange_fur_hsv[1]) &
    (hsv_enhanced[:, :, 2] >= lower_orange_fur_hsv[2]) & (hsv_enhanced[:, :, 2] <= upper_orange_fur_hsv[2])
)

plt.figure(figsize=(16,10))
plt.subplot(2,2,1), plt.imshow(cat), plt.title("Original Image")
plt.subplot(2,2,2), plt.imshow(rgb_enhanced), plt.title("LAB (CLAHE) Enhanced Image")
plt.subplot(2,2,3), plt.imshow(mask_normal, cmap="gray"), plt.title("Binary Mask")
plt.subplot(2,2,4), plt.imshow(mask_enhanced, cmap="gray"), plt.title("LAB (CLAHE) Enhanced Binary Mask")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# 이제 CLAHE를 사용하여 이전에 시도했던 조명 오염이 강한 농구공 이미지를 강화해봅시다.

# %%
lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

basketball = cv2.imread('src/images/basketball_crop.jpg')

# TODO: LAB 색공간으로 변환
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

# TODO: 각 채널 분리
l, a, b = cv2.split(basketball_lab)

# TODO: CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# TODO: L 채널에 CLAHE 적용 후 강화된 이미지 생성
l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

# TODO: 강화된 이미지로 binary mask 생성
mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

# TODO: 강화된 이미지를 RGB로 변환 후 마스크를 통해 이미지 내 관심 영역만 추출
basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)
final_result = cv2.bitwise_and(basketball_enhanced_rgb, basketball_enhanced_rgb, mask=mask_enhanced)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(basketball_rgb), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(result_lab), plt.title("LAB Segmentation Result")
plt.subplot(1,3,3), plt.imshow(final_result), plt.title("LAB (CLAHE) Enhanced Segmentation Result")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# CLAHE를 적용하여 기존 LAB 색 공간 color segmentation보다 농구공 영역이 더욱 정확하게 추출되는 것을 확인했습니다.
#
# 하지만 여전히 농구공 외에 불필요한 영역도 함께 추출되는 한계가 있습니다.
#
# 그렇다면 오직 농구공 영역만 정밀하게 검출하려면 어떻게 해야 할까요?

# %% [markdown]
# ---
