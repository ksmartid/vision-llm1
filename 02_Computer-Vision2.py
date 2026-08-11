# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
from ipywidgets import interact

# %% [markdown]
# ### E. Color Segmentation

# %% [markdown]
# 색상 기반 분할 (Color Segmentation)이란?
# * 기법 :   색상 기반 분할 (Color Segmentation)
# * 설명 :   원하는 특정 색상만 분리/추출
# * 방법 :   색상 임계값 처리  (Color Thresholding)
# * 결과 :   이진 마스크  (Binary Mask)

# %% [markdown]
# 이미지 내 빨간색 color thresholding을 통해 이진 마스크 (binary mask)를 생성해봅시다.

# %%
motor = skimage.data.stereo_motorcycle()[0]

plt.imshow(motor)
plt.axis("off");

# %% [markdown]
# 우선 color thresholding을 위해 이미지를 HSV 색공간으로 변환해줍시다.
#
# 그 다음, `cv2.inRange(hsv_이미지, color_lower, color_upper)` 함수를 사용하여 mask를 생성합니다.
#
# 빨강은 $0^\circ$에 가깝기에 H값 $0^\circ$ ~ ($0$ + $\theta)^\circ$와 $(180-\theta)^\circ$ ~ $180^\circ$ 두 가지 mask를 bitwise OR로 연산합니다.

# %%
def update_mask(h_upper1=10, h_lower2=170):
    lower1 = np.array([0, 100, 50])
    upper1 = np.array([h_upper1, 255, 255])
    lower2 = np.array([h_lower2, 100, 50])
    upper2 = np.array([180, 255, 255])

    # 이미지를 HSV 색공간으로 변환
    motor_hsv = cv2.cvtColor(motor, cv2.COLOR_RGB2HSV)

    # Mask 생성
    mask1 = cv2.inRange(motor_hsv, lower1, upper1)
    mask2 = cv2.inRange(motor_hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    plt.figure(figsize=(16,8))

    plt.subplot(1,2,1), plt.imshow(motor), plt.title("Original")
    plt.subplot(1,2,2), plt.imshow(mask, cmap="gray"), plt.title(f"H: 0~{h_upper1}, {h_lower2}~180")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.show()

# 슬라이더로 upper/lower 값 추측
interact(update_mask, h_upper1=(0, 30, 1), h_lower2=(150, 180, 1));

# %% [markdown]
# 선택한 값을 아래 코드에 적용

# %%
# TODO: 빈칸에 선택한 값 지정
red_lower1 = np.array([0, 100, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 100, 50])
red_upper2 = np.array([180, 255, 255])

# TODO: HSV로 색공간 변환
motor_hsv = cv2.cvtColor(motor, cv2.COLOR_RGB2HSV)

# TODO: Mask 생성
mask_hsv1 = cv2.inRange(motor_hsv, red_lower1, red_upper1)
mask_hsv2 = cv2.inRange(motor_hsv, red_lower2, red_upper2)
mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(motor), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# Binary Mask에 노이즈가 너무 많습니다.
#
# 노이즈를 제거하기 위해 Low-pass Filter (blur)를 적용합시다.

# %%
# TODO: Gaussian Filter를 적용하여 binary mask 생성 (커널 사이즈: 7x7)
motor_blur = cv2.GaussianBlur(motor, (7, 7), 0)
mask_blur_hsv = cv2.GaussianBlur(mask_hsv, (7, 7), 0)

plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1,2,2), plt.imshow(mask_blur_hsv, cmap="gray"), plt.title("Binary Mask (Blurred)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 너무 강하게 smoothing하면 정보 유실이 크기에 약하게만 smoothing filter를 적용합니다.

# %% [markdown]
# ---

# %% [markdown]
# ### F. Morphological Operation

# %% [markdown]
# 그럼 남아있는 노이즈는 어떻게 처리할까요?

# %% [markdown]
# 모폴로지 연산 (Morphological Operation)이란?
# * 형태학적 연산
# * Mask의 영역을 확대/축소
# * Erosion(침식), Dilation(팽창), Opening(열기), Closing(닫기)

# %% [markdown]
# Erosion (침식)  vs.  Dilation (팽창):
# * Erosion (침식)
#   * Kernel 내 모든 픽셀이 1 (흰색)일 때만 중심 픽셀을 1 (흰색)으로 설정
#   * AND
#   * 흰색 영역 감소
#   * 배경의 노이즈 제거
# * Dilation (팽창)
#   * Kernel 내 픽셀 하나라도 1 (흰색)이면 중심 픽셀을 1 (흰색)으로 설정
#   * OR
#   * 흰색 영역 확장
#   * 내부의 검정 구멍과 내부의 끊어진 윤곽선 연결

# %% [markdown]
# Erosion (침식) vs. Dilation (팽창)

# %%
binmask_crop = cv2.imread('src/images/binmask_crop.PNG', cv2.IMREAD_GRAYSCALE)

binmask_crop_erode = cv2.erode(binmask_crop, None, iterations=2)
binmask_crop_dilate = cv2.dilate(binmask_crop, None, iterations=2)

plt.figure(figsize=(16,8))

plt.subplot(1,3,1), plt.imshow(binmask_crop, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(binmask_crop_erode, cmap="gray"), plt.title("Eroded")
plt.subplot(1,3,3), plt.imshow(binmask_crop_dilate, cmap="gray"), plt.title("Dilated")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# Opening (열기) vs. Closing (닫기)

# %%
# kernel = np.ones((3, 3))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

binmask_crop_opened = cv2.morphologyEx(binmask_crop, cv2.MORPH_OPEN, kernel)
binmask_crop_closed = cv2.morphologyEx(binmask_crop, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(16,8))

plt.subplot(1,3,1), plt.imshow(binmask_crop, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(binmask_crop_opened, cmap="gray"), plt.title("Opened")
plt.subplot(1,3,3), plt.imshow(binmask_crop_closed, cmap="gray"), plt.title("Closed")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 위에서 확인한 방식들을 활용하여 `skimage.data.stereo_motorcycle[0]` 이미지에서 깔끔하게 빨강 영역 검출을 시도해봅시다.

# %%
# TODO: 적절한 morphological opertaion을 사용하여 최적의 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

mask_hsv_opened = cv2.morphologyEx(mask_blur_hsv, cv2.MORPH_OPEN, kernel)
mask_hsv_final = cv2.morphologyEx(mask_hsv_opened, cv2.MORPH_CLOSE, kernel)

# TODO: subplot 3개 (원본 이진 마스크, blur 처리된 이진 마스크, 최종 마스크) 생성
plt.figure(figsize=(16,8))

plt.subplot(1,3,1), plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1,3,2), plt.imshow(mask_blur_hsv, cmap="gray"), plt.title("Binary Mask (Blurred)")
plt.subplot(1,3,3), plt.imshow(mask_hsv_final, cmap="gray"), plt.title("Final Binary Mask (Opened + Closed)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# ---

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
cat_result = cv2.bitwise_and(cat, cat, mask=cat_mask)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(cat), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(cat_mask, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1,3,3), plt.imshow(cat_result), plt.title("Masked Result")

for ax in plt.gcf().axes:
    ax.axis("off")

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

# %% [markdown]
# ### H. Contour Detection

# %% [markdown]
# 외곽선 (Contour)이란?
# * binary mask에서 연결된 영역의 외곽선
# * 연결된 영역이기에 적절한 morphological operation 중요

# %% [markdown]
# 우선 최종 생성된 농구공 mask에 morphological operation을 적용하여
# i) 미세한 노이즈를 없애주거나 (opening)
# ii) 빈 공간을 채워 끊어진 영역들을 연결시켜 줍시다 (closing).

# %% [markdown]
# 오토바이 이미지로 한번 contour detection을 시도해봅시다.
#
# 우선 이미지에 약한 blur를 적용한 뒤 binary mask를 생성합니다.
#
# 그 다음, morphological operation으로 mask를 한층 더 깔끔하게 수정합니다.

# %%
# Gaussin blur 적용
motor_blur = cv2.GaussianBlur(motor, (7, 7), 0)

# HSV로 변환
motor_blur_hsv = cv2.cvtColor(motor_blur, cv2.COLOR_RGB2HSV)

# 빨간색 추출하여 binary mask 생성
motor_mask1 = cv2.inRange(motor_blur_hsv, red_lower1, red_upper1)
motor_mask2 = cv2.inRange(motor_blur_hsv, red_lower2, red_upper2)
motor_mask = cv2.bitwise_or(motor_mask1, motor_mask2)

plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(motor), plt.title("Original Image")
plt.subplot(1,2,2), plt.imshow(motor_mask, cmap="gray"), plt.title("Binary Mask (Blurred)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %%
# kernel = np.ones((5, 5), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

motor_mask_opened = cv2.morphologyEx(motor_mask, cv2.MORPH_OPEN, kernel)
motor_mask_closed = cv2.morphologyEx(motor_mask, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(motor_mask, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1,3,2), plt.imshow(motor_mask_opened, cmap="gray"), plt.title("Mask after Opening")
plt.subplot(1,3,3), plt.imshow(motor_mask_closed, cmap="gray"), plt.title("Mask after Closing")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# Opening을 적용한 결과가 오토바이의 외곽선을 추출하기에 좋아보입니다.
#
# Closing은 우측 하단에 바닥 그림자가 더욱 커져 자칫 원치 않는 영역이 더욱 강조가 될 우려가 있습니다.
#
# Opening의 결과가 좋아 보이기에 `iterations` 파라미터를 사용하여 여러번 적용해봅시다.

# %%
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

motor_mask_opened1 = cv2.morphologyEx(motor_mask, cv2.MORPH_OPEN, kernel)
motor_mask_opened2 = cv2.morphologyEx(motor_mask, cv2.MORPH_OPEN, kernel, iterations=2)
motor_mask_opened3 = cv2.morphologyEx(motor_mask, cv2.MORPH_OPEN, kernel, iterations=3)

plt.figure(figsize=(16,8))
plt.subplot(2,3,2), plt.imshow(motor), plt.title("Original Image")
plt.subplot(2,3,4), plt.imshow(motor_mask_opened1, cmap="gray"), plt.title("iteration = 1")
plt.subplot(2,3,5), plt.imshow(motor_mask_opened2, cmap="gray"), plt.title("iteration = 2")
plt.subplot(2,3,6), plt.imshow(motor_mask_opened3, cmap="gray"), plt.title("iteration = 3")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 2번이 노이즈가 많이 제거되어 좋은 binary mask로 보입니다.
#
# 3번은 좌측에 오토바이 후미 영역이 너무 멀리 끊어져 있어 2번을 가지고 추가 작업이 효율적일 것 같습니다.
#
# 2번 결과물에 closing을 적용해봅시다.

# %%
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

motor_mask_closed1 = cv2.morphologyEx(motor_mask_opened2, cv2.MORPH_CLOSE, kernel)
motor_mask_closed2 = cv2.morphologyEx(motor_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=2)
motor_mask_closed3 = cv2.morphologyEx(motor_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=3)

plt.figure(figsize=(16,8))
plt.subplot(2,3,2), plt.imshow(motor_mask_opened2, cmap="gray"), plt.title("Opening x2")
plt.subplot(2,3,4), plt.imshow(motor_mask_closed1, cmap="gray"), plt.title("Opening x2 & Closing x1")
plt.subplot(2,3,5), plt.imshow(motor_mask_closed2, cmap="gray"), plt.title("Opening x2 & Closing x2")
plt.subplot(2,3,6), plt.imshow(motor_mask_closed3, cmap="gray"), plt.title("Opening x2 & Closing x3")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 3번이 깔끔하게 오토바이의 빨간 영역을 검출할 수 있을 것 같습니다.
#
# Opening 2회, Closing 3회를 적용하여 최종 binary mask를 생성하였습니다.
#
# 자, 이제 binary mask 영역의 contour를 추출해봅시다.

# %%
# 최종 binary mask
final_motor_mask = motor_mask_closed3

# Contour를 그릴 원본 이미지 copy
final_img = motor.copy()

# 컨투어 추출 (두 번째 반환값인 hierarchy는 바깥쪽 테두리만 검출하는 현재 방식상 무의미하기에 저장하지 않음)
contours, _ = cv2.findContours(
    final_motor_mask,        # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
)

if len(contours) > 0:
    # Binary mask를 반투명 초록색으로 overlay
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = final_motor_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (초록색 외곽선)
    cv2.drawContours(final_img, contours, -1, (0,255,0), 2)

plt.imshow(final_img)
plt.axis("off");

# %% [markdown]
# 이진 마스크(Binary Mask) 및 윤곽선 검출(Contour Detection)을 수행하여 이미지의 빨간색 영역(반투명 초록색)과 해당 외곽선(진한 초록색)을 추출하였습니다.
#
# 하지만 저희가 원하는건 오직 오토바이의 빨간 영역이죠.
#
# 그러기 위해서는 검출된 윤곽선 중 면적이 가장 크거나 둘레가 가장 긴 외곽선을 최종 선택합니다.
#
# 일반적으로 노이즈 및 불필요한 영역은 상대적으로 작은 면적을 형성하기 때문입니다.
#
# 이전 단계에서 Morphological Operation을 신중히 적용한 목적 역시 주 영역의 연속성을 확보하여 윤곽선 면적을 최대화하기 위함이었습니다.

# %%
# 최종 binary mask
final_motor_mask = motor_mask_closed3

# Contour를 그릴 원본 이미지 copy
final_img = motor.copy()

# 컨투어 추출 (두 번째 반환값인 hierarchy는 바깥쪽 테두리만 검출하는 현재 방식상 무의미하기에 저장하지 않음)
contours, _ = cv2.findContours(
    final_motor_mask,        # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
)

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    final_mask = np.zeros_like(final_motor_mask)
    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 초록색으로 overlay
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = final_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (초록색 외곽선)
    cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2)

plt.imshow(final_img)
plt.axis("off");

# %% [markdown]
# 이제 동일한 방법으로 농구공의 외곽선을 검출해봅시다.
#
# 우선 morphological operation부터 적용해봅시다.

# %%
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
ball_mask_closed = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Enhanced Mask")
plt.subplot(1,3,2), plt.imshow(ball_mask_opened, cmap="gray"), plt.title("Mask after Opening")
plt.subplot(1,3,3), plt.imshow(ball_mask_closed, cmap="gray"), plt.title("Mask after Closing")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 저희 binary mask에는 노이즈가 많기에 확실히 Opening을 거친 결과가 훨씬 깔끔해진 것을 볼 수 있습니다.

# %% [markdown]
# Morphological operation부터 외곽선 검출, 그리고 외곽선 선택까지 해봅시다.

# %%
# TODO: Morphological operation을 적용하여 최종 binary mask 생성
ball_mask_opened_closed = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

ball_mask_dilated1 = cv2.dilate(ball_mask_opened_closed, kernel, iterations=1)
ball_mask_dilated2 = cv2.dilate(ball_mask_opened_closed, kernel, iterations=2)
ball_mask_dilated3 = cv2.dilate(ball_mask_opened_closed, kernel, iterations=3)

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    ball_mask_dilated3,       # binary mask
    cv2.RETR_EXTERNAL,        # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE   # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
)

# TODO: 최종 농구공 영역 및 외곽선 선택
if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(ball_mask_dilated1, cmap="gray"), plt.title("Dilated x1")
plt.subplot(1,3,2), plt.imshow(ball_mask_dilated2, cmap="gray"), plt.title("Dilated x2")
plt.subplot(1,3,3), plt.imshow(ball_mask_dilated3, cmap="gray"), plt.title("Dilated x3 (Final)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 이제 공의 중심 및 반지름을 찾아 그려봅시다.

# %% [markdown]
# 최소 외접원 (Minimum Enclosing Circle):
#
# “외곽선의 모든 점들을 포함하는 가장 작은 원”

# %%
# 최종 binary mask
final_ball_mask = ball_mask_dilated3

# Contour 및 외접원을 그릴 원본 이미지 copy
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # 최소 외접원 중심 및 반지름
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)

    # 파란색 원
    cv2.circle(final_img_enhanced, center_circle, radius, (0,0,255), 3)
    cv2.circle(final_img_enhanced, center_circle, 5, (0,0,255), -1)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1), plt.imshow(basketball_enhanced_rgb)
plt.subplot(1,2,2), plt.imshow(final_img_enhanced)

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 무게중심 (Centroid):
#
# "객체(Contour)의 픽셀 분포를 기반으로 계산한 기하학적 중심점(객체의 중심 좌표)"

# %% [markdown]
# 무게중심  =  픽셀 가중치 합 / 전체 면적
# $$
# C_x = \frac{M_{10}}{M_{00}}, \qquad
# C_y = \frac{M_{01}}{M_{00}}
# $$

# %%
# TODO: 최종 binary mask
final_ball_mask = ball_mask_dilated3

# Contour 및 무게중심 기준 원을 그릴 원본 이미지 copy
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # Moments를 통해 무게중심 (Centroid) 산출
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    centroid = (cX, cY)

    # 노란색 무게중심
    cv2.circle(final_img_enhanced, centroid, 5, (255,255,0), -1)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1), plt.imshow(basketball_enhanced_rgb)
plt.subplot(1,2,2), plt.imshow(final_img_enhanced)

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 무게중심을 산출하는 과정에서 반지름은 구할 수 없기에 최소 외접원 반지름과 함께 혼합하여 최종적인 공 영역을 예측할 수 있습니다.

# %%
# TODO: 최종 binary mask
final_ball_mask = ball_mask_dilated3

# Contour 및 무게중심 기준 원을 그릴 원본 이미지 copy
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # 최소 외접원 중심 및 반지름
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)

    # Moments를 통해 무게중심 (Centroid) 산출
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = center_circle

    centroid = (cX, cY)

    # 노란색 무게중심 기반 원
    cv2.circle(final_img_enhanced, centroid, radius, (255,255,0), 3)
    cv2.circle(final_img_enhanced, centroid, 5, (255,255,0), -1)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1), plt.imshow(basketball_enhanced_rgb)
plt.subplot(1,2,2), plt.imshow(final_img_enhanced)

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# 최소 외접원 vs. 무게중심 최종 비교

# %%
# TODO: 최종 binary mask
final_ball_mask = ball_mask_dilated3

# Contour 및 무게중심 기준 원을 그릴 원본 이미지 copy
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # 최소 외접원 중심 및 반지름
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)

    # 파란색 최소 외접원
    cv2.circle(final_img_enhanced, center_circle, radius, (0,0,255), 3)
    cv2.circle(final_img_enhanced, center_circle, 5, (0,0,255), -1)

    # Moments를 통해 무게중심 (Centroid) 산출
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = center_circle

    centroid = (cX, cY)

    # 노란색 무게중심 기반 원
    cv2.circle(final_img_enhanced, centroid, radius, (255,255,0), 3)
    cv2.circle(final_img_enhanced, centroid, 5, (255,255,0), -1)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1), plt.imshow(basketball_enhanced_rgb)
plt.subplot(1,2,2), plt.imshow(final_img_enhanced)

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# %% [markdown]
# ---

# %% [markdown]
# ### I. Interest Point Extraction

# %% [markdown]
# 특징점 추출 (Interest Point Extraction)이란?
# * 이미지 내에서 정보량이 높은 주요 지점(특징점, Interest Point)을 찾는 과정
# * 주변과 구별되는 중요한 위치 탐색
# * 모서리, 코너 등 주변 픽셀과 다른 패턴을 가진 위치를 검출
# * 추출된 특징점을 기반으로 이미지 간 비교 및 객체 추적 수행 가능
# * 크기, 회전, 조명 변화에도 비교적 안정적인 추적 가능
# * 대표적인 방법으로는 Harris Corner, SIFT, ORB 등

# %% [markdown]
# Harris Corner Detection 과정:
# 1. Sobel Filter 적용 → dx (x축 기울기), dy (y축 기울기), dxx (x축 변화율), dyy (y축 변화율), dxy (x-y축 변화율 상관관계)
# 2. Gaussian Filter 적용 (dxx, dyy, dxy)
# 3. Harris Response 계산
# 4. Response 값을 Thresholding
# 5. Thresholding 결과에 Non-Maximum Suppression 적용
# 6. 최종 코너 검출
#

# %% [markdown]
# Harris Corner Detection을 적용해봅시다.
#
# 우선 이미지를 불러옵시다.

# %%
cam_img = skimage.data.camera()

plt.imshow(cam_img, cmap="gray")
plt.axis("off");

# %% [markdown]
# 그 다음, `cv2.Sobel`와 `cv2.GaussianBlur` 함수를 사용하여 Harris Response를 계산합니다:
# $$
# Harris(\hat{M}) = \det(\hat{M})-\alpha\mathrm{trace}^2(\hat{M}) \approx G(I_x^2)G(I_y^2)-G(I_xI_y)^2-\alpha[G(I_x^2)+G(I_y^2)]^2
# $$
# $G$: Gaussian Filter
# $I_x,I_y$: 이미지 미분값 (Sobel Filter)
#

# %%
def harris(im, k=int(3), alpha=0.05):  # k = Gaussian filter의 kernel 크기
    # Sobel Filter를 사용하여 x와 y축의 미분값 계산
    dx = cv2.Sobel(im, -1, dx=1, dy=0)
    dy = cv2.Sobel(im, -1, dx=0, dy=1)

    # x-x, x-y and y-y 방향 미분값을 Gaussian blur 적용
    dxx = cv2.GaussianBlur(dx**2, (k,k), sigmaX=-1)
    dyy = cv2.GaussianBlur(dy**2, (k,k), sigmaX=-1)
    dxy = cv2.GaussianBlur(dx*dy, (k,k), sigmaX=-1)

    # Response function 계산
    return dxx * dyy - dxy**2 - alpha * (dxx + dyy)**2

# %% [markdown]
# Harris Response를 heatmap으로 시각화하면:

# %%
har = harris(np.float32(cam_img), 11, 0.05)

plt.figure(figsize=(12,8))
plt.imshow(cam_img, cmap='gray')
plt.imshow(har, cmap='jet', alpha=0.75), plt.colorbar();

plt.show()

# %% [markdown]
# 이제 Harris Response 이미지에서 코너를 검출해봅시다.

# %%
# 이미지에서 local maxima 포인트들을 검출
def findLocalMaxima(im, threshold=50):
    # Thresholding
    points = np.argwhere(im > threshold)
    points = [(x,y) for y,x in points]

    # 주변 8개 픽셀들과 비교하여 local maxima 검출
    maxima = []
    for p in points:
        # 이미지의 가장자리는 스킵
        if p[0] == 0 or p[0] == im.shape[1]-1 or p[1] == 0 or p[1] == im.shape[0]-1:
           continue

        neighbors = im[p[1]-1:p[1]+2, p[0]-1:p[0]+2]
        if np.all(neighbors <= im[p[1],p[0]]):
            maxima.append(p)

    return np.array(maxima)

# %%
harris_points = findLocalMaxima(har, 2e9)

plt.figure(figsize=(8,8))

plt.imshow(cam_img, cmap='gray')
plt.scatter(harris_points[:,0], harris_points[:,1], c='r', s=10)
plt.axis("off");

plt.show()

# %% [markdown]
# ---
