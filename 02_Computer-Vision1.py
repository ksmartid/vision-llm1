# -*- coding: utf-8 -*-
"""
Part 1: 이미지 조정 / 색공간 변환 / 필터링 / Edge Detection (A~D)
원본: 02_Computer-Vision.ipynb (cell 0~118)
"""

# %% [markdown]
# <h1 style="text-align: center;">Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템</h1>
#
# <br><br>
#
# <div style="text-align: right; color: gray; font-style: italic;">
# 강사 김규래&emsp;<br>
# kkr.kyurae.kim@gmail.com&emsp;
# </div><br>
#
# ---
# ---

# %% [markdown]
# ## 2. Computer Vision 기초
#
# 본 세션에서는 OpenCV를 활용한 기초 이미지 처리 기법과 Jetson 환경에서의 실시간 카메라 피드 처리 방법을 다룹니다
#
# 이번에 다룰 핵심 내용은 크게 3가지 단계로 나누어 진행합니다.
# * 이미지 전처리: 이미지 조정, 색공간 변환, 이미지 Enhancement, 필터링
# * 특징 추출: Color Segmentation, Contour 검출, Interest Point 추출
# * 객체 추적: SORT, Feature Matching

# %% [markdown]
# Python 기반의 Computer Vision 문제를 해결하기 위해서는 OpenCV, NumPy, Matplotlib, scikit-image와 같은 필수 라이브러리를 자유롭게 활용할 수 있어야 합니다.
#
# 먼저 JetPack에 기본으로 포함되지 않은 라이브러리를 설치합니다.
#
# 이때 주의할 점은 시스템 Python 환경과 가상환경의 라이브러리 버전이 충돌하지 않도록 관리하는 것입니다.<br>
# 특히 Jetson에서는 OpenCV와 같이 JetPack에서 제공하는 라이브러리가 하드웨어 가속 기능과 연동되어 있으므로, 기존 시스템 라이브러리를 유지하면서 필요한 패키지만 추가 설치하는 방식이 권장됩니다.

# %% [markdown]
# ```bash
# python -m pip install --upgrade pip setuptools wheel
# pip uninstall matplotlib-inline
# pip install "matplotlib-inline<0.2"
# pip install --no-deps \
#     scikit-image==0.19.3 \
#     imageio==2.19.3 \
#     tifffile==2022.5.4 \
#     PyWavelets==1.3.0 \
#     networkx==2.8.8
# ```

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data

# %% [markdown]
# 위 코드가 에러 없이 실행되었다면 설치가 잘 되었으니 다음으로 넘어가도 되겠습니다.

# %% [markdown]
# ---

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

plt.show()

print(f"ROTATE_90_CLOCKWISE = {cv2.ROTATE_90_CLOCKWISE}")
print(f"ROTATE_90_COUNTERCLOCKWISE = {cv2.ROTATE_90_COUNTERCLOCKWISE}")
print(f"ROTATE_180 = {cv2.ROTATE_180}")

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

# %% [markdown]
# Matplotlib이 실수 값은 [0,1]인데 왜 그 밖의 숫자를 건네주는지 불평하는군요.
#
# 각 픽셀 값을 255로 나눠줍시다.

# %%
plt.imshow(image.astype(np.float32) / 255.)
plt.axis("off");

# %% [markdown]
# 이제 32비트 실수를 사용하여 이전과 같은 이미지를 볼 수 있습니다.

# %% [markdown]
# ---

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

# %% [markdown]
# #### $2)$ Grayscale

# %% [markdown]
# 동일한 방식으로 흑백 (grayscale)로 변환할 수 있습니다.

# %%
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray, cmap="gray")
plt.axis("off");

# %% [markdown]
# 우리의 우주비행사도 흑백으로 변환해 볼까요?

# %%
# TODO: astronaut 이미지를 흑백으로 변환하여 렌더링

# %% [markdown]
# 두 결과물이 다른 이유:
#
# $$
# \text{Gray} = 0.299R + 0.587G + 0.114B
# $$

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

image_hsv = ...
h, s, v = cv2.split(image_hsv)

plt.figure(figsize=(15,8))

plt.subplot(...), plt.imshow(...), plt.title('H Channel (Hue)')
plt.subplot(...), plt.imshow(...), plt.title('S Channel (Saturation)')
plt.subplot(...), plt.imshow(...), plt.title('V Channel (Value)')

for ax in plt.gcf().axes:
    ax.axis("off")

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
# TODO: HSV 이미지의 각 채널 분리
# TODO: H 각도를 증가시켜 색상 변경 (360도 초과하지 않도록 유의)
# TODO: 각 채널을 cv2.merge([h, s, v])로 결합
# TODO: 색공간을 다시 변환하여 이미지 출력

# %% [markdown]
# S (채도) 값을 감쇠시켜 채도를 낮춰봅시다.
#
# S의 범위가 (float32 기준) 0.0 ~ 1.0 이기에 범위에서 벗어나지 않도록 코드를 작성합시다.

# %%
image = skimage.data.astronaut()
image = image.astype(np.float32) / 255.

# TODO: 이미지를 HSV 색공간으로 변환
# TODO: HSV 이미지의 각 채널 분리
# TODO: S 값에 배율을 적용하여 채도 감쇠
# TODO: np.clip(s, 0, 1)을 사용하여 최소/최대값 설정
# TODO: 각 채널을 cv2.merge([h, s, v])로 결합
# TODO: 색공간을 다시 변환하여 이미지 출력

# %% [markdown]
# V (명도) 값을 증폭시켜 명도를 높여봅시다.
#
# V의 범위가 (float32 기준) 0.0 ~ 1.0 이기에 범위에서 벗어나지 않도록 코드를 작성합시다.

# %%
image = skimage.data.astronaut()
image = image.astype(np.float32) / 255.

# TODO: 이미지를 HSV 색공간으로 변환
# TODO: HSV 이미지의 각 채널 분리
# TODO: V 값에 배율을 적용하여 명도 증폭
# TODO: np.clip(v, 0, 1)을 사용하여 최소/최대값 설정
# TODO: 각 채널을 cv2.merge([h, s, v])로 결합
# TODO: 색공간을 다시 변환하여 이미지 출력

# %% [markdown]
# 앞선 실습에서 확인했듯이, HSV 색공간은 색상/채도/명도가 독립적으로 분리되어 있습니다.
#
# 따라서 각 채널을 개별적으로 조절하여 이미지를 원하는 대로 변형할 수 있습니다.

# %% [markdown]
# ---

# %% [markdown]
# ### C. 필터링

# %% [markdown]
# 필터링 (Filtering)이란?
# * 불필요한 성분 (노이즈)을 제거하고 원하는 특징만 선별하는 이미지 전처리 과정
# * Kernel을 통한 합성곱 (Convolution) 연산 수행

# %% [markdown]
# 필터링의 종류
# * Smoothing: 이미지에서 노이즈를 제거하여 부드럽게 만드는 과정 (blur)
#   * Box Filter (박스 블러)
#   * Gaussian Filter (가우시안 블러)
# * Sharpening: 이미지에서 세부 디테일을 강조하여 선명하게 만드는 과정

# %% [markdown]
# #### $1)$ Box Filter

# %% [markdown]
# OpenCV의 `cv2.blur` 함수를 사용해 Box Filter를 적용해봅시다.

# %%
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

# %% [markdown]
# Box Filter vs. Gaussian Filter 비교

# %%
plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(box_blur), plt.title('Box Filter')
plt.subplot(1,2,2), plt.imshow(gaussian_blur), plt.title('Gaussian Filter')

for ax in plt.gcf().axes:
    ax.axis("off")

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

# %% [markdown]
# #### $3)$ Sharpening Filter

# %% [markdown]
# 다음으로는 sharpening filter를 적용해봅시다.
#
# 안타깝게도 OpenCV에 `cv2.blur`, `cv2.GaussianBlur`와 같은 sharpening 함수가 존재하지 않습니다.
#
# 그러므로 NumPy 배열을 사용하여 kernel을 직접 만들어 `cv2.filter2D` 함수를 통해 Sharpening Filter를 적용해봅시다.

# %% [markdown]
# `cv2.filter2D(src, ddepth, kernel)`
# * `src`: 원본 이미지
# * `ddepth`: 데이터 타입 (-1은 원본 이미지와 동일한 데이터 타입)
# * `kernel`: 커널 배열

# %%
sharp_kernel = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32)

sharpened = cv2.filter2D(image, cv2.CV_32F, sharp_kernel)
sharpened = np.clip(sharpened, 0, 1)

plt.imshow(sharpened)
plt.axis("off");

# %% [markdown]
# Sharpening Kernel을 변경하여 이미지를 더욱 선명하게 만들어봅시다.

# %%
# TODO: 기존 sharpening kernel 포함하여 총 3가지 kernel을 적용한 결과를 원본 이미지와 함께 비교하는 subplot 생성

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

# %% [markdown]
# 테두리 부근에만 영향이 미치므로 전반적인 차이는 크지 않지만, `cv2.BORDER_CONSTANT` 방식은 결과가 명확하게 다른 것을 확인할 수 있습니다.

# %% [markdown]
# ---

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

# %%
# TODO: 이미지를 grayscale로 변환
sky_img_gray = ...

# TODO: 이미지를 float32로 변환
sky_img_gray = ...

# TODO: Gaussian Filter 적용 (Kernel 사이즈는 55x55)
sky_blur = ...

# TODO: cv2.subtract를 사용하여 원본에서 저주파 성분 차감 (High-pass Filter)
sky_hf = ...

# TODO: 고주파 이미지 확인 (cmap 설정)
plt.figure(figsize=(5,10))
plt.imshow(...)
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
unsharp_mask_applied = ...

# TODO: subplot 3개 (원본 흑백 이미지, 고주파 이미지, unsharp mask 적용된 이미지) 생성
plt.figure(figsize=(16,8))
plt.subplot(...), plt.imshow(...), plt.title("Original")
plt.subplot(...), plt.imshow(...), plt.title("High Frequency")
plt.subplot(...), plt.imshow(...), plt.title("Unsharp Mask Applied")

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# #### $3)$ Sobel Filter

# %% [markdown]
# Sobel Filter:
# * 이미지의 미분 (밝기 변화율)을 이용하여 밝기 변화가 큰 edge 검출
# * 미분 커널 ($G_x$, $G_y$)을 이용해 수평/수직 방향의 gradient를 계산
# * 두 방향의 gradient를 조합하여 edge 방향과 강도 (magnitude)를 얻음

# %% [markdown]
# dx:
# * 세로 방향 edge 강도
# * Kernel이 오른쪽 방향 (x축 방향)으로 이동하며 edge 검출
# * `cv2.Sobel(src, ddepth, dx=1, dy=0, dst, ksize)` → `dx=1` 설정

# %% [markdown]
# dy:
# * 가로 방향 edge 강도
# * Kernel이 아래쪽 방향 (y축 방향)으로 이동하며 edge 검출
# * `cv2.Sobel(src, ddepth, dx=0, dy=1, dst, ksize)` → `dx=1` 설정

# %% [markdown]
# `cv2.Sobel(src, ddepth, dx, dy, dst, ksize)`
# * `src`: 원본 이미지
# * `ddepth`: 데이터 타입 (주로 `cv2.CV_64F`로 설정)
# * `dx`: x-방향 미분 차수 (세로 방향 edge 검출시 dx=1, dy=0 으로 설정)
# * `dy`: y-방향 미분 차수 (가로 방향 edge 검출시 dx=0, dy=1 으로 설정)
# * `dst`: 연산 결과를 저장할 출력 배열 (destination), Python에서는 주로 직접 지정하지 않음
# * `ksize`: 커널 크기, 주로 `dst`를 지정하지 않기에 `ksize=`처럼 argument 이름도 명시해야 함

# %% [markdown]
# Sobel Filter의 dx와 dy를 구해 시각화해 봅시다.

# %%
image = skimage.data.astronaut()

# TODO: RGB2GRAY/BGR2GRAY 둘 중 선택 고려하여 image를 grayscale로 변환 (skimage.data.astronaut은 RGB, cv2.imread는 BGR)
image_gray = ...

# TODO: image_gray를 float32로 변환
image_gray = ...

# TODO: image_gray에 Gaussian Filter 적용 (Kernel 크기 11x11)
image_blur = ...

# TODO: dx, dy 생성
sobel_x = cv2.Sobel(..., cv2.CV_64F, ..., ..., ksize=3)
sobel_y = cv2.Sobel(..., cv2.CV_64F, ..., ..., ksize=3)

# TODO: subplot 3개 (원본 흑백 이미지, dx, dy) 생성
plt.figure(figsize=(18,8))
plt.subplot(...), plt.imshow(...), plt.title("Original (grayscale)")
plt.subplot(...), plt.imshow(...), plt.title("dx")
plt.subplot(...), plt.imshow(...), plt.title("dy")

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# cv2.magnitude:
# * Sobel Filter의 edge 강도 (magnitude)
# * 가로/세로 edge가 아닌 전체 대각선 포함 edge 강도
# * 가로/세로 방향으로 찾은 edge를 하나로 뭉쳐서 찾은 전체 edge

# %% [markdown]
# $$magnitude = \sqrt{dx^2 + dy^2}$$

# %%
sobel_mag = cv2.magnitude(sobel_x, sobel_y)

plt.imshow(sobel_mag, cmap="gray")
plt.axis("off");

# %% [markdown]
# convertScaleAbs:
# * Magnitude (edge 강도)를 절대값으로 변환
# * 복잡한 결과를 일반 이미지 (0 ~ 255, 0.0 ~ 1.0) 형태로 변환

# %%
sobel_mag_abs = cv2.convertScaleAbs(sobel_mag)

plt.imshow(sobel_mag_abs, cmap="gray")
plt.axis("off");

# %% [markdown]
# Sobel Filter:
# * 장점
#   * 매우 빠른 속도
# * 단점
#   * 배경 노이즈에 취약
#   * 두꺼운 edge
#

# %% [markdown]
# #### $4)$ Canny Edge Detection

# %% [markdown]
# Sobel Filter의 단점을 보완하는 "Canny Edge Detection"
#
# 기존 Sobel Filter의 두꺼운 edge를 매우 얇고 정확한 edge로 검출하는 알고리즘/기법

# %% [markdown]
# Canny Edge Detection의 과정:
# 1. Gaussian Smoothing
#    * 노이즈 제거  (edge 검출에 치명적)
# 2. Sobel Filter
#    * Edge 후보 검출  (아직 너무 굵은 edge)
# 3. Non-maximum Suppression (비최대 억제, NMS)
#    * Gradient 방향 기준으로 local maximum만 유지  (나머지 0으로 설정)
# 4. Hysteresis Thresholding (히스테레시스 임계값 처리)
#    * Local maximum들을 이중 임계값으로 처리
#    * 중간값 (보류된 픽셀)은 “연결성” 추적

# %% [markdown]
# Canny Edge Detection을 구현해봅시다.

# %%
image = skimage.data.astronaut()

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image_gray = image_gray.astype(np.float32) / 255.

canny_edge = cv2.Canny(image_gray, 50, 150)

plt.imshow(canny_edge, cmap="gray")
plt.axis("off");

# %% [markdown]
# Sobel Filter와 동일한 방식으로 흑백 이미지를 float32로 변환하였더니 에러가 발생합니다.
#
# "`_src.depth() == CV_8U in function 'cv::Canny'`"
#
# `cv::Canny` 함수는 `CV_8U` 데이터 타입을 필요로 한다고 명시되어 있습니다.
#
# 그러므로 float32로 데이터 변환을 하면 안됩니다.

# %%
image = skimage.data.astronaut()

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# 해당 코드 제외: image_gray = image_gray.astype(np.float32) / 255.

canny_edge = cv2.Canny(image_gray, 50, 150)

plt.imshow(canny_edge, cmap="gray")
plt.axis("off");

# %% [markdown]
# 이제는 문제 없이 이미지가 렌더링됩니다.
#
# 확연히 Sobel Filter보다 edge가 더 얇고 정확해진 것을 확인할 수 있습니다.

# %%
plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(sobel_mag_abs, cmap="gray"), plt.title('Sobel Filter')
plt.subplot(1,2,2), plt.imshow(canny_edge, cmap="gray"), plt.title('Canny Edge Detection')

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# ---
