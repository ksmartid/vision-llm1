# -*- coding: utf-8 -*-
"""
Part H: Contour Detection
원본: 02_Computer-Vision.ipynb (cell 170~199)
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
# --- 이전 section 상태 재구성: 원본 cell 122 ---
motor = skimage.data.stereo_motorcycle()[0]

plt.imshow(motor)
plt.axis("off");

# %%
# --- 이전 section 상태 재구성: 원본 cell 126 ---
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


# %%
# --- 이전 section 상태 재구성: 원본 cell 145 ---
basketball = cv2.imread('src/images/basketball_crop.jpg')
basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

plt.imshow(basketball_rgb)
plt.axis("off");

# %%
# --- 이전 section 상태 재구성: 원본 cell 152 ---
lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

lower_orange_lab = np.array([ 70, 130,  70])
upper_orange_lab = np.array([180, 230, 185])

mask_lab = cv2.inRange(lab, lower_orange_lab, upper_orange_lab)

plt.imshow(mask_lab, cmap="gray")
plt.axis("off");

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

# %%
# --- 이전 section 상태 재구성: 원본 cell 147 ---
# TODO: 이미지를 HSV 색공간으로 변환
basketball_hsv = cv2.cvtColor(basketball_rgb, cv2.COLOR_RGB2HSV)

# TODO: lower_orange_hsv, upper_orange_hsv 배열 생성 (농구공 Hue는 대략 5~25, S와 V 범위는 시도)
lower_orange_hsv = np.array([5, 100, 100])
upper_orange_hsv = np.array([25, 255, 255])

# TODO: HSV mask 생성
mask_hsv = cv2.inRange(basketball_hsv, lower_orange_hsv, upper_orange_hsv)

plt.imshow(mask_hsv, cmap="gray"), plt.title("Binary Mask")
plt.axis("off");


# %%
# --- 이전 section 상태 재구성: 원본 cell 149 ---
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

# %%
# --- 이전 section 상태 재구성: 원본 cell 154 ---
result_hsv = cv2.bitwise_and(basketball_rgb, basketball_rgb, mask=mask_hsv)
result_lab = cv2.bitwise_and(basketball_rgb, basketball_rgb, mask=mask_lab)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(basketball_rgb), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(result_hsv), plt.title("HSV Mask Result")
plt.subplot(1,3,3), plt.imshow(result_lab), plt.title("LAB Mask Result")

for ax in plt.gcf().axes:
    ax.axis("off")

# %%
# --- 이전 section 상태 재구성: 원본 cell 167 ---
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
plt.show()

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
plt.show()

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
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ball_mask_dilated3 = cv2.dilate(ball_mask_opened, kernel, iterations=3)

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    ball_mask_dilated3,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# TODO: 최종 농구공 영역 및 외곽선 선택
final_img_enhanced_preview = basketball_enhanced_rgb.copy()
if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(final_img_enhanced_preview, [largest_contour], -1, (0,255,0), 2)

plt.figure(figsize=(8,8))
plt.imshow(final_img_enhanced_preview)
plt.axis("off");
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
