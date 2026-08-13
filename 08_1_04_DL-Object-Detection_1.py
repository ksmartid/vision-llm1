# -*- coding: utf-8 -*-
"""
Step 1/10: A. Bounding Box
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
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
# ## 4. 딥러닝 기반 Edge AI 객체 탐지 시스템

# %% [markdown]
# 객체 탐지(Object Detection)란?<br>
# "이미지나 영상에서 객체의 위치와 객체의 종류를 찾아내는 컴퓨터 비전 기술"
#
# * 이미지 또는 영상 속 객체를 자동으로 탐지
# * 객체의 위치(Bounding Box)와 종류(Class)를 예측
# * 위치 추정(Localization)과 이미지 분류(Image Classification)를 결합한 기술
# * 하나의 이미지에서 여러 객체를 동시에 인식 가능

# %% [markdown]
# Rule-based vs. Learning-based:
# * 규칙기반 (Rule-based) 객체 탐지
#   * 사람이 색상, 크기, 형태 등의 규칙을 직접 설정
#   * 학습 데이터가 필요하지 않음
#   * 환경이 달라지면 규칙을 다시 수정해야 함
#   * 조명, 배경, 객체의 크기와 방향 변화에 민감함
# * 학습기반 (Learning-based) 객체 탐지
#   * 객체의 특징을 모델이 데이터로부터 학습
#   * 많은 학습 데이터와 연산이 필요
#   * 복잡한 배경과 다양한 객체 변화에 대응 가능
#   * 내부 판단 과정을 직접 해석하기 어려움

# %% [markdown]
# ---

# %% [markdown]
# ### A. Bounding Box

# %% [markdown]
# Bounding Box (BBox)는 이미지나 영상의 프레임에서 탐지된 객체를 둘러싸는 직사각형 영역입니다.
#
# 기본적으로 대부분의 객체 탐지는 OpenCV 프레임워크에 기반을 두고있기에 OpenCV 좌표계 시스템을 사용합니다.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import skimage.data

# %%
image = skimage.data.astronaut()
plt.imshow(image)

h, w, c = image.shape

# 좌측 상단
circle0 = Circle((0, 0), radius=25, edgecolor='magenta', facecolor='magenta', linewidth=5)
plt.gca().add_patch(circle0)

# 우측 하단
circle1 = Circle((w, h), radius=25, edgecolor='cyan', facecolor='cyan', linewidth=5)
plt.gca().add_patch(circle1)

print(f"이미지 높이: {h}\n이미지 너비: {w}")

# %% [markdown]
# BBox의 표현 방식은 크게 네 가지가 있습니다:
# 1. (x1, y1, x2, y2)
# 2. (x, y, w, h)
# 3. (cx, cy, w, h)
# 4. (cx_norm, cy_norm, w_norm, h_norm)
#
# 각 표현 방식을 살펴보겠습니다.

# %% [markdown]
# 1. `bbox = (x1, y1, x2, y2)`
#     * `(x1, y1)`: 좌측 상단 좌표
#     * `(x2, y2)`: 우측 하단 좌표
#     * bbox의 너비: `x2 - x1`
#     * bbox의 높이: `y2 - y1`
#     * OpenCV로 박스를 그릴 때 편리: `cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)`

# %% [markdown]
# 2. `bbox = (x, y, w, h)`
#     * `(x, y)`: 좌측 상단 좌표
#     * `(w, h)`: bbox 크기
#     * 우측 하단 좌표: `(x + w, y + h)`
#     * OpenCV 일부 함수에서 이 방식을 사용 (`cv2.selectROI`, `cv2.boundingRect` 등)

# %% [markdown]
# 3. `bbox = (cx, cy, w, h)`
#     * `(cx, cy)`: bbox 중심 좌표
#     * `(w, h)`: bbox 크기
#     * 좌측 상단 좌표: `(int(cx - w/2), int(cy - h/2))`
#     * 우측 하단 좌표: `(int(cx + w/2), int(cy + h/2))`

# %% [markdown]
# 4. `bbox = (cx_norm, cy_norm, w_norm, h_norm)`
#     * `(cx, cy, w, h)`를 이미지 크기에 맞춰 0~1로 정규화
#     * `cx_norm = cx / image_width`
#     * `cy_norm = cy / image_height`
#     * `w_norm = w / image_width`
#     * `h_norm = h / image_height`

# %% [markdown]
# 간단한 `cv2.rectangle`을 사용해 1번 방식인 `(x1, y1, x2, y2)` BBox를 그려봅시다.

# %%
cup_img = skimage.data.coffee()

x1, y1 = 170, 16
x2, y2 = 410, 330

cv2.rectangle(cup_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.putText(cup_img, "Cup\n0.95", (x1-60, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

plt.imshow(cup_img);

# %% [markdown]
# 이 방식은 하드코딩으로 좌표를 직접 입력한 방식이기에 진정한 객체 탐지라고 볼 수 없습니다. 이번에는 직접 좌표를 설정하지 않고 BBox를 그리는 실제 객체 탐지를 구현해봅시다.
#
# 가장 간단한 규칙기반 객체 탐지 방법으로는 Color Segmentation이 있습니다. 사람이 원하는 객체에 맞춰 직접 색상 범위를 지정하여 탐지합니다. Color Segmentation의 결과로 Binary Mask가 생성이 되고, 마스크를 통해 Contour를 추출할 수 있습니다.
#
# 규칙기반 객체 탐지로 BBox를 그리기 위해서는 Contour가 필요합니다.
#
# 우선, 이전 Computer Vision 실습에서 연습한 방식과 동일하게 Color Segmentation을 통해 Contour를 추출해봅시다.

# %%
apple = cv2.imread("src/images/apple.png")
apple_rgb = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,8))
plt.imshow(apple_rgb)
plt.axis("off");

# %%
# Color Segmentation을 통해 Binary Mask 생성

# 사과 빨강 HSV 범위
red_lower1 = np.array([0, 100, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 100, 50])
red_upper2 = np.array([180, 255, 255])

# Gaussian Blur 적용
apple_blur = cv2.GaussianBlur(apple, (7, 7), 0)

# Binary Mask 생성
apple_hsv = cv2.cvtColor(apple_blur, cv2.COLOR_BGR2HSV)
apple_mask1 = cv2.inRange(apple_hsv, red_lower1, red_upper1)
apple_mask2 = cv2.inRange(apple_hsv, red_lower2, red_upper2)
apple_mask = cv2.bitwise_or(apple_mask1, apple_mask2)


# 이미지 출력
plt.figure(figsize=(16,8))

plt.subplot(1,2,1), plt.imshow(apple_rgb), plt.title("Original Image")
plt.subplot(1,2,2), plt.imshow(apple_mask, cmap="gray"), plt.title("Binary Mask")

for ax in plt.gcf().axes:
    ax.axis("off")

# %%
# Binary Mask를 사용하여 Contour 추출

# 필요하다면 Morphological Operation 적용
kernel = np.ones((5, 5), np.uint8)
apple_mask = cv2.morphologyEx(apple_mask, cv2.MORPH_OPEN, kernel)
apple_mask = cv2.morphologyEx(apple_mask, cv2.MORPH_CLOSE, kernel)

# Contour를 추출하여 원본 이미지 위에 overlay
final_img = apple_rgb.copy()
contours, _ = cv2.findContours(apple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 0:
    # 가장 큰 contour 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(apple_mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 가장 큰 contour만 채운 binary mask를 반투명 초록색으로 overlay
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = largest_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (초록색 외곽선)
    cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2)

plt.figure(figsize=(16,8))
plt.imshow(final_img)
plt.axis("off")

# %% [markdown]
# 추출한 Contour를 사용해 객체 탐지를 할 수 있습니다.
#
# `cv2.boundingRect()` 함수를 사용하면 이미지에서 찾은 Contour를 감싸는 최소 사각형을 자동으로 계산합니다.
#
# BBox 표현은 2번 방식인 "`bbox = (x, y, w, h)`"으로 표현됩니다.

# %%
x, y, w, h = cv2.boundingRect(largest_contour)
print(x, y, w, h)

# %% [markdown]
# 이 값을 사용하여 원본 이미지 위에 BBox를 그려봅시다.

# %%
def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# %%
apple = cv2.imread("src/images/apple.png")
apple_rgb = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)

# cv2.rectangle() 함수에 사용할 수 있는 좌표로 변환
x1, y1 = x, y
x2, y2 = x + w, y + h

draw_bbox(apple_rgb, (x1, y1), (x2, y2), (0, 255, 0), txt="Apple, 0.98")

plt.figure(figsize=(16,8))
plt.imshow(apple_rgb);

# %% [markdown]
# 이로써 BBox의 좌표를 하드코딩하는 것이 아닌 규칙기반 객체 탐지인 Color Segmentation으로 BBox를 그려봤습니다.

# %% [markdown]
# ---
