# -*- coding: utf-8 -*-
"""
Part D-3: Edge Detection - Sobel Filter
원본: 02_Computer-Vision.ipynb (cell 96~108)
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
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# TODO: image_gray를 float32로 변환
image_gray = image_gray.astype(np.float32) / 255.

# TODO: image_gray에 Gaussian Filter 적용 (Kernel 크기 11x11)
image_blur = cv2.GaussianBlur(image_gray, (11, 11), 0)

# TODO: dx, dy 생성
sobel_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3)

# TODO: subplot 3개 (원본 흑백 이미지, dx, dy) 생성
plt.figure(figsize=(18,8))
plt.subplot(1,3,1), plt.imshow(image_gray, cmap="gray"), plt.title("Original (grayscale)")
plt.subplot(1,3,2), plt.imshow(sobel_x, cmap="gray"), plt.title("dx")
plt.subplot(1,3,3), plt.imshow(sobel_y, cmap="gray"), plt.title("dy")

for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()

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
plt.show()

# %% [markdown]
# convertScaleAbs:
# * Magnitude (edge 강도)를 절대값으로 변환
# * 복잡한 결과를 일반 이미지 (0 ~ 255, 0.0 ~ 1.0) 형태로 변환

# %%
sobel_mag_abs = cv2.convertScaleAbs(sobel_mag)

plt.imshow(sobel_mag_abs, cmap="gray")
plt.axis("off");
plt.show()

# %% [markdown]
# Sobel Filter:
# * 장점
#   * 매우 빠른 속도
# * 단점
#   * 배경 노이즈에 취약
#   * 두꺼운 edge
#
