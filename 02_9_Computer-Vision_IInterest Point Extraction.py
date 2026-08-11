# -*- coding: utf-8 -*-
"""
Part I: Interest Point Extraction
원본: 02_Computer-Vision.ipynb (cell 200~212)
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
plt.show()

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
