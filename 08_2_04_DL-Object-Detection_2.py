# -*- coding: utf-8 -*-
"""
Step 2/10: B. Intersection over Union (IoU)
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# 이전 단계(Step 1, A. Bounding Box)에서 정의한 함수를 이어서 사용합니다.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# %% [markdown]
# ### B. Intersection over Union (IoU)

# %% [markdown]
# Intersection over Union (IoU):
# * 두 Bounding Box가 얼마나 겹치는지를 나타내는 값
# * (교집합 면적) / (합집합 면적)
# * 값의 범위: 0 ≤ IoU ≤ 1
#   * IoU = 0:  전혀 겹치지 않음
#   * IoU = 1:  두 BBox가 완전히 동일
#

# %% [markdown]
# ```python
# bbox_a = (xa_1, ya_1, xa_2, ya_2)
# bbox_b = (xb_1, yb_1, xb_2, yb_2)
# ```
#
# 두개의 BBox를 위와 같이 표현한다면, 좌표들을 사용해 교집합 박스의 좌표를 구할 수 있습니다.
#
# ```python
# intersection_x1 = max(xa_1, xb_1)
# intersection_y1 = max(ya_1, yb_1)
# intersection_x2 = min(xa_2, xb_2)
# intersection_y2 = min(ya_2, yb_2)
# ```
#
# <br>
# 그 다음, 교집합 박스의 너비와 높이를 계산합니다.
#
# ```python
# w = max(0, intersection_x2 - intersection_x1)
# h = max(0, intersection_y2 - intersection_y1)
# ```
#
# 겹치는 영역이 없다면 `intersection2 - intersection1`의 값이 음수가 되어 너비와 높이가 0이 됩니다.
#
# <br>
# 교집합의 넓이를 계산했다면, 최종적으로 합집합의 넓이까지 계산하여 IoU 값을 구할 수 있습니다.
#
# IoU 계산 함수를 구현해봅시다.

# %%
# 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    x, y, w, h = bbox

    x1, y1 = x, y
    x2, y2 = x + w, y + h

    return x1, y1, x2, y2


def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    intersection_w = max(0, intersection_x2 - intersection_x1)
    intersection_h = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_w * intersection_h

    area_a = (xa_2 - xa_1) * (ya_2 - ya_1)
    area_b = (xb_2 - xb_1) * (yb_2 - yb_1)
    union_area = area_a + area_b - intersection_area

    # ZeroDivisionError 주의
    if union_area == 0:
        return 0.0

    return intersection_area / union_area

# %% [markdown]
# 이제 3개의 이미지에서 IoU를 계산해봅시다.

# %%
apples1 = cv2.imread("src/images/two_apples1.png")
apples2 = cv2.imread("src/images/two_apples2.png")
apples3 = cv2.imread("src/images/two_apples3.png")
apples1_rgb = cv2.cvtColor(apples1, cv2.COLOR_BGR2RGB)
apples2_rgb = cv2.cvtColor(apples2, cv2.COLOR_BGR2RGB)
apples3_rgb = cv2.cvtColor(apples3, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,6))

plt.subplot(1,3,1), plt.imshow(apples1_rgb)
plt.subplot(1,3,2), plt.imshow(apples2_rgb)
plt.subplot(1,3,3), plt.imshow(apples3_rgb)

for ax in plt.gcf().axes:
    ax.axis("off")

# %%
def get_contour(img_bgr):
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([25, 10, 30])
    green_upper = np.array([100, 255, 255])

    img_blr = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    blr_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(blr_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(blr_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red, cntr_grn = None, None

    if len(contours_red) > 0:
        cntr_red = max(contours_red, key=cv2.contourArea)
    if len(contours_grn) > 0:
        cntr_grn = max(contours_grn, key=cv2.contourArea)

    return cntr_red, cntr_grn

# %%
apples1 = cv2.imread("src/images/two_apples1.png")
apples2 = cv2.imread("src/images/two_apples2.png")
apples3 = cv2.imread("src/images/two_apples3.png")

apples1_rgb = cv2.cvtColor(apples1, cv2.COLOR_BGR2RGB)
apples2_rgb = cv2.cvtColor(apples2, cv2.COLOR_BGR2RGB)
apples3_rgb = cv2.cvtColor(apples3, cv2.COLOR_BGR2RGB)

cntr_red1, cntr_grn1 = get_contour(apples1)
cntr_red2, cntr_grn2 = get_contour(apples2)
cntr_red3, cntr_grn3 = get_contour(apples3)

# %%
bbox1 = cv2.boundingRect(cntr_red1)
bbox2 = cv2.boundingRect(cntr_grn1)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples1_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples1_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples1_rgb);

# %%
iou1 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou1)

# %%
bbox1 = cv2.boundingRect(cntr_red2)
bbox2 = cv2.boundingRect(cntr_grn2)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples2_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples2_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples2_rgb);

# %%
iou2 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou2)

# %%
bbox1 = cv2.boundingRect(cntr_red3)
bbox2 = cv2.boundingRect(cntr_grn3)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples3_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples3_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples3_rgb);

# %%
iou3 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou3)

# %% [markdown]
# 이제 3가지 이미지의 IoU를 비교해봅시다.

# %%
plt.figure(figsize=(16,6))

plt.subplot(1,3,1), plt.imshow(apples1_rgb), plt.title(f"IoU: {iou1:.4f}")
plt.subplot(1,3,2), plt.imshow(apples2_rgb), plt.title(f"IoU: {iou2:.4f}")
plt.subplot(1,3,3), plt.imshow(apples3_rgb), plt.title(f"IoU: {iou3:.4f}")

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# ---
