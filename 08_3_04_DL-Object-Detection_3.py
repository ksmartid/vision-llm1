# -*- coding: utf-8 -*-
"""
Step 3/10: C. Non-Maximum Suppression (NMS)
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# 이전 단계(Step 1, A. Bounding Box / Step 2, B. Intersection over Union (IoU))에서 정의한 함수를 이어서 사용합니다.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data


def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


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

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

# %% [markdown]
# ### C. Non-Maximum Suppression (NMS)

# %% [markdown]
# 비최대 억제 (NMS):
# * 동일한 객체에 대해 여러 개의 Bounding Box가 높은 신뢰도로 예측될 수 있음
# * 중복 Bounding Box를 제거하는 알고리즘
# * Confidence Score와 IoU 기준으로 Bounding Box 선택
# * 중복 제거 과정을 반복하여 객체마다 하나의 대표 Bounding Box만 남김

# %% [markdown]
# NMS 과정 (2~6번):
# 1. 신뢰도가 너무 낮은 BBox 제거 (Confidence Thresholding)
# 2. Confidence가 가장 높은 BBox 선택
# 3. 선택한 BBox와 다른 박스들의 IoU 계산
# 4. IoU가 임계값보다 높은 박스 제거 (IoU Thresholding)
# 5. 다음으로 Confidence가 가장 높은 BBox 선택
# 6. 3, 4, 5번 반복
#

# %% [markdown]
# NMS 알고리즘을 구현하기 앞서, 우선 랜덤으로 BBox를 생성해봅시다.

# %%
from random import randint, gauss

def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))

# %%
hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

# %% [markdown]
# 이제 이 난잡한 BBox 정글을 NMS로 정리해보도록 합시다.

# %%
# Confidence Thresholding 함수 정의
# Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """

    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores

# %%
# NMS 함수 정의
# np.sort() 함수를 사용하면 1차원 배열을 정렬
# np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """

    if len(boxes) == 0:
        return [], []

    # Confidence 내림차순으로 정렬된 인덱스
    order = np.argsort(scores)[::-1]

    keep = []

    while len(order) > 0:
        current = order[0]
        keep.append(current)

        remaining = order[1:]
        ious = np.array([calculate_iou(boxes[current], boxes[idx]) for idx in remaining])

        # IoU가 임계값보다 높은 박스 제거
        order = remaining[ious <= iou_threshold]

    nms_boxes = [boxes[idx] for idx in keep]
    nms_scores = [scores[idx] for idx in keep]

    return nms_boxes, nms_scores

# %% [markdown]
# 이제 Confidence Thresholding과 NMS를 적용해봅시다.

# %%
print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(boxes, scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))

# %%
hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

# %% [markdown]
# 확실히 중복된 Bounding Box는 제거된 모습을 확인할 수 있습니다.
#
# 여전히 중복되어 있는 BBox들은, 겹친 영역의 면적이 작거나 합집합의 면적이 넓어 교집합:합집합 비율이 임계값보다 낮은 경우입니다.

# %% [markdown]
# 지금까지 NMS 알고리즘의 동작 원리를 이해하기 위해 직접 코드로 구현해 보았습니다.
#
# OpenCV에서도 동일한 역할을 하는 내장 함수 `cv2.dnn.NMSBoxes()`가 제공됩니다.
#
# OpenCV 함수를 사용해 같은 과정을 반복해 보고 결과를 비교해봅시다.

# %%
def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    """
    OpenCV의 cv2.dnn.NMSBoxes()를 이용해 NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록
        각 박스는 기존과 동일한 (x1, y1, x2, y2) 형식

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score
    """

    if len(boxes) == 0:
        return [], []

    # cv2.dnn.NMSBoxes()는 (x, y, width, height) 형식을 사용
    boxes_xywh = []

    for box in boxes:
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        boxes_xywh.append([int(x1), int(y1), int(width), int(height)])

    # OpenCV NMS 적용하여 BBox 제거 후 남은 BBox의 인덱스
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    # 1차원 리스트로 변환
    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    # 원래 xyxy 형식의 박스에서 선택
    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores

# %% [markdown]
# 다시 NMS 적용 전 이미지를 확인합시다.

# %%
hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

# %% [markdown]
# OpenCV의 NMS 적용:

# %%
print(len(boxes), len(scores))

cv2_new_boxes, cv2_new_scores = apply_nms_cv2(boxes, scores, iou_threshold=0.2)

print(len(cv2_new_boxes), len(cv2_new_boxes))

# %%
hubble_cv2_nms = skimage.data.hubble_deep_field()

for i, box in enumerate(cv2_new_boxes):
    draw_bbox(
        hubble_cv2_nms,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble_cv2_nms)
plt.axis("off");

# %% [markdown]
# 직접 구현한 NMS와 OpenCV 내장 NMS의 결과를 비교해봅시다.

# %%
plt.figure(figsize=(16,10))

plt.subplot(1,2,1), plt.imshow(hubble), plt.title("Custom NMS Result")
plt.subplot(1,2,2), plt.imshow(hubble_cv2_nms), plt.title("CV2 NMS Result")

for ax in plt.gcf().axes:
    ax.axis("off")

# %% [markdown]
# 직접 구현한 결과와 정확히 일치하는 것을 확인할 수 있습니다.
#
# OpenCV 내장 함수를 사용하면 간단하게 NMS를 적용할 수 있지만, 알고리즘을 직접 구현해 봄으로써 그 내부 동작 원리를 깊이 있게 이해할 수 있습니다.

# %% [markdown]
# ---
