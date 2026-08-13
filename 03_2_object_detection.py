# -*- coding: utf-8 -*-
"""
Step 2/3: 실시간 객체 검출 - Morphological Operation으로 마스크 정제
원본: object_detection.py (단계별로 재구성)
"""


# %% [markdown]
# ### 2단계: Morphological Operation으로 마스크 정제
#
# 1단계에서 만든 Binary Mask에는 노이즈(작은 반점, 구멍)가 섞여 있습니다.
# Opening과 Dilation을 적용해 노이즈를 제거하고 객체 영역을 매끄럽게 다듭니다.

# %%
import numpy as np
import cv2
import time


# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# Morphological operation용 타원형 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    start = time.time()
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame = cv2.flip(frame, 0)

    # 가우시안 블러
    blr = cv2.GaussianBlur(frame, (11, 11), 0)

    # LAB 색공간 변환
    lab = cv2.cvtColor(blr, cv2.COLOR_BGR2LAB)

    # LAB color segmentation
    mask = cv2.inRange(lab, green_lower, green_upper)

    # Opening 2회, Dilation 2회로 마스크 정제
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask (raw)", mask)
    cv2.imshow("Mask (cleaned)", mask_clean)

    # while loop rate (FPS) 설정
    time.sleep(max(1. / 25 - (time.time() - start), 0))

cap.release()
cv2.destroyAllWindows()
