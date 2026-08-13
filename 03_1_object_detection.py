# -*- coding: utf-8 -*-
"""
Step 1/3: 실시간 객체 검출 - 색상 기반 Binary Mask 생성
원본: object_detection.py (단계별로 재구성)
"""


# %% [markdown]
# ### 1단계: LAB 색공간 기반 Binary Mask 생성
#
# 실시간 카메라 프레임에서 원하는 색상(초록)만 이진 마스크로 뽑아내는 첫 단계입니다.
# 아직 Morphological Operation이나 Contour Detection은 적용하지 않고, 순수한 색상 마스크만 확인합니다.

# %%
import numpy as np
import cv2
import time


# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

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


def open_camera(pipeline, max_retries=6, retry_delay=2.0):
    # CSI 센서 초기화 시간이 매번 달라 첫 시도에서 간헐적으로 열리지 않는 경우가 있어 재시도합니다.
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()
        print(f"카메라 연결 시도 {attempt}/{max_retries} 실패, {retry_delay}초 후 재시도합니다...")
        time.sleep(retry_delay)
    return cap


cap = open_camera(pipeline)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 아래를 확인해보세요:")
    print("  1) 다른 노트북/스크립트가 카메라를 사용 중인지 확인 후 종료")
    print("  2) 잠시 후 이 스크립트를 다시 실행")
    print("  3) 그래도 안 되면: sudo systemctl restart nvargus-daemon 실행 후 재시도")
else:
    try:
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

            cv2.imshow("Original", frame)
            cv2.imshow("Color Mask (raw)", mask)

            # while loop rate (FPS) 설정
            time.sleep(max(1. / 25 - (time.time() - start), 0))
    finally:
        cap.release()
        cv2.destroyAllWindows()
