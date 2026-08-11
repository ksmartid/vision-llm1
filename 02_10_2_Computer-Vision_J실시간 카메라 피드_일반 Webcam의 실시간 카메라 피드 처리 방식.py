# -*- coding: utf-8 -*-
"""
Part J-2: 실시간 카메라 피드 - 일반 Webcam의 실시간 카메라 피드 처리 방식
원본: 02_Computer-Vision.ipynb (cell 226~230)
"""


# %% [markdown]
# #### $2)$ 일반 Webcam의 실시간 카메라 피드 처리 방식

# %% [markdown]
# #### `cv2.VideoCapture()`:

# %%
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. USB 웹캠이 연결되어 있는지 확인하세요.")
else:
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            cv2.imshow("VideoCapture", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()

# %% [markdown]
# #### `imutils.video.VideoStream()`:

# %%
import cv2
import time
from imutils.video import VideoStream

vs = VideoStream(src=0).start()
time.sleep(1.0)

try:
    while True:
        frame = vs.read()

        if frame is None:
            print("프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("VideoStream", frame)
finally:
    vs.stop()
    cv2.destroyAllWindows()
