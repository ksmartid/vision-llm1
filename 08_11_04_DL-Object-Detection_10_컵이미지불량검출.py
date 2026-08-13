# -*- coding: utf-8 -*-
"""
컵 이미지 불량 검출 - 카메라 정상/비정상 판별 (+ 저장 시 즉시 학습)

1) Jetson CSI 카메라 프레임에서 YOLO로 컵을 검출(추출)
   - 근접 촬영 시 YOLO가 "cup"보다 "bowl"로 더 높게 잡는 경우가 많아 (실측 확인)
     "cup"과 "bowl"을 함께 후보로 인식하고, 신뢰도 임계값도 0.15로 낮춤
2) 검출된 컵 영역에서 흰색/크림색 계열 픽셀 비율(white_ratio)을 계산
3) 판별 기준 (둘 중 하나만 만족해도 정상):
   - white_ratio가 임계값 이상인 경우
   - 지금까지 카메라로 저장해 둔 정상 컵들과의 색 유사도가 임계값 이상인 경우
4) 정상으로 판별된 컵 이미지만 's' 키 입력 시 정상이미지 폴더에 저장하고,
   저장 즉시 그 이미지의 특징을 참조 데이터에 추가합니다 (실시간 학습).
   ('q' 키: 종료)

※ 인터넷에서 가져온 스톡 사진(정상이미지 폴더의 camera_ 접두사가 없는 파일)은
   배경/조명이 실제 카메라 환경과 크게 달라, 실측 결과 카메라 촬영본과의 색
   유사도가 0.002~0.25에 불과했습니다 (같은 카메라로 찍은 사진끼리는 0.67~0.99).
   따라서 학습(참조 데이터)에는 카메라로 직접 촬영한 이미지(camera_*.jpg)만 사용합니다.
"""

from pathlib import Path

import cv2
from ultralytics import YOLO


DATA_DIR = Path("src/datasets/cup_defect")
NORMAL_DIR = DATA_DIR / "정상이미지"
ABNORMAL_DIR = DATA_DIR / "비정상이미지"

NORMAL_DIR.mkdir(parents=True, exist_ok=True)
ABNORMAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "src/models/YOLO/yolo11n.pt"
# 카메라에 컵을 가깝게 대고 찍으면 YOLO가 "cup"보다 "bowl"로 더 높게 잡는 경우가 많아
# (실측: bowl 0.45 vs cup 미검출 등) 두 클래스를 모두 후보로 인식합니다.
CANDIDATE_CLASS_NAMES = ["cup", "bowl"]
DETECT_CONF = 0.15  # 근접 촬영 시 cup 신뢰도가 0.1~0.3대로 낮게 나오는 경우가 많아 낮춤

WHITE_SAT_MAX = 60    # 채도(S)가 이 값 이하이면 흰색/크림색 계열로 판단
WHITE_VAL_MIN = 140   # 명도(V)가 이 값 이상이면 흰색/크림색 계열로 판단
WHITE_RATIO_THRESHOLD = 0.5  # 컵 영역 중 흰색 계열 비율이 이 값 이상이면 정상

REFERENCE_SIMILARITY_THRESHOLD = 0.6  # 학습된 카메라 참조 이미지와의 색 유사도(0~1) 기준


def white_ratio(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (saturation <= WHITE_SAT_MAX) & (value >= WHITE_VAL_MIN)

    return float(mask.mean())


def compute_masked_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation <= WHITE_SAT_MAX) & (value >= WHITE_VAL_MIN)).astype("uint8") * 255

    hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)

    return hist


def load_camera_reference_histograms(image_dir):
    histograms = []

    for path in sorted(image_dir.glob("camera_*.jpg")):
        image = cv2.imread(str(path))

        if image is None:
            continue

        histograms.append(compute_masked_histogram(image))

    return histograms


def judge_normal(crop, reference_histograms):
    ratio = white_ratio(crop)

    if ratio >= WHITE_RATIO_THRESHOLD:
        return True, ratio, 1.0

    if not reference_histograms:
        return False, ratio, 0.0

    crop_hist = compute_masked_histogram(crop)
    best_score = max(
        cv2.compareHist(crop_hist, ref_hist, cv2.HISTCMP_CORREL)
        for ref_hist in reference_histograms
    )

    return best_score >= REFERENCE_SIMILARITY_THRESHOLD, ratio, best_score


def detect_cup(model, candidate_class_ids, frame):
    results = model.predict(
        source=frame,
        conf=DETECT_CONF,
        classes=candidate_class_ids,
        verbose=False,
    )
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    best_box = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

    return max(x1, 0), max(y1, 0), x2, y2


def capture_normal_images():
    model = YOLO(MODEL_PATH)
    candidate_class_ids = [
        idx for idx, name in model.names.items() if name in CANDIDATE_CLASS_NAMES
    ]

    reference_histograms = load_camera_reference_histograms(NORMAL_DIR)
    print(f"참조 데이터 {len(reference_histograms)}장으로 시작합니다.")

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

    save_count = len(list(NORMAL_DIR.glob("camera_*")))

    print("카메라 화면에서 's' 키: 정상으로 판별된 컵 저장(+학습) / 'q' 키: 종료")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        display_frame = frame.copy()
        box = detect_cup(model, candidate_class_ids, frame)
        crop = None
        is_normal = False
        ratio = 0.0
        similarity = 0.0

        if box is not None:
            x1, y1, x2, y2 = box
            candidate = frame[y1:y2, x1:x2]

            if candidate.size > 0:
                crop = candidate
                is_normal, ratio, similarity = judge_normal(crop, reference_histograms)
                color = (0, 255, 0) if is_normal else (0, 0, 255)
                label = f"{'정상' if is_normal else '비정상'} (백색비율 {ratio:.2f}, 유사도 {similarity:.2f})"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(
            display_frame,
            f"Saved: {save_count} / Refs: {len(reference_histograms)} (s: save / q: quit)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Normal Cup Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            if crop is not None and is_normal:
                save_path = NORMAL_DIR / f"camera_{save_count:02d}.jpg"
                cv2.imwrite(str(save_path), crop)
                save_count += 1

                reference_histograms.append(compute_masked_histogram(crop))
                print(f"[저장+학습] {save_path} (참조 데이터 {len(reference_histograms)}장)")
            else:
                print("컵이 검출되지 않았거나 비정상으로 판별되어 저장하지 않았습니다.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_normal_images()
