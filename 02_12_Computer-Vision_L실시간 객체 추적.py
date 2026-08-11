# -*- coding: utf-8 -*-
"""
Part L: 실시간 객체 추적
원본: 02_Computer-Vision.ipynb (cell 253~264)
"""


# %% [markdown]
# ### L. 실시간 객체 추적

# %% [markdown]
# Kalman Filter (칼만 필터)란?
# * 노이즈가 포함된 측정값으로부터 실제 상태(위치, 속도 등)를 추정하는 알고리즘
# * 예측(Prediction)과 보정(Update) 과정을 반복하여 추정값을 지속적으로 개선
# * 센서 오차나 일시적인 측정 실패가 있어도 안정적인 추적 가능
# * 로봇 제어, 자율주행, 객체 추적 등 다양한 실시간 시스템에 활용

# %% [markdown]
# 한번 직접 Kalman Filter 함수를 만들어봅시다.
#
# 우선 역행렬 (inverse matrix)를 만들기 위해 NumPy의 `linalg.inv`를 import 합니다.
#
# Inverse는 `inv(_)`, transpose는 `_.transpose()`, dot product는 `_.dot(_)` 함수를 사용합니다.

# %% [markdown]
# ```python
# from numpy.linalg import inv
# ```

# %% [markdown]
# Kalman Filter 함수 및 변수 정의:

# %% [markdown]
# ```python
# def KalmanFilter(mu_prev, sigma_prev, z):
#     mu_bar = A_t.dot(mu_prev)
#     sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
#     if z is None:
#         return mu_bar, sigma_bar
#     else:
#         K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
#         mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
#         sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
#         return mu, sigma
#
#
# # Kalman filter 변수 정의
# A_t = np.array([[1, 1], [0, 1]])
# G = np.array([[0.5], [1]])
# R_t = G.dot(G.transpose())
# C_t = np.array([[1, 0]])
# Q_t = np.array([[1]])
# mu_t = np.array([[0, 0], [0, 0]])
# sigma_t = np.array([[0, 0], [0, 0]])
# ```

# %% [markdown]
# Kalman Filter 함수를 적용해봅시다.

# %% [markdown]
# ```python
# # TODO: Kalman Filter를 활용하여 객체 추적
#
# # TODO: 탐지된 객체의 중심과 반지름을 구해 프레임 위에 overlay
# # TODO: 예측된 객체의 중심과 반지름을 구해 프레임 위에 다른 색으로 overlay
#
# contour_lst, _ = cv2.findContours(...)
#
# # 객체 최초 검출 여부 확인용 boolean
# found = False
#
# while True:
#     frame = ...
#
#     if len(contour_lst) > 0:
#         # 가장 큰 contour 선택
#         contour = ...
#         # 최소 외접원 반지름
#         radius = ...
#         # 무게중심
#         center = ...
#
#         # 검출된 객체에 파란 원 overlay
#         cv2.circle(frame, center, int(radius), ...)
#         cv2.circle(frame, center, ...)
#
#         # 객체 최초 검출
#         if not found:
#             found = True
#
#     # 최초 검출 이후 Kalman Filter 적용
#     # 측정값 사용 (visible) -> Prediction & Update
#     if found and (len(contour_lst) > 0):
#         mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
#         x_bel, y_bel = mu_t[0][0], mu_t[0][1]
#
#     # 측정값 미사용 (occluded) -> Prediction
#     elif found and (len(contour_lst) <= 0):
#         mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
#         x_bel, y_bel = mu_t[0][0], mu_t[0][1]
#
#     # 예측한 객체 위치에 노란 원 overlay
#     cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius),...)
#     cv2.circle(frame, (int(x_bel), int(y_bel)), ...)
# ```

# %% [markdown]
# 직접 정의한 Kalman Filter 대신 OpenCV의 `cv2.KalmanFilter` 함수를 사용하여 더욱 부드러운 예측이 가능합니다.

# %% [markdown]
# ```python
# # TODO: OpenCV의 Kalman Filter 함수를 활용하여 객체 추적
#
# # --------------- OpenCV Kalman Filter 설정 ---------------
# # 상태 벡터: [x, y, vx, vy]
# # 측정 벡터: [x, y]
# kalman = cv2.KalmanFilter(4, 2)
#
# # Transition Matrix (상태 전이 행렬)
# kalman.transitionMatrix = np.array(
#     [
#         [1, 0, 1, 0],
#         [0, 1, 0, 1],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ],
#     dtype=np.float32,
# )
#
# # Measurement Matrix (측정 행렬)
# # 측정값으로부터 x, y만 관측
# kalman.measurementMatrix = np.array(
#     [
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#     ],
#     dtype=np.float32,
# )
#
# # Process Noise Covariance Matrix (프로세스 노이즈 공분산)
# # 값이 클수록 모델 예측보다 측정값 변화에 더 유연하게 반응
# kalman.processNoiseCov = np.array(
#     [
#         [1e-2, 0, 0, 0],
#         [0, 1e-2, 0, 0],
#         [0, 0, 5e-2, 0],
#         [0, 0, 0, 5e-2],
#     ],
#     dtype=np.float32,
# )
#
# # Measurement Noise Covariance Matrix (측정 노이즈 공분산)
# # 값이 클수록 측정값을 덜 신뢰하고 예측값을 더 신뢰
# kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
#
# # Posteriori Error Covariance Matrix (초기 추정 오차 공분산)
# kalman.errorCovPost = np.eye(4, dtype=np.float32)
#
# # -------------------------------------------------------
#
# contour_lst, _ = cv2.findContours(...)
#
# # 객체 최초 검출 여부 확인용 boolean
# found = False
#
# while True:
#     frame = ...
#
#     if len(contour_lst) > 0:
#         # 가장 큰 contour 선택
#         contour = ...
#         # 최소 외접원 반지름
#         radius = ...
#         # 무게중심
#         center = ...
#
#         # 검출된 객체에 파란 원 overlay
#         cv2.circle(frame, center, int(radius), ...)
#         cv2.circle(frame, center, ...)
#
#         # 객체 최초 검출 (최초 a priori state 저장)
#         if not found:
#             kalman.statePre = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
#             found = True
#
#     # 최초 검출 이후 Kalman Filter 적용
#         # 측정값 사용 (visible) -> Prediction & Update
#         if found and (len(contour_lst) > 0):
#             predicted_state = kalman.predict()   # prediction
#             corrected_state = kalman.correct(z)  # update
#             x_bel = corrected_state[0, 0]
#             y_bel = corrected_state[1, 0]
#
#         # 측정값 미사용 (occluded) -> Prediction
#         elif found and (len(contour_lst) <= 0):
#             predicted_state = kalman.predict()   # prediction (update X)
#             x_bel = predicted_state[0, 0]
#             y_bel = predicted_state[1, 0]
#
#     # 예측한 객체 위치에 노란 원 overlay
#     cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), ...)
#     cv2.circle(frame, (int(x_bel), int(y_bel)), ...)
# ```

# %% [markdown]
# 이로써 객체를 탐지와 함께 추적 및 예측도 할 수 있게 되었습니다.
#
# Kalman Filter를 사용하여, 선형적으로 객체의 위치를 예측하여 조명 오염, 센서 불량, 객체 가려짐 등 상황에서도 끊임 없는 추적을 할 수 있으며, 몇 초 뒤의 객체 위치도 대략적으로 예측이 가능합니다.

# %% [markdown]
# ---
