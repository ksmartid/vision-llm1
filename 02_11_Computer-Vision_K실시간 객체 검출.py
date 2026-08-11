# -*- coding: utf-8 -*-
"""
Part K: 실시간 객체 검출
원본: 02_Computer-Vision.ipynb (cell 249~252)
"""


# %% [markdown]
# ### K. 실시간 객체 검출

# %% [markdown]
# 이제 앞선 실습에서 Color Segmentation을 활용해 이미지 속 원하는 색상 객체를 검출한 것과 동일하게 실시간 카메라 프레임 속 객체를 검출해봅시다.
#
# 기존과 동일한 방식을 사용하지만, 이제는 `while`문을 사용하여 매 프레임을 처리하여 출력합니다.

# %% [markdown]
# ```python
# # TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
#
# # 초록 LAB lower/upper range
# green_lower = np.array([30, 60, 90], dtype=np.uint8)
# green_upper = np.array([230, 115, 180], dtype=np.uint8)
#
# # TODO: 각 프레임 이미지를 전처리
# # TODO: Color Segmentation으로 Binary Mask 생성
# # TODO: Morphological Operation을 적용하여 객체 영역 추출
# # TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay
# ```

# %% [markdown]
# ---
