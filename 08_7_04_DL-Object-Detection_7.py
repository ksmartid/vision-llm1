# -*- coding: utf-8 -*-
"""
Step 7/10: G. Jetson 성능 모니터링
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# ### G. Jetson 성능 모니터링

# %% [markdown]
# Jetson은 제한된 연산 자원과 메모리, 전력 및 발열 제약을 가진 엣지 디바이스입니다.
#
# 따라서 시스템을 효율적으로 최적화하기 위해 Jetson의 자원 사용 상태와 성능 병목을 실시간으로 확인할 필요가 있습니다.
#
# 다음 명령어로 실시간 성능 모니터링을 실시해봅시다.

# %% [markdown]
# ```bash
# tegrastats
# ```

# %% [markdown]
# 혹은, Jetson 화면의 우측 상단에 Power Mode 메뉴에서 `Run tegrastats`를 선택하여 실행하는 방법도 있습니다.

# %% [markdown]
# 처음 봤을 때에는 너무 많은 텍스트에 압도되어 어떤 정보도 쉽게 눈에 들어오지 않을 겁니다.
#
# `tegrastats`의 출력 중 아래 키워드들을 먼저 이해하도록 합시다.

# %% [markdown]
# | 항목 | 의미 |
# | --- | --- |
# | `RAM` | 시스템 메모리 사용량 |
# | `SWAP` | Swap 메모리 사용량 |
# | `CPU` | 각 CPU 코어 사용률 및 동작 클럭 |
# | `GR3D_FREQ` | GPU 사용률 및 GPU 클럭 |
# | `EMC_FREQ` | 메모리 컨트롤러 사용률 및 클럭 |
# | `cpu@...C` | CPU 온도 |
# | `gpu@...C` | GPU 온도 |
# | `VDD_IN` | Jetson 전체 전력 소비량 |

# %% [markdown]
# 이 중에서 중요하게 확인해야 할 항목들은 아래와 같습니다.
#
# ```text
# RAM            → 메모리 사용량
# CPU            → CPU 사용률
# GR3D_FREQ      → GPU 사용률
# cpu@ / gpu@    → 온도
# VDD_IN         → 전체 전력 소비량
# ```

# %% [markdown]
# `tetrastats` 출력 화면에서 필요한 키워드들을 직접 찾아가며 확인하는 것은 아무래도 직관성이 떨어지고 눈에 잘 들어오지 않을 겁니다.
#
# 따라서 아래 명령어를 실행하여 실시간 성능을 모니터링해 봅시다.

# %% [markdown]
# ```bash
# tegrastats | grep "RAM"
# ```

# %% [markdown]
# `grep` 명령어는 텍스트 데이터나 파일 내부에서 특정 키워드를 검색하여 강조하며 해당 줄만 추출해 주는 도구입니다.
#
# 복잡하고 길게 출력되는 tegrastats 정보 속에서 우리가 필요한 성능 지표만 선별하여 한눈에 확인하고 싶을 때 매우 유용하게 활용됩니다.

# %% [markdown]
# 이제 YOLO 실시간 객체 탐지 코드를 실행하고, GPU에서 객체 탐지가 진행되는 동안 주요 성능 키워드들이 강조되도록 설정하여 실시간 모니터링을 진행해 봅시다.

# %% [markdown]
# ---
