# -*- coding: utf-8 -*-
"""
Step 8/10: G. Power Mode
원본: 04_DL-Object-Detection.ipynb (단계별로 재구성)
"""

# %% [markdown]
# ### G. Power Mode

# %% [markdown]
# Jetson에는 CPU, GPU, 메모리 등의 최대 동작 범위를 조절하여 시스템의 성능 수준을 선택하는 Power Mode가 있습니다.
#
# Jetson Orin Nano에는 기본적으로 다음과 같은 Power Mode가 제공됩니다.
#
# | Mode ID | Power Mode | 전력 제한 | CPU 최대 클럭 | GPU 최대 클럭 | Memory 최대 클럭 | 성능 | 발열/전력 | 적합한 용도 |
# |---:|---|---:|---:|---:|---:|---|---|---|
# | **0** | **15W** | 15W | 1497.6 MHz | 612 MHz | 2133 MHz | 중간 | 중간 | 일반 개발, 가벼운 AI 추론 |
# | **1** | **25W** | 25W | 1344 MHz | **918 MHz** | **3199 MHz** | 높음 | 높음 | YOLO 실시간 추론, 일반적인 고성능 AI |
# | **2** | **MAXN SUPER** | 고정 전력 제한 없음 | **1728 MHz** | **1020 MHz** | **3199 MHz** | **최고** | **가장 높음** | 최대 FPS, TensorRT 벤치마크, 고성능 AI 추론 |

# %% [markdown]
# Power Mode를 고성능 옵션으로 설정하는 것이 항상 좋은 것은 아닙니다. Power Mode를 높이면 CPU와 GPU가 더 높은 성능으로 동작할 수 있지만, 그만큼 전력 소비와 발열도 증가합니다.
#
# 따라서 처리 속도는 향상될 수 있지만, 항상 일정한 최고 성능이 보장되는 것은 아닙니다. 시스템이 전력 또는 열 한계에 도달하면 Throttling이 발생하여 CPU/GPU 클럭이 제한되고 성능이 저하될 수 있습니다. 또한 전원 공급이 불안정하거나 온도가 위험 수준까지 상승하면 하드웨어 보호를 위해 Jetson이 강제로 Shutdown될 수 있습니다.
#
# 따라서 Power Mode는 무조건 가장 높은 모드를 사용하는 것이 아니라, 필요한 연산 성능, 전력 공급 능력, 배터리 사용 시간, 냉각 환경 등을 고려하여 적절하게 선택해야 합니다.

# %% [markdown]
# 기본적으로 Power Mode는 Jetson 화면의 우측 상단에서 선택할 수 있습니다.
#
# 하지만 대부분의 경우, 모니터 없이 원격으로 제어하는 Headless 모드로 사용하기에 터미널 명령어로 설정해줍시다.
#
# 우선, 현재 Power Mode를 확인해볼까요?

# %% [markdown]
# ```bash
# sudo nvpmodel -q
# ```

# %% [markdown]
# 위 명령어로 현재 Power Mode와 인덱스를 확인할 수 있습니다.
#
# Jetson Orin Nano 기준, Power Mode 인덱스는 아래와 같습니다.
#
# ```text
# 0: 15W
# 1: 25W
# 2: MAXN
# ```

# %% [markdown]
# 이제 Power Mode를 변경해봅시다.
#
# `<mode_id>`를 원하는 Power Mode 인덱스로 바꿔 설정할 수 있습니다.
#
# 최대 성능을 활용할 수 있는 2번(MAXN)으로 설정해봅시다.

# %% [markdown]
# ```bash
# sudo nvpmodel -m <mode_id>
# ```

# %% [markdown]
# Power Mode를 바꿔가며 YOLO 실시간 객체 탐지 FPS를 비교해보도록 합시다.

# %% [markdown]
# ---
