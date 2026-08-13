# -*- coding: utf-8 -*-
"""
Step 2/10: B. 다층 퍼셉트론의 구현
원본: 03_DL-and-GPU.ipynb (단계별로 재구성)
"""

# %% [markdown]
# 이전 단계(Step 1, A. 퍼셉트론의 구현)에서 정의한 함수를 이어서 사용합니다.

# %%
import numpy as np


def step_function(z_):
    if z_ >= 0:
        return 1
    return 0


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    z = np.sum(x * w) + b

    return step_function(z)


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    z = np.sum(x * w) + b

    return step_function(z)


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    z = np.sum(x * w) + b

    return step_function(z)


def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.2

    z = np.sum(x * w) + b

    return step_function(z)


def print_gate_results():
    gates = ["AND", "OR", "NAND", "NOR", "XOR"]
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]

    header = "".join(f"{name:>{len(name)+2}s}" for name in gates)
    print(f"x1 x2 |{header}")
    print("-" * (7 + sum(len(name) + 2 for name in gates)))

    for x1, x2 in inputs:
        outputs = (AND(x1,x2), OR(x1,x2), NAND(x1,x2), NOR(x1,x2), XOR(x1,x2))

        row = "".join(f"{str(v):>{len(name)+2}s}"
                      if v is not None
                      else f"{'-':>{len(name)+2}s}"
                      for v, name in zip(outputs, gates))

        print(f"{x1:2d} {x2:2d} |{row}")

# %% [markdown]
# ### B. 다층 퍼셉트론의 구현

# %% [markdown]
# #### 단층 vs. 다층 인공신경망 구조

# %% [markdown]
# 단층 퍼셉트론 (Single-Layer Perceptron, SLP)
# * “입력층 – 출력층”만 존재
# * 선형 분리 문제만 해결 가능

# %% [markdown]
# 다층 퍼셉트론 (Multi-Layer Perceptron, MLP)
# * “입력층 – 은닉층 – 출력층” 존재 (하나 이상의 은닉층)
# * 비선형 분리 문제 해결 가능

# %% [markdown]
# 슬라이드에서 확인했듯이, XOR 논리회로는 선형적으로 분리할 수 없는 비선형 문제에 해당합니다.
#
# 따라서 이를 해결하기 위해서는 비선형 분리가 가능한 다층 퍼셉트론(MLP) 접근법이 필요하며,
# 이를 위해 앞서 실습한 단층 구조에 '은닉층(Hidden Layer)'을 추가해야 합니다.

# %%
def XOR_(x1, x2):
    # ------------- 입력층 -------------
    x = np.array([x1, x2])

    # ------------- 은닉층 -------------
    w1 = np.array([-0.5, -0.5])
    b1 = 0.7
    z1 = np.sum(x * w1) + b1
    y1 = step_function(z1)

    w2 = np.array([0.5, 0.5])
    b2 = -0.2
    z2 = np.sum(x * w2) + b2
    y2 = step_function(z2)

    # ---------- 은닉층 결과 ------------
    hidden_result = np.array([y1, y2])

    # ---------- 출력층 입력 ------------
    x_output_layer = hidden_result

    # ------------- 출력층 -------------
    w3 = np.array([0.5, 0.5])
    b3 = -0.7
    z3 = np.sum(x_output_layer * w3) + b3
    y = step_function(z3)

    return y

print(f"XOR(0,0) :  {XOR_(0, 0)}")  # 출력: 0
print(f"XOR(1,0) :  {XOR_(1, 0)}")  # 출력: 1
print(f"XOR(0,1) :  {XOR_(0, 1)}")  # 출력: 1
print(f"XOR(1,1) :  {XOR_(1, 1)}")  # 출력: 0

# %% [markdown]
# 위 코드에서 볼 수 있듯이, XOR 논리회로는 "`AND(NAND(x1,x2), OR(x1,x2))`"입니다.
#
# 그러므로, 이전에 정의한 `AND()`, `NAND()`, `OR()` 함수를 사용하여 위의 `XOR()` 함수를 간소화할 수 있습니다.

# %%
# XOR(x1,x2) = AND(NAND(x1,x2), OR(x1,x2))
def XOR(x1, x2):
    s1 = NAND(x1, x2)  # 은닉층 노드 1
    s2 = OR(x1, x2)    # 은닉층 노드 2
    y = AND(s1, s2)    # 출력층

    return y

print(f"XOR(0,0) :  {XOR(0, 0)}")  # 출력: 0
print(f"XOR(1,0) :  {XOR(1, 0)}")  # 출력: 1
print(f"XOR(0,1) :  {XOR(0, 1)}")  # 출력: 1
print(f"XOR(1,1) :  {XOR(1, 1)}")  # 출력: 0

# %% [markdown]
# 논리회로 결과 비교:

# %%
print_gate_results()

# %% [markdown]
# 앞서 살펴본 바와 같이, 단층 퍼셉트론을 통해 AND, OR, NAND, NOR 논리 게이트를 성공적으로 구현할 수 있었습니다. 하지만 단층 구조로는 해결할 수 없는 비선형 분류 문제인 XOR 게이트의 경우, 은닉층을 추가한 다층 퍼셉트론 구조를 통해 해결할 수 있음을 확인했습니다.

# %% [markdown]
# ---
