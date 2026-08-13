# -*- coding: utf-8 -*-
"""
Step 1/10: A. 퍼셉트론의 구현
원본: 03_DL-and-GPU.ipynb (단계별로 재구성)
"""

# %% [markdown]
# <h1 style="text-align: center;">Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템</h1>
#
# <br><br>
#
# <div style="text-align: right; color: gray; font-style: italic;">
# 강사 김규래&emsp;<br>
# kkr.kyurae.kim@gmail.com&emsp;
# </div><br>
#
# ---
# ---

# %% [markdown]
# ## 3. 딥러닝 기초 및 GPU 가속

# %% [markdown]
# 딥러닝(Deep Learning, DL)이란?
# "사람의 뇌 신경망을 본떠 만든 인공신경망을 사용해 복잡한 데이터를 학습하는 기술"

# %% [markdown]
# 인공신경망 (Artificial Neural Network, ANN):
# * 인간의 뇌 신경세포(뉴런) 구조를 본떠 만든 학습 모델
# * 입력층 (Input Layer), 은닉층 (Hidden Layer), 출력층 (Output Layer)
# * 퍼셉트론 (Perceptron)이라는 인공 뉴런을 최소 단위로 사용

# %% [markdown]
# ---

# %% [markdown]
# ### A. 퍼셉트론의 구현

# %% [markdown]
# #### $1)$ 퍼셉트론의 구조와 개념

# %% [markdown]
# 퍼셉트론(Perceptron)은 복수의 입력을 받아 각각의 가중치를 곱하고, 여기에 편향을 더해 최종 출력을 만드는 과정이자 인경신공망의 최소 단위입니다. 이번 구현에서는 가장 직관적이고 이해하기 형태인 2개의 입력을 사용하는 퍼셉트론을 살펴봅시다.
# * 입력 (input): 모델이 받아들이는 외부의 데이터
# * 가중치 (weight): 각 입력이 결과에 미치는 중요도나 영향력을 조절하는 계수
# * 가중합: 입력에 각각의 가중치를 곱한 값의 총합
# * 편향 (bias): 모델이 얼마나 쉽게 활성화될지(1을 출력할지) 기준을 조절하는 상수

# %% [markdown]
# $$
# \text{입력: (} x_1 \text{, } x_2 \text{), \hspace{8pt} 가중치: (} w_1 \text{, } w_2 \text{), \hspace{8pt} 가중치: (} b \text{), \hspace{8pt} 출력: (} y \text{)}
# $$

# %% [markdown]
# #### $2)$ 퍼셉트론의 연산 과정

# %% [markdown]
# 1. 입력을 받아 입력의 가중합을 구합니다.
# $$
# w_1x_1 + w_2x_2
# $$
# 2. 구한 가중합에 편향을 더하여 결과 $z$를 계산합니다.
# $$
# z \hspace{4pt} = \hspace{4pt} w_1x_1 + w_2x_2 + b
# $$
# 3. 최종 산출된 $z$ 값을 사용하여 최종 출력인 $y$ 값으로 변환하기 위하여 활성화 함수 (Activation Function)을 적용합니다 ($y = f(z)$). 다양한 종류의 활성화 함수들이 있지만, 가장 기본적인 계단 함수 (Step Function)을 사용합시다.
# $$
# y \hspace{5pt} = \hspace{5pt}
# \begin{cases}
# 0 \text{, \hspace{8pt}} z \ge 0\\
# 1 \text{, \hspace{8pt}} z < 0
# \end{cases}
# $$

# %% [markdown]
# #### $3)$ 퍼셉트론을 활용한 논리회로 구현

# %% [markdown]
# 수식으로 살펴본 퍼셉트론의 구조와 연산 과정을 바탕으로, Python의 강력한 연산 라이브러리인 `NumPy`를 이용해 실제 코드로 구현합니다.
#
# 앞서 정의한 2개의 입력과 편향을 활용하여 논리 회로 중 가장 기초적인 AND 게이트를 퍼셉트론으로 구현하여 결과를 확인해봅시다.
#
# 우선 `NumPy` import가 필수입니다.

# %%
import numpy as np

# %% [markdown]
# 논리회로 구현에 사용할 활성화 함수인 step function도 정의합시다.

# %%
def step_function(z_):
    if z_ >= 0:
        return 1
    return 0

# %%
print(step_function(-1.8))
print(step_function(-100000))
print(step_function(0.0))
print(step_function(2.1))
print(step_function(9999))

# %% [markdown]
# 퍼셉트론을 활용한 AND 게이트 구현:

# %%
def AND_(x1, x2):
    w1, w2 = 0.5, 0.5  # 가중치 (weight)
    b = -0.7           # 편향 (bias)

    # 가중합과 편향 연산
    z = w1 * x1 + w2 * x2 + b

    # 활성화 함수 적용 (step function)
    y = step_function(z)

    # 최종 출력 반환
    return y

# AND 게이트 예시 실행 결과 확인
print(f"AND(0,0) :  {AND_(0, 0)}")  # 출력: 0
print(f"AND(1,0) :  {AND_(1, 0)}")  # 출력: 0
print(f"AND(0,1) :  {AND_(0, 1)}")  # 출력: 0
print(f"AND(1,1) :  {AND_(1, 1)}")  # 출력: 1

# %% [markdown]
# 동일한 코드를 `NumPy`의 함수를 활용해 다시 구현해봅시다.

# %%
def AND(x1, x2):
    x = np.array([x1, x2])    # 입력(input)을 NumPy 배열로 정의
    w = np.array([0.5, 0.5])  # 가중치(weight)를 NumPy 배열로 정의
    b = -0.7                  # 편향(bias)

    # 가중합과 편향 연산
    z = np.sum(x * w) + b

    # 활성화 함수 적용 (step function)
    y = step_function(z)

    # 최종 출력 반환
    return y

# AND 게이트 예시 실행 결과 확인
print(f"AND(0,0) :  {AND(0, 0)}")  # 출력: 0
print(f"AND(1,0) :  {AND(1, 0)}")  # 출력: 0
print(f"AND(0,1) :  {AND(0, 1)}")  # 출력: 0
print(f"AND(1,1) :  {AND(1, 1)}")  # 출력: 1

# %% [markdown]
# 이제 위와 같은 방식으로 OR, NAND, NOR, XOR을 순서대로 퍼셉트론을 활용하여 구현해봅시다.

# %%
# OR 게이트 구현
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    z = np.sum(x * w) + b

    return step_function(z)


print(f"OR(0,0) :  {OR(0, 0)}")  # 출력: 0
print(f"OR(1,0) :  {OR(1, 0)}")  # 출력: 1
print(f"OR(0,1) :  {OR(0, 1)}")  # 출력: 1
print(f"OR(1,1) :  {OR(1, 1)}")  # 출력: 1

# %%
# NAND 게이트 구현 (AND의 가중치/편향 부호를 반전)
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    z = np.sum(x * w) + b

    return step_function(z)


print(f"NAND(0,0) :  {NAND(0, 0)}")  # 출력: 1
print(f"NAND(1,0) :  {NAND(1, 0)}")  # 출력: 1
print(f"NAND(0,1) :  {NAND(0, 1)}")  # 출력: 1
print(f"NAND(1,1) :  {NAND(1, 1)}")  # 출력: 0

# %%
# NOR 게이트 구현 (OR의 가중치/편향 부호를 반전)
def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.2

    z = np.sum(x * w) + b

    return step_function(z)


print(f"NOR(0,0) :  {NOR(0, 0)}")  # 출력: 1
print(f"NOR(1,0) :  {NOR(1, 0)}")  # 출력: 0
print(f"NOR(0,1) :  {NOR(0, 1)}")  # 출력: 0
print(f"NOR(1,1) :  {NOR(1, 1)}")  # 출력: 0

# %%
# XOR 게이트 시도: AND/OR/NAND/NOR와 동일한 단일 퍼셉트론 방식으로 구현
# XOR은 선형적으로 분리할 수 없어(비선형 문제) 단층 퍼셉트론으로는 4가지 경우를
# 모두 만족하는 가중치/편향이 존재하지 않는다. 아래는 OR과 동일한 가중치를 사용한
# 시도이며, XOR(1,1)에서 정답(0)과 다른 결과(1)가 나오는 것을 확인할 수 있다.
# 이 한계는 다음 단계(B. 다층 퍼셉트론의 구현)에서 은닉층을 추가해 해결한다.
def XOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    z = np.sum(x * w) + b

    return step_function(z)

print(f"XOR(0,0) :  {XOR(0, 0)}")  # 출력: 0
print(f"XOR(1,0) :  {XOR(1, 0)}")  # 출력: 1
print(f"XOR(0,1) :  {XOR(0, 1)}")  # 출력: 1
print(f"XOR(1,1) :  {XOR(1, 1)}")  # 단층 퍼셉트론의 한계로 1 (정답은 0)

# %% [markdown]
# 논리회로 결과 비교:

# %%
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

print_gate_results()

# %% [markdown]
# ---
