# -*- coding: utf-8 -*-
"""
Step 3/10: C. 활성화 함수
원본: 03_DL-and-GPU.ipynb (단계별로 재구성)
"""

# %%
import numpy as np

# %% [markdown]
# ### C. 활성화 함수

# %% [markdown]
# 활성화 함수 (Activation Function) 이란?
# * 다음 뉴런으로 얼마나 신호를 전달할지 결정하는 함수
# * 입력 신호를 출력 신호로 변환
# * 다양한 종류의 함수 존재 (Step Function, Sigmoid, ReLU 등)

# %% [markdown]
# #### $1)$ Step Function

# %% [markdown]
# $$
# f(x) \hspace{5pt} = \hspace{5pt}
# \begin{cases}
# 0 \text{, \hspace{8pt}} x \ge 0\\
# 1 \text{, \hspace{8pt}} x < 0
# \end{cases}
# $$

# %% [markdown]
# 우선, 앞선 실습에서 정의한 계단 함수부터 살펴보도록 하겠습니다.

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
# 이 같은 경우, 오직 하나의 값을 입력하여 하나의 값을 출력하는 구조입니다. 하지만, 만약 하나의 층(Layer)에 100개의 퍼셉트론이 존재한다면 활성화 함수를 매번 100번 호출해야 하는 비효율성이 발생합니다.
# 더 효율적인 연산을 위해 `NumPy` 라이브러리를 활용하여 하나의 층에 포함된 모든 퍼셉트론의 출력값을 하나의 배열로 묶어, 활성화 함수를 단 한 번에 일괄 적용할 수 있습니다.

# %%
def step(x):
    return np.array(x > 0, dtype=int)

# %%
print(step(np.array([-1.8, -100000, 0.0, 2.1, 9999])))

# %% [markdown]
# #### $2)$ Sigmoid

# %% [markdown]
# $$
# \sigma(x) \hspace{2pt} = \hspace{2pt} \frac{1}{1+e^{-x}}
# $$

# %% [markdown]
# 이번엔 Sigmoid 함수를 동일한 방식으로 정의해봅시다.

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# %% [markdown]
# #### $3)$ ReLU

# %% [markdown]
# $$
# \mathrm{ReLU}(x) \hspace{2pt} = \hspace{2pt} \max(0,x)
# $$

# %% [markdown]
# 마찬가지로 ReLU 함수도 정의할 수 있습니다.

# %%
def relu(x):
    return np.maximum(0, x)

# %% [markdown]
# #### 활성화 함수 시각화

# %% [markdown]
# 이제 정의한 세 활성화 함수를 그래프로 확인해볼까요?

# %%
import matplotlib.pyplot as plt

# 입력값 범위
x = np.linspace(-5, 5, 100)

# 활성화 함수 적용 후 출력값
y_step = step(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)

plt.figure(figsize=(16,6))
plt.subplot(1,3,1), plt.plot(x, y_step), plt.title("Step Function"), plt.ylim(-0.1, 1.1), plt.grid()
plt.subplot(1,3,2), plt.plot(x, y_sigmoid), plt.title("Sigmoid Function"), plt.ylim(-0.1, 1.1), plt.grid()
plt.subplot(1,3,3), plt.plot(x, y_relu), plt.title("ReLU Function"), plt.ylim(-0.5, 5), plt.grid();

# %% [markdown]
# ---
