# -*- coding: utf-8 -*-
"""
Step 4/10: D. 인공신경망 학습
원본: 03_DL-and-GPU.ipynb (단계별로 재구성)
"""

# %%
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# %% [markdown]
# ### D. 인공신경망 학습

# %% [markdown]
# 학습 (Learning):
# * 데이터와 정답을 비교해 가중치와 편향을 반복적으로 수정
# * 오차를 최소화하는 방향으로 최적화
# * 손실 함수(Loss Function)를 기준으로 학습 성능 평가
# * 학습률(Learning Rate)을 조절하여 가중치 업데이트 정도 (학습 안정성) 제어

# %% [markdown]
# 에포크 (Epoch):
# * 모델을 한 번 학습하는 과정 (cycle/iteration)
# * 에포크를 너무 적게 설정한 경우
#   * 학습이 충분히 이루어지지 않음
#   * 학습 데이터를 충분히 학습하지 못해 낮은 정확도를 보이는 과소적합(Underfitting) 발생 가능
# * 에포크를 너무 많이 설정한 경우
#   * 학습 데이터에 과도하게 맞춰짐
#   * 학습된 데이터에만 과도하게 최적화되어 처음 보는 데이터에는 대응하지 못하는 과적합(Overfitting) 발생 가능
#

# %% [markdown]
# 손실 함수 (Loss Function):
# * 모델의 예측값과 실제 정답이 얼마나 다른지 계산
# * 손실이 작을수록 모델의 예측이 정답과 가깝다는 의미
#
# 손실 함수의 기울기 (Gradient):
# * 가중치를 증가/감소시키면 손실이 감소하는지 알 수 있는 지표
# * 기울기가 하강하는 방향으로 가중치 수정
# * 손실 함수의 기울기는 활성화 함수의 기울기로 계산
#
# $$
# \frac{dL}{dw} \hspace{4pt} = \hspace{4pt} \frac{dL}{dy} \hspace{2pt} × \hspace{2pt} \frac{dy}{dz} \hspace{2pt} × \hspace{2pt} \frac{dz}{dw}
# $$

# %% [markdown]
# 학습률 (Learning Rate):
# * 한 번의 에포크에서 가중치를 얼마나 많이 변경할 것인지 결정
# * 학습률이 너무 작은 경우
#   * 학습이 매우 느림
#   * 최적 가중치에 도달하는 시간이 오래 걸림
# * 학습률이 너무 큰 경우
#   * 적절 가중치 값을 지나칠 수 있음
#   * 학습이 불안정해질 수 있음

# %% [markdown]
# 순전파 (Forward Propagation):
# * 데이터를 입력하여 다음 층으로 전달하는 과정
# * 정답 예측
#
# 역전파 (Backpropagation):
# * 손실 함수의 기울기를 계산하여 가중치 수정
# * 손실(오차)를 거꾸로 층을 거슬러 올라가며 기울기 계산과 가중치 수정 수행

# %% [markdown]
# 경사하강법 (Gradient Descent):
# * 가중치 수정 계산법
# * 손실 함수의 기울기를 사용하여 손실(오차)를 가장 낮추는 최적의 가중치를 찾는 과정
#
# $$
# w_{new} \hspace{4pt} = \hspace{4pt} w_{old} \hspace{2pt} - \hspace{2pt} \eta \frac{dL}{dw} \text{, \hspace{12pt}} \eta \text{ : 학습률}
# $$

# %% [markdown]
# #### $1)$ 논리회로 단층 인공신경망 학습

# %% [markdown]
# 앞서 논리회로 퍼셉트론을 구현할 때는 가중치와 편향을 직접 설정했으나, 이번에는 데이터를 바탕으로 이 값들을 스스로 학습하는 방법과 과정을 살펴보겠습니다.

# %% [markdown]
# 우선 각 논리회로의 정답을 준비합니다.

# %%
targets = {
    "AND": np.array([
        [0],
        [0],
        [0],
        [1],
    ], dtype=float),

    "OR": np.array([
        [0],
        [1],
        [1],
        [1],
    ], dtype=float),

    "NAND": np.array([
        [1],
        [1],
        [1],
        [0],
    ], dtype=float),

    "NOR": np.array([
        [1],
        [0],
        [0],
        [0],
    ], dtype=float),

    "XOR": np.array([
        [0],
        [1],
        [1],
        [0],
    ], dtype=float),
}

# %% [markdown]
# 그 다음, 인공신경망에 입력될 학습 데이터를 준비합니다.

# %%
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

# %% [markdown]
# 논리회로는 이진 분류 (Binary Classification) 문제로 볼 수 있어, 일반적으로 활성화 함수는 sigmoid와 손실 함수는 Binary Cross-Entropy를 사용합니다.
#
# Sigmoid 함수는 미리 정의해 두었기에 손실 함수만 새로 정의합시다.

# %%
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8

    loss = -np.mean(
        y_true * np.log(y_pred + eps)
        + (1 - y_true) * np.log(1 - y_pred + eps)
    )

    return loss

# %% [markdown]
# 이제 활성화 함수 `sigmoid()`와 손실 함수 `binary_cross_entropy()`를 사용하여 단층 구조의 신경망을 학습시키는 함수를 정의하고, AND 게이트를 학습해봅시다.

# %%
def train_single_neuron(X, y, learning_rate=0.1, epochs=10000):
    # 입력 데이터 배열 크기
    sample_count, input_count = X.shape

    # 입력 데이터 배열 크기에 맞춰 가중치/편향을 초기화
    w = np.zeros((input_count, 1))
    b = np.zeros((1,))

    loss_history = []

    for epoch in range(epochs):
        # 1. 순전파
        z = X @ w + b
        y_pred = sigmoid(z)

        # 2. 손실 계산
        loss = binary_cross_entropy(y, y_pred)
        loss_history.append(loss)

        # 3. 기울기 계산
        error = y_pred - y
        w_gradient = X.T @ error / sample_count
        b_gradient = np.mean(error, axis=0)

        # 4. Weight와 Bias 업데이트
        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

    return w, b, loss_history

# %%
and_weight, and_bias, and_loss = train_single_neuron(X, targets["AND"], learning_rate=0.1, epochs=10000)

and_probability = sigmoid(X @ and_weight + and_bias)
and_prediction = (and_probability >= 0.5).astype(int)

print("학습된 Weight:")
print(and_weight)
print("학습된 Bias:")
print(and_bias)
print("예측 확률:")
print(and_probability)
print("최종 예측:")
print(and_prediction)

# %%
training_results = {}

for gate_name, target in targets.items():
    weight, bias, loss_history = train_single_neuron(X, target, learning_rate=0.1, epochs=10000)

    probability = sigmoid(X @ weight + bias)
    prediction = (probability >= 0.5).astype(int)

    training_results[gate_name] = {
        "weight": weight,
        "bias": bias,
        "probability": probability,
        "prediction": prediction,
        "loss_history": loss_history,
    }

    print("=" * 50)
    print(f"{gate_name} 게이트")
    print("Weight:")
    print(weight.flatten())
    print("Bias:")
    print(bias)
    print("Probability:")
    print(probability.flatten())
    print("Prediction:")
    print(prediction.flatten())

# %% [markdown]
# 학습 결과는 아래와 같아야 합니다.
# * AND
#   * Weight: 양수, 양수
#   * Bias: 절댓값이 큰 음수
#   * Prediction: [0, 0, 0, 1]
# * OR
#   * Weight: 양수, 양수
#   * Bias: AND보다 절댓값이 작은 음수
#   * Prediction: [0, 1, 1, 1]
# * NAND
#   * Weight: 음수, 음수
#   * Bias: 큰 양수
#   * Prediction: [1, 1, 1, 0]
# * NOR
#   * Weight: 음수, 음수
#   * Bias: NAND보다 작은 양수
#   * Prediction: [1, 0, 0, 0]
# * XOR (단일 뉴런으로 학습되지 않는다)
#   * Weight: 0에 가까운 값
#   * Bias: 0에 가까운 값
#   * Probability: 약 [0.5, 0.5, 0.5, 0.5]

# %%
for gate_name, target in targets.items():
    prediction = training_results[gate_name]["prediction"]
    accuracy = np.mean(prediction == target)

    print(f"{gate_name} accuracy: {accuracy:.2f}")

# %% [markdown]
# XOR 정확도가 약 50%라는 것은 단일 뉴런이 XOR 규칙을 학습하지 못했다는 의미이다.

# %% [markdown]
# 이번에는 epoch가 진행됨에 따라 손실이 어떻게 변화하는지 그래프로 시각화해 봅시다.

# %%
plt.figure(figsize=(10, 6))

for gate_name in targets:
    loss_history = training_results[gate_name]["loss_history"]
    plt.plot(loss_history, label=gate_name)

plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("Single-neuron Training Loss"), plt.grid(), plt.legend();

# %% [markdown]
# * AND, OR, NAND, NOR 손실은 학습이 진행되면서 감소한다.
# * XOR 손실은 충분히 감소하지 않는다.
# * 모델 구조가 문제를 해결할 수 없으면 Epoch를 증가시켜도 해결되지 않는다.
# * 데이터가 충분하고 Epoch가 많더라도 모델 구조 자체가 문제에 적합하지 않으면 학습할 수 없다.

# %% [markdown]
# #### $2)$ 논리회로 다층 인공신경망 학습

# %% [markdown]
# XOR을 학습하기 위해 다음 구조를 사용합니다.
#
# 입력층: 2개
# 은닉층: 4개 뉴런
# 출력층: 1개 뉴런

# %% [markdown]
# 파라미터는 다음과 같습니다.
#
# W1: 입력층 → 은닉층 가중치
# b1: 은닉층 편향
# W2: 은닉층 → 출력층 가중치
# b2: 출력층 편향

# %% [markdown]
# 파라미터 크기:
# X:  (4, 2)
# W1: (2, 4)
# b1: (1, 4)
#
# Y1: (4, 4)
#
# W2: (4, 1)
# b2: (1, 1)
#
# 출력: (4, 1)

# %% [markdown]
# 이제 다층 인공신경망 학습 함수를 정의합시다.

# %%
def train_multi_neuron(X, y, hidden_size=4, learning_rate=1.0, epochs=20000, seed=42):
    sample_count, input_count = X.shape
    output_size = 1

    rng = np.random.default_rng(seed)

    W1 = rng.normal(loc=0.0, scale=1.0, size=(input_count, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = rng.normal(loc=0.0, scale=1.0, size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))

    loss_history = []

    for epoch in range(epochs):
        # 1. 순전파: 입력층 → 은닉층
        Z1 = X @ W1 + b1
        Y1 = sigmoid(Z1)

        # 2. 순전파: 은닉층 → 출력층
        Z2 = Y1 @ W2 + b2
        y_pred = sigmoid(Z2)

        # 3. 손실 계산
        loss = binary_cross_entropy(y, y_pred)
        loss_history.append(loss)

        # 4. 출력층 기울기
        dZ2 = (y_pred - y) / sample_count
        dW2 = Y1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # 5. 은닉층 기울기
        dY1 = dZ2 @ W2.T
        dZ1 = dY1 * Y1 * (1 - Y1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # 6. 파라미터 업데이트
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d}, " f"Loss: {loss:.6f}")

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }

    return parameters, loss_history

# %% [markdown]
# 이제 정의한 함수를 사용하여 학습을 실행합니다.

# %%
xor_parameters, xor_loss_history = train_multi_neuron(
    X,
    targets["XOR"],
    hidden_size=4,
    learning_rate=1.0,
    epochs=20000,
    seed=42,
)

# %% [markdown]
# 학습 결과를 확인해볼까요?

# %%
W1 = xor_parameters["W1"]
b1 = xor_parameters["b1"]
W2 = xor_parameters["W2"]
b2 = xor_parameters["b2"]

hidden_output = sigmoid(X @ W1 + b1)
xor_probability = sigmoid(hidden_output @ W2 + b2)
xor_prediction = (xor_probability >= 0.5).astype(int)

print("입력:")
print(X.astype(int))
print("예측 확률:")
print(xor_probability)
print("최종 예측:")
print(xor_prediction)

# %% [markdown]
# 학습된 Weight와 Bias도 확인해봅시다.

# %%
print("W1:")
print(W1)
print("b1:")
print(b1)
print("W2:")
print(W2)
print("b2:")
print(b2)

# %% [markdown]
# 마지막으로, XOR 인공신경망의 손실 변화를 그래프로 확인해봅시다.

# %%
plt.figure(figsize=(10, 6))
plt.plot(xor_loss_history)
plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("XOR Multi-layer Neural Network Training Loss"), plt.grid();

# %% [markdown]
# 이와 같이 은닉층을 활용한 다층 인공신경망을 학습시켜 비선형적인 분류 문제를 해결할 수 있음을 확인했습니다.

# %% [markdown]
# ---
