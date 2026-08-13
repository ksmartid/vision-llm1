# -*- coding: utf-8 -*-
"""
Step 6/10: F. CNN 실습 – MNIST 데이터셋
원본: 03_DL-and-GPU.ipynb (단계별로 재구성)
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### F. CNN 실습 – MNIST 데이터셋

# %% [markdown]
# 실습에 앞서 필요한 라이브러리를 설치해야 합니다.
#
# 그러기 위해서는 먼저 현재 JetPack과 CUDA 버전을 확인합시다.

# %% [markdown]
# ```bash
# dpkg -l | grep nvidia-jetpack
# cat /usr/local/cuda/version.json
# ```

# %% [markdown]
# JetPack 6.2.1 + CUDA 12.6 기준,
# ```text
# torch 2.8.0
# torchvision 0.23.0
# cu126
# ```
# 조합을 사용할 수 있습니다.

# %% [markdown]
# 설치:

# %% [markdown]
# ```bash
# pip install torch==2.8.0 torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
# pip uninstall torchvision
# pip install torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
# pip install "sympy>=1.13.3" --no-deps
# ```

# %%
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# %% [markdown]
# 위 코드가 에러 없이 실행이 되었다면 다음으로 넘어가면 됩니다.

# %% [markdown]
# #### $1)$ 필수 라이브러리 import

# %% [markdown]
# 우선, 필요한 라이브러리를 전부 import 합시다.

# %%
import time
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %% [markdown]
# #### $2)$ 난수 고정

# %% [markdown]
# 같은 코드를 반복 실행했을 때 최대한 비슷한 결과를 얻기 위해 난수 seed를 고정합니다.
# 결과가 완전히 동일하지 않을 수 있지만, 실습 결과의 재현성을 높일 수 있습니다.

# %%
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);

# %% [markdown]
# #### $3)$ MNIST 데이터셋 설치 및 불러오기

# %% [markdown]
# 손글씨 데이터셋인 MNIST 데이터를 불러옵시다.
#
# 최초로 데이터를 불러올 경우, 지정된 경로에 설치합니다.

# %%
# 데이터를 불러올 때 이미지 전처리 방식 정의
# PIL 이미지 또는 NumPy 이미지를 PyTorch Tensor로 변환
# 일반적인 8비트 이미지의 픽셀값을 0~1 범위의 실수로 변환
transform = transforms.ToTensor()

# 학습 데이터 불러오기 (최초 다운로드)
train_dataset = datasets.MNIST(
    root="src/datasets",
    train=True,
    transform=transform,
    download=True,
)

# 테스트 데이터 불러오기 (최초 다운로드)
test_dataset = datasets.MNIST(
    root="src/datasets",
    train=False,
    transform=transform,
    download=True,
)

# %% [markdown]
# 데이터셋을 설치할 `src/datasets` 폴더가 존재하는지 확인해봅시다.
#
# 만약 MNIST 데이터가 올바르게 설치되었다면, `datasets` 안에 `MNIST` 폴더가 자동으로 생성되고 그 내부에 데이터가 위치해 있을 것입니다.
#
# 아래 코드를 통해 확인해봅시다.

# %%
print(
    "Train data:",
    len(train_dataset)
)

print(
    "Test data:",
    len(test_dataset)
)

# %% [markdown]
# 정상적으로 설치되었다면 학습 데이터는 60000개, 테스트 데이터는 10000개가 존재할 것입니다.

# %% [markdown]
# #### $4)$ MNIST 데이터 확인

# %% [markdown]
# 설치된 MNIST 데이터를 확인해봅시다.

# %%
image, label = train_dataset[0]

print("Image type:", type(image))
print("Image shape:", image.shape)
print("Label:", label)
print("Minimum:", image.min().item())
print("Maximum:", image.max().item())

# %% [markdown]
# 해당 코드의 결과는 다음과 같아야 합니다.
#
# Image type: <class 'torch.Tensor'><br>
# Image shape: torch.Size([1, 28, 28])<br>
# Label: 5<br>
# Minimum: 0.0<br>
# Maximum: 1.0
#
# 특히, 이미지의 크기가 `1x28x28`인 것을 확인해야 합니다.

# %% [markdown]
# #### $5)$ MNIST 데이터 시각화

# %% [markdown]
# 우선, 위에서 간단한 확인을 위해 선택한 `train_data[0]`을 확인해봅시다.

# %%
image_2d = image.squeeze(0)

plt.figure(figsize=(4,4))
plt.imshow(image_2d, cmap="gray"), plt.title(f"Label: {label}")
plt.axis("off")

# %% [markdown]
# 이번에는 `train_data` 총 20개를 시각화해 봅시다.

# %%
plt.figure(figsize=(10,8))

for index in range(20):
    image, label = train_dataset[index]
    plt.subplot(4,5,index+1), plt.imshow(image.squeeze(0), cmap="gray"), plt.title(f"Label: {label}")
    plt.axis("off")

# %% [markdown]
# 여기서 알 수 있는 사실은:
# * 같은 숫자라도 사람마다 쓰는 방식이 다르다.
# * 숫자의 위치가 조금씩 다르다.
# * 획의 굵기와 기울기가 다르다.
# * 일부 숫자는 다른 숫자와 비슷하게 보일 수 있다.

# %% [markdown]
# #### $6)$ DataLoader 생성

# %% [markdown]
# DataLoader의 역할:
# * Dataset은 이미지 한 장과 Label 하나를 반환 (`image, label = train_dataset[0]`)
# * DataLoader는 총 70,000개의 데이터를 지정한 사이즈의 하나의 묶음(Batch)으로 불러와 모델에 전달
# * 실제 학습에서는 한 장씩 모델에 입력 값으로 전달하기 보다는, 여러 이미지를 하나의 Batch로 묶어서 입력에 전달한다
#
# DataLoader이 제공하는 기능:
# * Batch 생성
# * Batch 단위 반복
# * 데이터 순서 섞기
# * 데이터 로딩 과정 관리

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=0,
)

# %% [markdown]
# DataLoader가 묶은 Batch를 확인해봅시다.

# %%
images, labels = next(
    iter(train_loader)
)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

images, labels = next(
    iter(test_loader)
)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# %% [markdown]
# #### $7)$ CNN 모델 정의

# %% [markdown]
# 28x28 크기를 가진 MNIST 이미지를 활용하는 CNN의 전체 구조는 아래와 같습니다.

# %% [markdown]
# <div style="text-align: center;">
#   [입력]<br>
#   1 × 28 × 28<br>
#   ↓<br>
#   [Conv1]<br>
#   8 × 28 × 28<br>
#   ↓<br>
#   [ReLU]<br>
#   8 × 28 × 28<br>
#   ↓<br>
#   [Max Pooling]<br>
#   8 × 14 × 14<br>
#   ↓<br>
#   [Conv2]<br>
#   16 × 14 × 14<br>
#   ↓<br>
#   [ReLU]<br>
#   16 × 14 × 14<br>
#   ↓<br>
#   [Max Pooling]<br>
#   16 × 7 × 7<br>
#   ↓<br>
#   [Flatten]<br>
#   784<br>
#   ↓<br>
#   [Fully Connected]<br>
#   64<br>
#   ↓<br>
#   [Fully Connected]<br>
#   10
# </div>

# %% [markdown]
# CNN 모델 정의:

# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            10,
        )

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }

            return logits, features

        return logits

# %% [markdown]
# 위에서 정의한 CNN 모델 구조를 시각화해 봅시다. 그러기 위해서는 모델 시각화 도구인 `visualtorch`가 필요합니다.

# %% [markdown]
# ```bash
# pip install Pillow==10.4.0 --no-deps
# pip install aggdraw==1.3.19 --no-deps
# pip install visualtorch --no-deps
# ```

# %%
import visualtorch

# %% [markdown]
# Import가 오류 없이 되었다면 다음 코드를 통해 시각화해 봅시다.

# %%
def visualize_model(model, input_shape):
    model_img = visualtorch.render(
        model,
        input_shape=input_shape,
        style="flow",
        legend=True,
        scale_xy=4,
        min_xy=2,
        max_xy=500,
        )

    return model_img

# %%
model_struct = visualize_model(SimpleCNN(), (1,1,28,28))

plt.figure(figsize=(6,8))
plt.imshow(model_struct)
plt.axis("off");

# %% [markdown]
# #### $8)$ 모델 생성 및 CPU 지정

# %% [markdown]
# 위에 정의한 CNN 모델을 생성합시다. 이 모델이 학습되기 위해서는 CPU/GPU 사용 여부를 설정해야 합니다.

# %%
device = torch.device("cpu")

torch.manual_seed(seed)

model = SimpleCNN()
model = model.to(device)

print("Device:", device)
print(model)

# %% [markdown]
# 모델이 학습할 파라미터 수를 확인해봅시다.

# %%
parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total parameters:", parameter_count)
print("Trainable parameters:", trainable_parameter_count)

# %% [markdown]
# #### $9)$ 순전파 기능 확인

# %% [markdown]
# 실제 데이터를 학습하기 전에 임의의 입력(Dummy Input)을 모델에 전달하여 순전파(forward) 기능을 확인합시다.

# %%
dummy_input = torch.randn(1, 1, 28, 28)
dummy_output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", dummy_output.shape)

# %% [markdown]
# 입력 데이터는 MNIST의 기본 규격인 28×28 크기의 흑백(단일 채널) 이미지이며, 순전파 연산 결과 10개의 클래스(정수 0~9)로 출력됩니다.

# %% [markdown]
# 이번에는, Batch 하나를 불러와 순전파의 각 단계별 데이터의 크기를 확인해봅시다.

# %%
sample_images, sample_labels = next(
    iter(train_loader)
)

sample_images = sample_images.to(device)

with torch.no_grad():
    logits, features = model(
        sample_images,
        return_features=True,
    )

print("Input:", sample_images.shape)

for name, feature in features.items():
    print(f"{name}:", feature.shape)

print("Logits:", logits.shape)

# %% [markdown]
# #### $10)$ Feature Map 확인

# %% [markdown]
# CNN 각 층의 Feature Map을 확인하기 위한 함수를 정의합니다.

# %%
def show_feature_maps(feature_tensor, title, max_maps=8, cmap="gray"):
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")

    feature_maps = feature_tensor[0].detach().cpu()

    channel_count = min(feature_maps.shape[0], max_maps)
    column_count = 4
    row_count = int(np.ceil(channel_count / column_count))

    plt.figure(figsize=(12, 3 * row_count))

    for channel_index in range(channel_count):
        plt.subplot(row_count, column_count, channel_index + 1)
        plt.imshow(feature_maps[channel_index], cmap=cmap)
        plt.title(f"Channel {channel_index}")
        plt.axis("off")
        plt.suptitle(title)

# %% [markdown]
# 이제 입력 이미지 한 장을 선택하여 순전파를 적용해봅시다.
#
# 이미지 한 장만 선택하기에 Batch 차원이 없습니다. 그러므로 `unsqueeze(0)`를 사용하여 차원을 추가합니다.

# %%
image, label = test_dataset[0]

input_batch = image.unsqueeze(0).to(device)

print("Single image:", image.shape)
print("Batch image:", input_batch.shape)

# %% [markdown]
# 순전파 실행:

# %%
model.eval()

with torch.no_grad():
    logits, features = model(
        input_batch,
        return_features=True,
    )

# %% [markdown]
# 원본 이미지 출력:

# %%
plt.figure(figsize=(4,4))

plt.imshow(image.squeeze(0), cmap="gray"), plt.title(f"Input label: {label}")
plt.axis("off")

# %% [markdown]
# 첫 번째 Convolution 결과:

# %%
show_feature_maps(
    features["conv1"],
    title="Conv1 Feature Maps",
    max_maps=8,
)
for i in range(8):
    conv = features["conv1"][0, i]

    print(
        f"Channel {i}: "
        f"Conv min={conv.min().item():.6f}, "
        f"Conv max={conv.max().item():.6f}, "
        )

# %% [markdown]
# 학습 전 모델에서 Feature Map을 출력하면 Filter가 무작위로 초기화되어 있기 때문에 의미 있는 특징이 명확하게 나타나지 않습니다.
#
# 학습 후에는 숫자 분류에 유용한 선, 방향, 경계 등에 반응하는 Feature Map이 나타납니다.

# %% [markdown]
# 이번엔 ReLU까지 적용하여 확인해봅시다.
#
# 첫 번째 ReLU 결과:

# %%
show_feature_maps(
    features["relu1"],
    title="Conv1 + ReLU Feature Maps",
    max_maps=8,
)

# %% [markdown]
# ReLU를 적용하면 음수 반응(원치 않는 특성)은 모두 0으로 처리됩니다. 여전히 학습되기 전이기에 원하는 특성을 추출하는 가중치(필터)가 없어 큰 의미를 찾기는 어렵습니다.

# %% [markdown]
# 다음으로 Pooling을 적용해봅시다.
#
# 첫 번째 Pooling 결과:

# %%
show_feature_maps(
    features["pool1"],
    title="Pool1 Feature Maps",
    max_maps=8,
)

# %% [markdown]
# 이미지의 크기가 28x28에서 14x14로 줄어든 것을 확인할 수 있습니다.

# %% [markdown]
# 두 번째 Convolution 결과:

# %%
show_feature_maps(
    features["relu2"],
    title="Conv2 + ReLU Feature Maps",
    max_maps=12,
)

# %% [markdown]
# 두 번째 Layer는 첫 번째 Layer에서 추출한 특징들을 조합합니다.
#
# 따라서 단순한 Edge보다 숫자의 획이나 모양 일부에 반응하는 형태가 나타날 수 있습니다.

# %% [markdown]
# 두 번째 Pooling 결과:

# %%
show_feature_maps(
    features["pool2"],
    title="Pool2 Feature Maps",
    max_maps=12,
)

# %% [markdown]
# 최종 Feature Map 크기는 16×7×7 입니다.
#
# 이 특징들이 Flatten과 Fully Connected Layer로 전달됩니다.

# %% [markdown]
# #### $11)$ 가중치(필터) 확인

# %% [markdown]
# 첫 번째 Convolution Layer의 가중치 크기를 확인합시다.

# %%
conv1_weights = model.conv1.weight.detach().cpu()
print(conv1_weights.shape)

# %% [markdown]
# 3x3 필터 8개를 가중치로 가지고 있는 것을 확인할 수 있습니다.
#
# 필터들을 시각화해 봅시다.

# %%
plt.figure(figsize=(12,5))

for filter_index in range(conv1_weights.shape[0]):
    plt.subplot(2, 4, filter_index + 1)
    plt.imshow(conv1_weights[filter_index, 0], cmap="gray")
    plt.title(f"Filter {filter_index}")
    plt.axis("off")

# %% [markdown]
# 학습 전에는 무작위 Filter가 나타납니다.
#
# 학습 후 같은 코드를 다시 실행하면 가중치가 변경된 것을 확인할 수 있습니다.

# %% [markdown]
# #### $12)$ 손실 함수와 Optimizer 설정

# %% [markdown]
# MNIST는 10개 클래스 중 하나를 선택하는 다중 클래스 분류 문제이다. 해당 실습에서는 정답 분포와 예측 분포의 차이를 측정하여 다중 클래스를 분류하는 손실 함수인 Cross Entropy를 사용합니다.

# %%
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# CrossEntropyLoss에는 모델의 원시 출력인 Logit을 직접 입력합니다. 그러므로 학습 코드에서 Cross Entropy 앞에 Softmax를 별도로 적용하여 입력하지 않습니다.

# %% [markdown]
# Optimizer는 역전파로 계산된 Gradient를 사용하여 CNN의 Filter, Bias, Fully Connected Weight 등을 수정합니다.

# %%
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

# %% [markdown]
# #### $13)$ CNN 모델 학습

# %% [markdown]
# CNN 모델 학습은 Batch 단위로 진행됩니다.
#
# 하나의 Batch 학습 순서는 다음과 같습니다.
#
# 1. 이미지와 Label을 CPU로 이동
# 2. 기존 Gradient 초기화
# 3. CNN 순전파
# 4. Loss 계산
# 5. CNN 역전파
# 6. Weight 업데이트

# %% [markdown]
# 해당 과정을 함수로 정의합니다.

# %%
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # 모델을 학습 모드(training mode)로 변경
    model.train()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # 학습 데이터를 DataLoader(train_loader)의 Batch 단위로 불러와 학습 반복
    for images, labels in data_loader:
        # CPU로 이동
        images = images.to(device)
        labels = labels.to(device)

        # Gradient 초기화 (이전 Batch에서 계산한 Gradient를 제거)
        optimizer.zero_grad()

        # 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
        logits = model(images)

        # Loss 계산
        loss = criterion(logits, labels)
        # 역전파를 통한 Gradient 계산
        loss.backward()
        # Gradient를 사용하여 파라미터 수정
        optimizer.step()

        # Batch 한 개의 이미지 개수
        batch_size = images.size(0)

        # 결과 누적
        total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
        predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
        correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
        sample_count += batch_size                             # 전체 샘플 수 누적

    # 평균 Loss 및 Accuracy 계산
    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy

# %% [markdown]
# #### $14)$ CNN 모델 평가

# %% [markdown]
# 모델 학습이 완료되었다면, 제대로 학습이 되었는지 테스트 데이터로 평가를 해야합니다.
#
# 평가(테스트) 과정에서는 가중치를 수정하지 않습니다.
#
# 학습된 모델을 평가하는 함수를 정의합시다.

# %%
def evaluate(model, data_loader, criterion, device):
    # 모델을 평가 모드(evaluation mode)로 변경
    model.eval()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # Gradient 계산을 비활성화 (파라미터 수정 비활성화)
    with torch.no_grad():
        # 평가 데이터를 DataLoader(test_loader)의 Batch 단위로 불러와 평가 반복
        for images, labels in data_loader:
            # CPU로 이동
            images = images.to(device)
            labels = labels.to(device)

            # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
            logits = model(images)

            # Loss 계산 (역전파는 수행하지 않음)
            loss = criterion(logits, labels)

            # Batch 한 개의 이미지 개수
            batch_size = images.size(0)

            # 결과 누적
            total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
            predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
            correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
            sample_count += batch_size                             # 전체 샘플 수 누적

        # 평균 Loss 및 Accuracy 계산
        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy

# %% [markdown]
# #### $15)$ MNIST 숫자 손글씨 추론

# %% [markdown]
# 지금까지 MNIST 손글씨 숫자 분류 모델을 추론하기 위한 전체 구성 요소를 단계적으로 구현했습니다.
#
# 진행한 과정은 다음과 같습니다.
#
# - MNIST 데이터셋을 설치하고 학습 데이터와 테스트 데이터를 불러오기
# - DataLoader를 생성하여 데이터를 batch 단위로 처리할 수 있도록 구성
# - CNN 기반 분류 모델을 class 형태로 정의하고 Convolution, Pooling, Fully Connected Layer 설정
# - 정의한 CNN 모델을 불러와 생성하고 연산 장치(device)를 CPU로 지정
# - 모델 학습에 사용할 손실 함수와 optimizer 설정
# - CNN 모델의 학습 과정을 수행하는 `train_one_epoch()` 함수 정의
# - 학습된 모델의 성능을 평가하는 `evaluate()` 함수 정의
#
# 이번 단계에서는 앞에서 구현한 데이터 처리 과정, CNN 모델, loss 함수, optimizer, 학습 및 평가 함수를 연결하여 모델을 실제로 학습시키고, 테스트 데이터에 대한 성능을 확인한 뒤 새로운 손글씨 숫자를 예측합니다.

# %%
# 총 Epoch 정의 (전체 학습 데이터를 몇 번 반복해서 볼 것인지)
epochs = 3

# 학습 기록 저장 공간 준비
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# 전체 학습 시간 측정 시작
training_start_time = time.perf_counter()

for epoch in range(epochs):
    # Epoch 시작 시간 저장
    epoch_start_time = time.perf_counter()

    # 모델 학습
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    # 모델 평가
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    # 결과 저장 (학습 & 평가)
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    # Epoch 소요 시간 계산
    epoch_time = time.perf_counter() - epoch_start_time

    # 현재 Epoch의 학습/평가 결과 출력
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: "
        f"{train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

# 전체 학습 시간 계산 및 출력
total_training_time = time.perf_counter() - training_start_time
print(f"Total CPU training time: {total_training_time:.2f}s")

# 학습된 모델로 이미지 하나 테스트
image, true_label = test_dataset[0]          # 테스트할 이미지(image)와 정답(label)
input_batch = image.unsqueeze(0).to(device)  # Batch 차원 추가(unsqueeze)하여 입력 Batch로 저장 (이미지 1장)

# 모델을 평가 모드(evaluation mode)로 변경
model.eval()

# Gradient 계산을 비활성화 (파라미터 수정 비활성화)
with torch.no_grad():
    logits = model(input_batch)                           # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
    probabilities = torch.softmax(logits, dim=1)          # 원시 점수에 softmax 함수 적용 (클래스별 확률 계산)
    predicted_label = probabilities.argmax(dim=1).item()  # 정답 예측

# 결과 출력 (정답, 예측, confidence)
print("True label:", true_label)
print("Predicted label:", predicted_label)
print("Confidence:", probabilities[0, predicted_label].item())

# 결과 이미지 출력
plt.figure(figsize=(4,4))
plt.imshow(image.squeeze(0), cmap="gray")
plt.title(f"True: {true_label}, " f"Prediction: {predicted_label}")
plt.axis("off")
plt.show()

# %% [markdown]
# 이제 여러 장의 손글씨 숫자를 추론해 확인해봅시다.

# %%
plt.figure(figsize=(15,10))

num_images = 40

model.eval()

with torch.no_grad():
    for i in range(num_images):
        image, true_label = test_dataset[i]
        input_batch = image.unsqueeze(0).to(device)

        logits = model(input_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_label].item()

        plt.subplot(4, 10, i+1)
        plt.imshow(image.squeeze(0), cmap="gray")
        plt.title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label}\n"
            f"Conf: {confidence*100:.1f}%"
        )
        plt.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### $16)$ 모델 성능 확인

# %% [markdown]
# 위 과정에서, 학습이 진행됨에 따라 Epoch별 학습 및 평가의 Loss와 Accuracy를 리스트에 저장했습니다.
#
# 해당 지표들을 Epoch별 그래프로 시각화하여 확인해봅시다.

# %% [markdown]
# 우선 Loss 그래프입니다.

# %%
epoch_range = range(1, epochs + 1)

plt.figure(figsize=(8,5))

plt.plot(epoch_range, train_loss_history, marker="o", label="Train Loss")
plt.plot(epoch_range, test_loss_history, marker="o", label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.grid()
plt.legend()
plt.show()

# %% [markdown]
# 다음은 Accuracy 그래프입니다.

# %%
plt.figure(figsize=(8,5))

train_accuracy_percent = [accuracy * 100 for accuracy in train_accuracy_history]
test_accuracy_percent = [accuracy * 100 for accuracy in test_accuracy_history]

plt.plot(epoch_range, train_accuracy_percent, marker="o", label="Train Accuracy")
plt.plot(epoch_range, test_accuracy_percent, marker="o", label="Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy")
plt.grid()
plt.legend()
plt.show()

# %% [markdown]
# #### $17)$ 학습된 모델의 Feature Map 확인

# %% [markdown]
# 앞선 과정에서 모델 학습이 완료되었으니, 이제 각 층(layer)의 Feature Map을 다시 확인해봅시다. 이미 `show_feature_maps()` 함수를 위에서 정의했으니 활용하면 됩니다.

# %%
# CNN 각 층의 Feature Map 시각화 (cmap="viridis" 사용)

# 이미지 선택
image, true_label = test_dataset[0]
input_batch = image.unsqueeze(0).to(device)

# 역전파(파라미터 수정) 없이 순전파 수행
model.eval()

with torch.no_grad():
    logits, features = model(
        input_batch,
        return_features=True,
    )

# Conv1 Feature Maps 8개 시각화
show_feature_maps(
    features["conv1"],
    title="Conv1 Feature Maps (Trained)",
    max_maps=8,
    cmap="viridis",
)

# Conv2 Feature Maps 16개 시각화
show_feature_maps(
    features["conv2"],
    title="Conv2 Feature Maps (Trained)",
    max_maps=16,
    cmap="viridis",
)

# 6개 Layer별 대표 Feature Map 4개씩 비교
layer_names = [
    "conv1",
    "relu1",
    "pool1",
    "conv2",
    "relu2",
    "pool2"
]

for layer_name in layer_names:
    show_feature_maps(
        features[layer_name],
        title=f"{layer_name} Feature Maps (Trained)",
        max_maps=4,
        cmap="viridis",
    )

# %% [markdown]
# #### $18)$ 모델 저장 및 불러오기

# %% [markdown]
# 학습이 완료된 모델을 추후에 사용하기 위해 저장하여 불러올 수 있습니다. 이를 통해 오래 걸리는 학습 과정을 반복하지 않아도 됩니다.
#
# 모델을 저장하고 재사용한다는 것은 결국 학습된 가중치를 보존한다는 의미입니다.

# %% [markdown]
# 모델 저장:

# %%
torch.save(
    model.state_dict(),
    "src/models/MNIST/MNIST_CNN.pth",
)

# %% [markdown]
# 모델 불러오기:

# %%
loaded_model = SimpleCNN()

state_dict = torch.load(
    "src/models/MNIST/MNIST_CNN.pth",
    map_location="cpu",
    weights_only=True,
)

loaded_model.load_state_dict(state_dict)
loaded_model.to(device)
loaded_model.eval()

# %% [markdown]
# ---
