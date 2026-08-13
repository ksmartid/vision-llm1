# -*- coding: utf-8 -*-
"""
Step 8/10: H. GPU 병렬 연산 (PyTorch)
원본: 03_DL-and-GPU.ipynb (단계별로 재구성)
"""

# %%
import torch
import matplotlib.pyplot as plt

# %% [markdown]
# ### H. GPU 병렬 연산 (PyTorch)

# %% [markdown]
# #### $1)$ PyTorch 개요

# %% [markdown]
# GPU를 활용하면 배열 연산뿐만 아니라 딥러닝 모델의 학습과 추론도 빠르게 수행할 수 있습니다.
#
# 이를 위해 GPU 연산, 자동 미분, 신경망 모델 구성을 지원하는 딥러닝 프레임워크인 `PyTorch`를 사용합니다.
#
# <br>
#
# `PyTorch`:
# * GPU 연산을 지원하는 오픈소스 딥러닝 프레임워크
# * NumPy 배열과 유사한 다차원 배열인 Tensor 사용
# * NVIDIA CUDA를 활용하여 딥러닝 연산을 GPU에서 가속
# * 자동 미분을 통해 신경망의 가중치와 편향을 학습
# * CNN, RNN, Transformer 등 다양한 딥러닝 모델 구현 가능
#
# <br>
#
# `PyTorch` 활용 사례:
# * 이미지 분류
# * 객체 검출
# * 이미지 Segmentation
# * 자세 추정
# * 자연어 처리
# * 음성 인식
# * 생성형 AI
# * 강화학습
# * 로봇 인지
# * Edge AI 추론
#
# <br>
#
# `PyTorch` 주요 기능:
# * Tensor 연산
# * 자동 미분
# * 신경망 Layer, 손실 함수, Optimizer 제공
# * 데이터 로딩 및 전처리
# * 모델 구성 및 학습
# * 모델 저장 및 불러오기

# %% [markdown]
# #### $2)$ PyTorch 기본 문법

# %% [markdown]
# **`PyTorch` 주요 구성 요소:**

# %% [markdown]
# * 다차원 배열
# ```text
#     torch.Tensor
# ```
#
# * 자동 미분과 Gradient 계산
# ```text
#     torch.autograd
#     torch.no_grad
#     torch.enable_grad
# ```
#
# * 신경망 Layer와 Loss Function
# ```text
#     torch.nn
#     torch.nn.functional
# ```
#
# * SGD, Adam 등의 Optimizer
# ```text
#     torch.optim
# ```
#
# * Dataset과 DataLoader
# ```text
#     torch.utils.data
# ```
#
# * PyTorch의 컴퓨터 비전 라이브러리 (이미지 Dataset, Transform, 모델)
# ```text
#     torchvision
# ```

# %% [markdown]
# **a) 기본적인 Import:**

# %%
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# %% [markdown]
# **b) CPU Tensor 생성:**

# %%
x_cpu = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]])

print(x_cpu)
print(x_cpu.device)

# %% [markdown]
# **c) GPU Tensor 생성:**

# %%
if torch.cuda.is_available():
    x_gpu = x_cpu.to("cuda")

    print(x_gpu)
    print(x_gpu.device)

# %% [markdown]
# **d) Device 설정:**

# %%
device = torch.device("cpu")

print("Selected device:", device)

# %%
device = torch.device("cuda")

print("Selected device:", device)

# %%
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("CUDA Availability:", torch.cuda.is_available())
print("Selected device:", device)

# %% [markdown]
# **e) 모델 생성 및 장치 할당:**

# %% [markdown]
# 앞선 실습에서 정의한 `SimpleCNN` 클래스를 다시 보도록 합시다.

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

# %%
# 모델 생성
model = SimpleCNN()

# 모델을 장치에 할당
model = model.to(device)

print("Device:", device)
print(model)

# %% [markdown]
# **f) 데이터셋 불러오기:**

# %% [markdown]
# 데이터셋의 원본은 이미지입니다. 이미지는 기본적으로 ‘높이 × 너비 × 채널’ 구조로 존재합니다. 하지만 PyTorch 모델은 계산 효율을 위해 ‘채널 × 높이 × 너비’ 형태로 입력을 받습니다.
#
# 또한, 원본 이미지는 0 ~ 255 사이의 정수값을 가지는 반면, PyTorch 모델은 학습의 안정성을 위해 0.0 ~ 1.0 사이의 실수값을 요구합니다.
#
# 이때 `transforms.ToTensor()`라는 변환기를 사용하면, 이미지 형식을 PyTorch 모델에 맞춘 최적의 형식으로 자동 변환할 수 있습니다.
#
# <br>
#
# 구체적으로는 다음 두 가지 중요한 일을 자동으로 처리합니다.
# * 형태 변환: 일반 이미지 파일 형식(H, W, C)을 PyTorch 연산 표준인 (C, H, W) 구조로 순서를 변경
# * 스케일링: 0~255 사이의 정수로 표현된 픽셀 값을 0.0~1.0 사이의 실수(Float32) 값으로 자동 변환(정규화)

# %% [markdown]
# 해당 과정은 딥러닝 모델의 안정적인 학습을 위해 필수입니다.

# %%
# 이미지 데이터를 PyTorch 모델이 학습할 수 있는 텐서(Tensor) 형태로 변환하는 변환기 생성
transform = transforms.ToTensor()

# %%
# 학습 데이터 불러오기
train_dataset = datasets.MNIST(
    root="src/datasets",  # 데이터셋을 저장 및 불러오기 위한 디렉토리
    train=True,           # 학습용 데이터만 불러오는 설정
    transform=transform,  # 텐서 변환기를 데이터셋에 적용
    download=True,        # 지정한 경로에 데이터가 없으면 자동으로 다운로드
)

# 테스트 데이터 불러오기
test_dataset = datasets.MNIST(
    root="src/datasets",  # 데이터셋 디렉토리
    train=False,          # 학습용/테스트용 데이터 로드 설정
    transform=transform,  # 텐서 변환기
    download=True,        # 최초 다운로드
)

# %% [markdown]
# **g) DataLoader 생성:**

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
# **h) 데이터셋의 단일 image/label 확인:**

# %%
image, label = train_dataset[0]

print("Image type:", type(image))
print("Image shape:", image.shape)
print("Label:", label)
print("Minimum:", image.min().item())
print("Maximum:", image.max().item())

image_2d = image.squeeze(0)

plt.imshow(image_2d, cmap="gray"), plt.title(f"Label: {label}")
plt.axis("off");

# %% [markdown]
# **i) 데이터셋을 GPU에 할당:**

# %% [markdown]
# DataLoader를 '반복자(Iterator)'로 변환한 뒤, 첫 번째 Batch만 불러오기:

# %%
images, labels = next(iter(train_loader))

print(type(images))
print(type(labels))
print()
print("이미지 Batch 형태:", images.shape)
print()
print("레이블 Batch 텐서 :\n", labels)

# %% [markdown]
# 이미지와 레이블 Batch 텐서를 장치(GPU)에 할당:

# %%
print("images 텐서의 과거 위치:", images.device)
print("labels 텐서의 과거 위치:", labels.device)

images = images.to(device)
labels = labels.to(device)

outputs = model(images)

print("모델 위치:", next(model.parameters()).device)
print("입력 이미지 위치:", images.device)
print("출력 위치:", outputs.device)

print("images 텐서의 현재 위치:", images.device)
print("labels 텐서의 현재 위치:", labels.device)

# %% [markdown]
# ***※ 주의: 모델과 Tensor는 같은 Device에 있어야 한다***

# %%
model = SimpleCNN()
model = model.to(device)

print("모델 위치:", device)

# %%
images, labels = next(iter(train_loader))

print("입력 이미지 텐서 위치:", images.device)

# %%
try:
    outputs = model(images)
except RuntimeError as e:
    print(f"예러: {e}")

# %% [markdown]
# ---
