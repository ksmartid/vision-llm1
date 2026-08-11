# Vision-LLM-260810
[한기대] Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템

윈도우즈에서 테스트 하는 방법
1) 파워쉘
winget install --id Git.Git -e --source winget

2) 
git clone https://github.com/ksmartid/vision-llm1.git

cd vision-llm1
uv venv venv_jupyter

uv venv venv_jupyter --python 3.10 --system-site-packages

============= 윈도우즈 =========
source venv_jupyter/Scripts/activate
============= 윈도우즈 =========

uv pip install --python venv_jupyter/bin/python jupyter ipykernel notebook ipywidgets imutils

uv pip install --python venv_jupyter/bin/python "matplotlib-inline<0.2"

uv pip install --python venv_jupyter/bin/python --no-deps scikit-image==0.19.3 imageio==2.19.3 tifffile==2022.5.4 PyWavelets==1.3.0 networkx==2.8.8

uv pip install --python venv_jupyter/bin/python --no-deps mediapipe

uv pip install --python venv_jupyter/bin/python absl-py flatbuffers sounddevice

venv_jupyter/bin/python -m ipykernel install --user --name venv_jupyter --display-name "Python (venv_jupyter)"

venv_jupyter/bin/python "02_1_Computer-Vision_A이미지 조정.py"

