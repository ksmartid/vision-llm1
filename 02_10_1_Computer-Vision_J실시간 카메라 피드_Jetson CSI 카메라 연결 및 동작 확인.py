# -*- coding: utf-8 -*-
"""
Part J-1: 실시간 카메라 피드 - Jetson CSI 카메라 연결 및 동작 확인
원본: 02_Computer-Vision.ipynb (cell 213~225)
"""


# %% [markdown]
# ### J. 실시간 카메라 피드

# %% [markdown]
# #### $1)$ Jetson CSI 카메라 연결 및 동작 확인

# %% [markdown]
# 먼저 시스템에 인식된 카메라 장치 파일(/dev/video0, /dev/video1 등)을 확인합니다.

# %%
import glob
import subprocess

video_devices = sorted(glob.glob('/dev/video*'))
print("인식된 카메라 장치:", video_devices if video_devices else "없음")

# %% [markdown]
# 만약 `/dev/video*` 장치가 존재하지 않는다면, Jetson에서 카메라 인터페이스 설정이 활성화되어 있는지 확인해야 합니다. 아래 도구를 실행하여 CSI 카메라 설정을 활성화합니다.
#
# (설정 변경 + 재부팅이 필요한 작업이라 스크립트에서 자동 실행하지 않고 안내만 출력합니다.)

# %%
if not video_devices:
    print("카메라 장치가 없습니다. 터미널에서 아래 명령으로 CSI 카메라 인터페이스를 활성화하세요:")
    print("  sudo /opt/nvidia/jetson-io/jetson-io.py")
    print("활성화 후에는 재부팅이 필요합니다:")
    print("  sudo reboot")

# %% [markdown]
# 활성화가 되었다면 재부팅 후 카메라 장치를 다시 확인합니다.
#
# (재부팅은 스크립트에서 실행하지 않습니다. 재부팅 후 이 셀부터 다시 실행하세요.)

# %%
video_devices = sorted(glob.glob('/dev/video*'))
print("재확인된 카메라 장치:", video_devices if video_devices else "없음")

# %% [markdown]
# 터미널에 `/dev/video0` 혹은 `/dev/video1`이 출력된다면 정상적으로 카메라 장치가 인식이 되었다는 의미입니다.
#
# 실시간 카메라 피드를 확인해봅시다. (5초간 미리보기 후 자동 종료됩니다.)

# %%
if video_devices:
    preview_cmd = [
        "gst-launch-1.0", "nvarguscamerasrc", "sensor-id=0", "!",
        "nvvidconv", "!", "autovideosink"
    ]
    try:
        subprocess.run(preview_cmd, timeout=5)
    except subprocess.TimeoutExpired:
        print("5초간 카메라 프리뷰 테스트 완료 (정상 종료).")
    except FileNotFoundError:
        print("gst-launch-1.0 명령을 찾을 수 없습니다. GStreamer가 설치되어 있는지 확인하세요.")
    except Exception as e:
        print("카메라 프리뷰 테스트 중 에러 발생:", e)
        cam_error = e
else:
    print("카메라 장치가 없어 프리뷰 테스트를 건너뜁니다.")
    cam_error = None

# %% [markdown]
# 에러가 발생하며 카메라 영상 출력이 되지 않는다면 아래 명령어를 실행하고 다시 카메라 피드를 확인합시다.
#
# (`nvargus-daemon` 재시작은 sudo 권한이 필요해 스크립트에서 자동 실행하지 않습니다.)

# %%
if 'cam_error' in dir() and cam_error is not None:
    print("아래 명령으로 nvargus 데몬을 재시작한 뒤, 위 프리뷰 테스트 셀을 다시 실행하세요:")
    print("  sudo systemctl restart nvargus-daemon")

# %% [markdown]
# 카메라 영상이 확인이 된다면 다음으로 넘어가도록 합시다.
