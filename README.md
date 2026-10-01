# Visual Servoing RL — 6-DOF Robot Arm

Deep Reinforcement Learning for precise reaching/drawing on a 6-DOF robotic arm, using visual servoing with ArUco marker detection in Gazebo.

---

## 📦 Setup from Scratch (Fresh Machine)

### Prerequisites

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| Gazebo | Harmonic (gz-sim8) |
| Python | 3.10+ |

> **Note**: Install ROS 2 Humble first: [docs.ros.org/en/humble/Installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

### Automatic Setup (Recommended)

```bash
git clone git@github.com:do010303/new_robot_rl_ros2.git ~/new_rl_ros2
cd ~/new_rl_ros2
chmod +x setup_workspace.sh
./setup_workspace.sh
```

The script will:
1. Install all ROS 2 / Gazebo apt packages (`ros-humble-ros-gzharmonic`, `ros2-control`, `cv-bridge`, etc.)
2. Clone external source dependencies via vcstool (`gz_ros2_control`)
3. Create a Python venv with all pip dependencies (torch, onnx, opencv, etc.)
4. Build the workspace with `colcon build`
5. Verify everything works

### Manual Setup (Step-by-step)

<details>
<summary>Click to expand manual steps</summary>

```bash
# 1. Install ROS 2 system packages
sudo apt-get update
sudo apt-get install -y \
    ros-humble-ros-gzharmonic \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-cv-bridge \
    ros-humble-rviz2 \
    python3-vcstool \
    python3-venv

# 2. Clone the repo
git clone git@github.com:do010303/new_robot_rl_ros2.git ~/new_rl_ros2
cd ~/new_rl_ros2

# 3. Clone external source dependencies
cd ros2_ws/src
vcs import --input deps.repos .
cd ../..

# 4. Python virtual environment
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U pip
pip install -r ros2_ws/src/visual_servoing/requirements.txt

# 5. Build
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```
</details>

---

# 🤖 9. Option 9 - Pi-Only Training (No Gazebo)

Train trực tiếp trên robot Pi, **không cần Gazebo**. Thích hợp cho máy mới hoặc máy không cài Gazebo.

```text
┌──────────┐  /pca9685_servo/joint_states  ┌──────────────┐
│    Pi     │ ────────────────────────────→ │   Laptop     │
│  (servo)  │                               │  train.py    │
│           │ ←──────────────────────────── │  option 9    │
└──────────┘  /pca9685_servo/trajectory     │  FK reward   │
                                            └──────────────┘
```

### Pre-trained models included

Repo đã bao gồm best models (~4MB), máy mới clone về là có sẵn:
- `checkpoints/neural_ik.pth` — Neural IK model
- `checkpoints/sac_pid_tuning_drawing_sim/*_best.*` — SAC best checkpoints (actor, critic, alpha)

### Cách chạy

**Bước 1: Pi — Khởi chạy servo node**

```bash
ssh piros2@192.168.50.1
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

**Bước 2: Laptop — Chạy Option 9**

```bash
cd ~/new_rl_ros2
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
cd ros2_ws/src/visual_servoing/scripts
python3 train_visual_servoing.py
```

Chọn **9**, sau đó chọn sub-option:
- `a` — PID Reaching (random targets)
- `b` — PID Drawing (shape waypoints) ← khuyến nghị
- `c` — Manual Control

---

# 🚀 8. Option 8 - Digital Twin Realtime Mirror

Chạy training + mirroring cùng lúc trong 1 terminal. Mirror chạy background bắt `/joint_states` từ Gazebo, training chạy foreground di chuyển robot trong sim. Khi robot sim cử động, robot thật trên Pi bám theo realtime.

```text
┌──────────┐   /joint_states   ┌──────────────────┐  /pca9685_servo/command  ┌─────┐
│  Gazebo  │ ───────────────→  │ sim_to_pi_mirror │ ───────────────────────→ │ Pi  │
│  + Train │   (radian, 50Hz)  │  (background)    │   (degree, 10Hz)        │     │
└──────────┘                   │  deadband 0.5°   │                         └─────┘
                               └──────────────────┘
```

### Environment Variables (BẮT BUỘC trên cả 2 máy)

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
```

### Bước 1: Pi — Khởi chạy node servo

```bash
ssh piros2@192.168.50.1
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

### Bước 2: Laptop Terminal 1 — Khởi chạy Gazebo

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
ros2 launch visual_servoing visual_servoing_test.launch.py
```

### Bước 3: Laptop Terminal 2 — Chạy Option 8 (mirror + training)

```bash
cd ~/new_rl_ros2/ros2_ws/src/visual_servoing/scripts
source /opt/ros/humble/setup.bash
source ~/new_rl_ros2/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
python3 train_visual_servoing.py
```

Chọn **8** từ menu. Script sẽ hỏi:

```text
Mirror publish rate Hz (default 10):     ← Enter để dùng 10Hz
Mirror deadband degrees (default 0.5):   ← Enter để dùng 0.5°
```

---


## Xem vùng làm việc tay mới (newarm) — trang tương tác

`docs/newarm_workspace_interactive.html` là một file HTML độc lập: không cần ROS, không cần cài gì, không cần server. Mở thẳng bằng trình duyệt:

```bash
xdg-open docs/newarm_workspace_interactive.html      # Linux
# hoặc kéo file vào Chrome/Firefox; trên Windows/macOS bấm đúp vào file
```

Chạy được khi không có mạng (chỉ thiếu font chữ, trình duyệt tự dùng font thay thế).

Trong trang:

- **4 thanh góc khớp J1–J4** (độ, kèm lệnh servo 0–180 tương ứng) — kéo để xem tay và đầu bút di chuyển.
- **Thiết lập bảng:** cửa sổ servo khuỷu (±90° hoặc [-30°,150°]), khoảng cách bảng tới trục J1, độ cao bảng, chiều dài bút, dung sai độ sâu.
- **Nút:** về home, chạm tâm bảng, tìm vị trí bảng tốt nhất, chạy thử hình vuông 10 cm.
- **3 khung nhìn:** nhìn thẳng vào bảng (đứng ở phía tay), nhìn ngang, và khung 3D xoay được bằng chuột (có sẵn các góc Chéo / Ngang / Từ trên / Từ sau tay; cung màu cam là góc J1).
- Vùng tô màu trên bảng là vùng vẽ được chắc chắn với thiết lập hiện tại.

FK/IK trong trang là bản JavaScript của `ros2_ws/src/visual_servoing/scripts/rl/fk_newarm.py` (đã so khớp với bản Python). Nếu đổi thiết kế tay thì trang này phải cập nhật theo. Mô phỏng Gazebo của tay mới: xem `CoVip/PLAN.md` mục 2c.

## Repo layout

- `ros2_ws/` — Gazebo + `visual_servoing` stack (laptop)
- `wicom_roboarm/` — Raspberry Pi hardware package (PCA9685 servo driver)
- `docs/` — documentation
- `ref/` — reference snapshots / notes (not meant for deployment)

## Overview

This project trains a robot arm to perform precise reaching and drawing tasks on a vertical board, using:
- SAC (Soft Actor-Critic) with HER (Hindsight Experience Replay)
- ArUco-based visual servoing for dynamic workspace detection
- Neural Inverse Kinematics (optional) for XYZ control
- Gazebo + `ros2_control` joint trajectory controller

### Architecture

```
Camera → ArUco Detection → Board Transform → RL Agent → Joint Commands → Robot
                ↓                                           ↑
         Board-relative                              Neural IK (optional)
         target generation                           Position → Joints
```

## Project structure (main package)

```
ros2_ws/src/visual_servoing/
├── launch/                       # main launch files
├── urdf/new_arm/                 # robot URDF / xacros
├── worlds/                       # Gazebo world(s)
├── models/                       # ArUco marker models
├── scripts/
│   ├── train_visual_servoing.py  # interactive training entrypoint
│   ├── rl/                       # environments + FK/IK utilities
│   └── drawing/                  # Gazebo drawing visualization
└── vs_lib/vision/                # camera viewer + ArUco detector
```

## Quick start (simulation)

Prepare Python first. The RL/drawing stack uses SciPy, Gymnasium, Torch, Matplotlib, and OpenCV ArUco on top of ROS 2:

```bash
cd /home/ducanh/new_rl_ros2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U pip
pip install -r ros2_ws/src/visual_servoing/requirements.txt
```

Build:

```bash
cd ros2_ws
colcon build --packages-select visual_servoing
source install/setup.bash
```

Launch simulation:

```bash
ros2 launch visual_servoing visual_servoing_test.launch.py
```

Run training menu (separate terminal):

```bash
cd /home/ducanh/new_rl_ros2
source .venv/bin/activate
cd ros2_ws/src/visual_servoing/scripts
python3 train_visual_servoing.py
```

## Square PID training

The most reliable square-learning path in the current codebase is:

1. Launch Gazebo with `ros2 launch visual_servoing visual_servoing_test.launch.py`
2. Run `python3 train_visual_servoing.py`
3. Choose `7` for PID tuning
4. Choose `b` for drawing
5. Choose backend `a` for `sim`
6. Answer `n` to live board detection unless your camera pipeline is already publishing `/vision/board_pose`

Without live board detection, the drawing environment now uses a deterministic fallback board plane in `base_link` at roughly `[-0.50, 0.0, 0.60]` instead of leaving square waypoints in raw board-local coordinates near the origin.

## Digital Twin (Sim-to-Real)

Sim-to-Real is implemented as **timed command mirroring** for *single-command* moves:
- Gazebo: timed `FollowJointTrajectory`
- Pi: `/pca9685_servo/trajectory` (`trajectory_msgs/JointTrajectory`) using `time_from_start`

Docs + tools:
- Full walkthrough: `docs/digital_twin_sim_to_real.md`
- Pi command cheatsheet: `docs/pi_robot_control_commands.md`
- Pi deploy helper: `scripts/deploy_pi_wicom_roboarm.sh`
- FastDDS SUPER_CLIENT template: `scripts/super_client.xml`
