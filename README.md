# 6-DOF Robot Arm RL Training - ROS2 Humble + Gazebo Fortress

A complete ROS2 workspace for training a 6-DOF robot arm with reinforcement learning in Gazebo Fortress, with deployment support for Raspberry Pi.

![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)
![Gazebo](https://img.shields.io/badge/Gazebo-Fortress-orange)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-purple)
![Python](https://img.shields.io/badge/Python-3.10+-green)

## ✨ Features

- ✅ **6-DOF Robot Arm** with full kinematics
- ✅ **Gazebo Fortress** integration with physics simulation
- ✅ **ros2_control** for position control of all joints
- ✅ **End-Effector Tracking** using TF2
- ⭐ **RL Training System** - TD3 and SAC agents with direct joint control
- ⭐ **Drawing Training** - Learn to draw shapes using Neural IK (3D position control)
- ⭐ **Pi Deployment** - Export trained models to Raspberry Pi

## 🤖 RL Training System

### Architecture

| Component | Description |
|-----------|-------------|
| **State** | 16D: joints(6), robot_xyz(3), target_xyz(3), dist(4) |
| **Action** | 6D: absolute joint angles (±90° / ±1.57 rad) |
| **Control** | Direct joint control (6D) OR Neural IK (3D position) |
| **Workspace** | 3D: X±24cm, Y=-35 to -5cm, Z=8-40cm |
| **Drawing** | Fixed Y-plane (Y=0.20m), follow 30-point shape trajectories |

### Quick Start

```bash
# Terminal 1: Launch Gazebo simulation
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch robot_arm2 rl_training.launch.py

# Terminal 2: Start training
cd ~/new_rl_ros2/ros2_ws/src/robot_arm2/scripts
python3 train_robot.py
```

### Training Menu

```
======================================================================
🎮 TRAINING MENU
======================================================================
1. Manual Test Mode (control_robot.py)
2. Reaching Training (TD3 Direct 6D)
3. Reaching Training (SAC Direct 6D)
4. Reaching Training (TD3 + Neural IK 3D)
5. Reaching Training (SAC + Neural IK 3D)
6. Train Neural IK Model Only (Supervised)
7. Drawing Training (SAC Direct 6D)
8. Drawing Training (SAC + Neural IK 3D)
======================================================================
```

### Training Results

Saved to `scripts/training_results/`:
- **png/**: Training plots
  - Reaching: Rewards, success rate, distance, losses
  - Drawing: + Waypoints reached (0-30), shape completion %, trajectory vs target
- **csv/**: Episode-by-episode metrics
- **pkl/**: Replay buffers (auto-cleaned, keeps best/final) - Separation for Reaching/Drawing modes

## 🍓 Raspberry Pi Deployment

Deploy trained models to Raspberry Pi for real robot control.

### Model Specs
- **State**: 16D (joints(6), robot_xyz(3), target_xyz(3), dist_xyz(3), dist_3d(1))
- **Action**: 6D absolute joint angles (±90° / ±1.57 rad)

### Quick Deploy

```bash
cd ros2_ws/src/robot_arm2/scripts/deployment
./deploy_to_pi.sh
```

### Manual Deployment

#### 1. Export Model to TFLite (on PC)
Since TFLite conversion requires specific TensorFlow versions incompatible with some system packages, we use a temporary virtual environment:

```bash
cd ros2_ws/src/robot_arm2/scripts/deployment

# 1. Create conversion environment (one-time setup)
python3 -m venv /tmp/tflite_env
/tmp/tflite_env/bin/pip install tensorflow onnx==1.12.0 onnx-tf

# 2. Export & Convert
# Step A: PyTorch -> ONNX
python3 pytorch_to_onnx.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth

# Step B: ONNX -> TFLite (using virtualenv)
/tmp/tflite_env/bin/python3 onnx_to_tflite.py --model ../checkpoints/sac_gazebo/actor_sac_best.onnx --quantize
# Output: ../checkpoints/sac_gazebo/actor_sac_best_quantized.tflite
```

#### 2. Copy to Pi & Run

```bash
# Easy way: use the script (prompts for IP)
./deploy_to_pi.sh

# Manual way:
scp ../checkpoints/sac_gazebo/actor_sac_best_quantized.tflite pi@<pi_ip>:~/rl_deployment/
python3 deploy_on_pi.py --model actor_sac_best_quantized.tflite
```

### Deployment Files

| File | Purpose |
|------|---------|
| `deploy_to_pi.sh` | One-command deployment via SSH |
| `export_tflite.py` | Convert PyTorch → TFLite (no ONNX) |
| `pi_servo_interface.py` | PCA9685 servo control node |
| `deploy_on_pi.py` | Run TFLite model on Pi |
| `fk_ik_utils.py` | Forward kinematics (required) |

## 🔧 Prerequisites

- **OS**: Ubuntu 22.04 LTS
- **ROS**: ROS2 Humble
- **Gazebo**: Gazebo Fortress 6.x
- **Python**: 3.10+

```bash
# Install dependencies
sudo apt install ros-humble-desktop-full
sudo apt install ros-humble-ros-gz ros-humble-gz-ros2-control
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install ros-humble-xacro python3-colcon-common-extensions

# Python ML dependencies
pip install torch numpy matplotlib pandas tensorflow
```

## 📦 Installation

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_arm2
source install/setup.bash
```

## 📁 Project Structure

```
new_rl_ros2/ros2_ws/src/robot_arm2/
├── config/               # Controller configurations
├── launch/               # Launch files
│   ├── rl_training.launch.py    # Main RL training launch
│   └── display.launch.py        # RViz visualization
├── meshes/               # Robot STL mesh files
├── models/               # Gazebo models (target_sphere, drawing_surface)
├── scripts/
│   ├── train_robot.py           # ⭐ Main training script
│   ├── control_robot.py         # 🎮 Manual control script
│   ├── target_manager.py        # Visual target teleportation
│   ├── agents/                  # TD3 and SAC implementations
│   ├── rl/                      # Environment and utilities
│   ├── deployment/              # 🍓 Pi deployment scripts
│   │   ├── deploy_to_pi.sh      # One-command deploy
│   │   ├── export_tflite.py     # PyTorch → TFLite
│   │   ├── pi_servo_interface.py
│   │   └── deploy_on_pi.py
│   ├── checkpoints/             # Saved model weights
│   └── training_results/        # Logs, plots, CSVs
├── urdf/                 # Robot description (URDF/Xacro)
└── worlds/               # Gazebo world files
```

## 🔧 Troubleshooting

### Meshes not loading in Gazebo
The launch file auto-sets `GZ_SIM_RESOURCE_PATH`. If issues persist:
```bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$(ros2 pkg prefix robot_arm2)/share
```

### Controllers not loading
```bash
ros2 control list_controllers
# Should show: joint_state_broadcaster [active], arm_controller [active]
```

### Training not improving
- Ensure replay buffer has enough samples (>1000)
- Check reward function parameters
- Try loading existing checkpoints: `python3 train_robot.py` then choose to load models

## 📝 License

MIT License

## 👤 Author

**ducanh** - [do010303](https://github.com/do010303)
