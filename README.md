# 6-DOF Robot Arm RL Training

Train a 6-DOF robot arm using reinforcement learning in Gazebo simulation.

![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)
![Gazebo](https://img.shields.io/badge/Gazebo-Fortress-orange)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-purple)

## What You Need

- **Ubuntu 22.04**
- **ROS2 Humble**
- **Gazebo Fortress**

## Installation

### Step 1: Install ROS2 and Gazebo

```bash
# Install ROS2 Humble
sudo apt update
sudo apt install ros-humble-desktop-full

# Install Gazebo and controllers
sudo apt install ros-humble-ros-gz ros-humble-gz-ros2-control
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install ros-humble-xacro python3-colcon-common-extensions

# Install Python packages
pip install torch numpy matplotlib pandas
```

### Step 2: Build the Project

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select robot_arm2
source install/setup.bash
```

> **Note**: Run the `source` commands every time you open a new terminal!

---

## How to Run

You need **2 terminals**.

### Terminal 1: Launch Simulation

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# For reaching training (robot reaches random targets)
ros2 launch robot_arm2 rl_training.launch.py

# OR for drawing training (robot draws shapes)
ros2 launch robot_arm2 drawing_training.launch.py
```

Wait ~10 seconds for Gazebo to fully load.

### Terminal 2: Start Training

```bash
cd ~/new_rl_ros2/ros2_ws/src/robot_arm2/scripts
source /opt/ros/humble/setup.bash
python3 train_robot.py
```

---

## Training Menu Options

```
======================================================================
🎮 TRAINING MENU
======================================================================
1. 🎮 Manual Test Mode (Verify environment)
2. 🤖 SAC Training (6-DOF Direct Control)
3. 🧠 SAC Training + Neural IK (3D Position Control)
4. 🧠 Train Neural IK Model
5. 🖋️ Drawing Task Training (SAC 6D Direct)
6. 🖋️ Drawing Task Training (SAC + Neural IK)
======================================================================
```

### Quick Start Guide

| What You Want | Launch File | Menu Option |
|---------------|-------------|-------------|
| Test manually | `rl_training.launch.py` | **1** |
| Train reaching (simple) | `rl_training.launch.py` | **2** |
| Train drawing (simple) | `drawing_training.launch.py` | **5** |

### Neural IK Options (Advanced)

Options **3** and **6** use Neural IK (robot controls XYZ position instead of joints).

⚠️ **You must train Neural IK first!**

```
Step 1: Run option 4 → Creates neural_ik.pth
Step 2: Then you can use option 3 or 6
```

---

## Two Training Scenarios

### 1. Reaching Training
- Robot learns to touch random target spheres
- Launch: `ros2 launch robot_arm2 rl_training.launch.py`
- Options: **2** (direct) or **3** (Neural IK)

### 2. Drawing Training  
- Robot learns to draw shapes (triangle, square, etc.)
- Launch: `ros2 launch robot_arm2 drawing_training.launch.py`
- Options: **5** (direct) or **6** (Neural IK)

---

## Training Results

Results are saved in `scripts/training_results/`:
- `png/` - Training graphs
- `csv/` - Data files  
- `pkl/` - Saved models

---

## Troubleshooting

### Gazebo won't open
Make sure you ran both source commands:
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### Robot not moving
```bash
ros2 control list_controllers
# Should show: joint_state_broadcaster [active], arm_controller [active]
```

### "Neural IK not found" error
You need to train Neural IK first (option 4) before using options 3 or 6.

---

## Project Structure

```
ros2_ws/src/robot_arm2/
├── launch/            # Simulation launch files
├── scripts/           # Training code
│   ├── train_robot.py      # Main training script
│   ├── rl/                 # RL environments
│   ├── agents/             # SAC agent
│   └── drawing/            # Drawing visualization
├── urdf/              # Robot description
└── worlds/            # Gazebo worlds
```

---

## License

MIT License

## Author

**ducanh** - [do010303](https://github.com/do010303)
