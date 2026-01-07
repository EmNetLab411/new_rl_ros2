#!/usr/bin/bash
set -e

echo "========================================================================"
echo "🚀 Deploying RL Model to Raspberry Pi (ROS2)"
echo "========================================================================"

# Configuration
PI_USER="pi"
PI_HOST="192.168.11.1"  # ⚠️ CHANGE THIS to your Pi's actual IP address!
PI_RL_DIR="/home/pi/rl_deployment"

# Model configuration (TFLite preferred - smaller and faster on Pi)
TFLITE_MODEL="checkpoints/sac_gazebo/actor_sac_best.tflite"
SCRIPTS_DIR="ros2_ws/src/robot_arm2/scripts"

echo ""
echo "📋 Deployment Configuration:"
echo "   Target: ${PI_USER}@${PI_HOST}"
echo "   RL Directory: ${PI_RL_DIR}"
echo "   Model: ${TFLITE_MODEL}"
echo ""

# Check if TFLite model exists
if [ ! -f "${TFLITE_MODEL}" ]; then
    echo "❌ ERROR: TFLite model not found at ${TFLITE_MODEL}"
    echo ""
    echo "📝 Please export your trained model first:"
    echo "   cd ${SCRIPTS_DIR}/deployment"
    echo "   python3 export_model.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth --format tflite"
    echo ""
    echo "   Or with quantization (smaller size):"
    echo "   python3 export_model.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth --format tflite --quantize"
    echo ""
    exit 1
fi

echo "✅ TFLite model found ($(du -h ${TFLITE_MODEL} | cut -f1))"

# ========================================================================
# Part 1: Deploy RL Model and Scripts
# ========================================================================
echo ""
echo "========================================================================"
echo "📦 Part 1: Deploying RL Model and Scripts"
echo "========================================================================"

# Test SSH connection
echo ""
echo "🔌 Testing SSH connection to ${PI_USER}@${PI_HOST}..."
if ! ssh -o ConnectTimeout=5 ${PI_USER}@${PI_HOST} "echo 'SSH OK'" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to ${PI_USER}@${PI_HOST}"
    echo "   Please check:"
    echo "     1. Pi is powered on and connected to network"
    echo "     2. SSH is enabled on Pi"
    echo "     3. IP address is correct (edit PI_HOST in this script)"
    exit 1
fi
echo "✅ SSH connection successful"

# Create target directory on Pi
echo ""
echo "📁 Creating RL deployment directory on Pi..."
ssh ${PI_USER}@${PI_HOST} "mkdir -p ${PI_RL_DIR}"

# Copy TFLite model
echo ""
echo "📦 Copying TFLite model..."
scp "${TFLITE_MODEL}" ${PI_USER}@${PI_HOST}:${PI_RL_DIR}/

# Copy deployment scripts
echo ""
echo "📦 Copying deployment scripts..."
scp ${SCRIPTS_DIR}/deployment/deploy_on_pi.py ${PI_USER}@${PI_HOST}:${PI_RL_DIR}/
scp ${SCRIPTS_DIR}/deployment/pi_servo_interface.py ${PI_USER}@${PI_HOST}:${PI_RL_DIR}/

# Copy FK/IK utilities if exists
if [ -f "${SCRIPTS_DIR}/rl/fk_ik_utils.py" ]; then
    echo ""
    echo "📦 Copying FK/IK utilities..."
    scp ${SCRIPTS_DIR}/rl/fk_ik_utils.py ${PI_USER}@${PI_HOST}:${PI_RL_DIR}/
fi

# Make scripts executable on Pi
echo ""
echo "🔧 Making scripts executable on Pi..."
ssh ${PI_USER}@${PI_HOST} "chmod +x ${PI_RL_DIR}/*.py"

echo ""
echo "✅ Part 1 Complete: RL model and scripts deployed"

# ========================================================================
# Part 2: Install Dependencies on Pi
# ========================================================================
echo ""
echo "========================================================================"
echo "📦 Part 2: Installing Dependencies on Pi"
echo "========================================================================"

echo ""
echo "📦 Installing tflite-runtime on Pi (if not present)..."
ssh ${PI_USER}@${PI_HOST} "pip3 install --quiet tflite-runtime 2>/dev/null || echo 'tflite-runtime may already be installed'"

echo ""
echo "📦 Installing servo libraries on Pi (if not present)..."
ssh ${PI_USER}@${PI_HOST} "pip3 install --quiet adafruit-circuitpython-servokit 2>/dev/null || echo 'servokit may already be installed'"

echo ""
echo "✅ Part 2 Complete: Dependencies installed"

# ========================================================================
# Deployment Complete
# ========================================================================
echo ""
echo "========================================================================"
echo "✅ Deployment Complete!"
echo "========================================================================"
echo ""
echo "📝 Files deployed to Pi (${PI_RL_DIR}):"
echo "   ✓ $(basename ${TFLITE_MODEL})"
echo "   ✓ deploy_on_pi.py"
echo "   ✓ pi_servo_interface.py"
echo ""
echo "========================================================================"
echo "🚀 Next Steps on Raspberry Pi"
echo "========================================================================"
echo ""
echo "  Terminal 1 (Servo Interface):"
echo "    cd ~/rl_deployment"
echo "    python3 pi_servo_interface.py"
echo ""
echo "  Terminal 2 (RL Deployment):"
echo "    cd ~/rl_deployment"
echo "    python3 deploy_on_pi.py --model $(basename ${TFLITE_MODEL})"
echo ""
echo "  Options:"
echo "    --target 0.15 0.0 0.30    # Custom target position (x y z)"
echo "    --episodes 5              # Number of episodes"
echo "    --steps 100               # Max steps per episode"
echo ""
echo "========================================================================"

