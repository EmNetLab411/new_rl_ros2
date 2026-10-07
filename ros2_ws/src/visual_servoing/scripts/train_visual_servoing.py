#!/usr/bin/env python3
"""
Main RL Training Script for 6-DOF Robot Arm
Trains TD3+HER agent to reach target positions on drawing surface

Usage:
    python3 train_robot.py --episodes 500 --max-steps 10
"""

import os
import sys
import tempfile
# Suppress C++ TF_OLD_DATA warnings (harmless sim-time clock mismatch)
# Must be set BEFORE importing rclpy/tf2_ros
os.environ['TF2_CPP_LOGGING_LEVEL'] = 'ERROR'
os.environ['TF_CPP_LOG_LEVEL'] = 'ERROR'
_mpl_cache_dir = os.path.join(tempfile.gettempdir(), 'visual_servoing_mpl')
os.makedirs(_mpl_cache_dir, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', _mpl_cache_dir)

import rclpy
import numpy as np
import argparse
import time
import copy
from datetime import datetime
import torch

# Import RL components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl.rl_environment import RLEnvironment
from rl.drawing_environment import DrawingEnvironment  # Import Drawing Environment
# from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgentGazebo
from utils.her import her_augmentation
from rl.neural_ik import NeuralIK
from rl.control_backends import SUPPORTED_CONTROL_BACKENDS, resolve_control_backend
# PIDController removed - not used in training

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from drawing.drawing_config import SHAPE_TYPE, SHAPE_SIZE, X_PLANE


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Episode settings
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100
LEARNING_STARTS = 10

# Training settings
OPT_STEPS_PER_EPISODE = 64
SAVE_INTERVAL = 25
EVAL_INTERVAL = 10
MIN_EPISODES = 25

# HER (Hindsight Experience Replay) settings
HER_ENABLED = True
HER_K = 4
HER_STRATEGY = 'future'

# Reward settings (sparse)
GOAL_THRESHOLD = 0.0075  # 0.75cm
SUCCESS_REWARD = 0.0
STEP_PENALTY = -1.0

# Learning hyperparameters
ACTOR_LR = 0.001
CRITIC_LR = 0.002
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)
BATCH_OPT_STEPS = 64

# Auto-cleanup settings
MAX_BUFFER_FILES = 3      # Keep only N most recent buffer files (per type)
MAX_CHECKPOINT_FILES = 3  # Keep only N most recent checkpoints


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_old_files(directory: str, pattern: str, keep_count: int = 3, dry_run: bool = False):
    """
    Auto-cleanup old files, keeping only the most recent 'keep_count' files.

    Args:
        directory: Directory to clean
        pattern: Glob pattern for files (e.g., "*.pkl")
        keep_count: Number of files to keep
        dry_run: If True, only print what would be deleted

    Returns:
        Number of files deleted
    """
    import glob

    files = glob.glob(os.path.join(directory, pattern))
    if len(files) <= keep_count:
        return 0

    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)

    # Delete old files (keep the newest 'keep_count')
    files_to_delete = files[keep_count:]
    deleted_count = 0

    for f in files_to_delete:
        try:
            if dry_run:
                print(f"   [DRY RUN] Would delete: {f}")
            else:
                os.remove(f)
                deleted_count += 1
        except Exception as e:
            print(f"   ⚠️  Failed to delete {f}: {e}")

    return deleted_count


def _latest_file(directory: str, pattern: str):
    """Return most recent file matching pattern or None."""
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _capture_sac_snapshot(agent):
    """Capture an in-memory SAC snapshot for fast rollback after divergence."""
    snapshot = {
        'actor': copy.deepcopy(agent.actor.state_dict()),
        'critic1': copy.deepcopy(agent.critic1.state_dict()),
        'critic2': copy.deepcopy(agent.critic2.state_dict()),
        'critic1_target': copy.deepcopy(agent.critic1_target.state_dict()),
        'critic2_target': copy.deepcopy(agent.critic2_target.state_dict()),
        'actor_opt': copy.deepcopy(agent.actor_optimizer.state_dict()),
        'critic1_opt': copy.deepcopy(agent.critic1_optimizer.state_dict()),
        'critic2_opt': copy.deepcopy(agent.critic2_optimizer.state_dict()),
        'total_it': int(agent.total_it),
    }
    if agent.auto_entropy_tuning:
        snapshot['log_alpha'] = agent.log_alpha.detach().clone()
        snapshot['alpha'] = float(agent.alpha)
        snapshot['alpha_opt'] = copy.deepcopy(agent.alpha_optimizer.state_dict())
    return snapshot


def _restore_sac_snapshot(agent, snapshot):
    """Restore SAC networks/optimizers from an in-memory snapshot."""
    if not snapshot:
        return

    agent.actor.load_state_dict(snapshot['actor'])
    agent.critic1.load_state_dict(snapshot['critic1'])
    agent.critic2.load_state_dict(snapshot['critic2'])
    agent.critic1_target.load_state_dict(snapshot['critic1_target'])
    agent.critic2_target.load_state_dict(snapshot['critic2_target'])
    agent.actor_optimizer.load_state_dict(snapshot['actor_opt'])
    agent.critic1_optimizer.load_state_dict(snapshot['critic1_opt'])
    agent.critic2_optimizer.load_state_dict(snapshot['critic2_opt'])
    agent.total_it = int(snapshot.get('total_it', agent.total_it))

    if agent.auto_entropy_tuning and 'log_alpha' in snapshot:
        agent.log_alpha.data.copy_(snapshot['log_alpha'].to(agent.device))
        agent.alpha = float(snapshot['alpha'])
        agent.alpha_optimizer.load_state_dict(snapshot['alpha_opt'])

    agent.actor.train()
    agent.critic1.train()
    agent.critic2.train()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    """Main training function"""
    print("="*70)
    print("SAC+HER Training for 6-DOF Robot Arm")
    print("="*70)

    env = None  # Initialize env to prevent unbound error in finally block
    ros_initialized = False

    try:
        # Initialize ROS2
        rclpy.init()
        ros_initialized = True

        # Create environment (RLEnvironment for reaching task)
        print("\n📦 Creating RL environment (Reaching Mode)...")
        print(f"   Max steps: {args.max_steps}")
        env = RLEnvironment(
            max_episode_steps=args.max_steps,
            goal_tolerance=GOAL_THRESHOLD,
            control_backend=getattr(args, 'control_backend', None),
        )
        if not env.motion_backend.supports_reward_feedback:
            raise RuntimeError(
                "The selected backend does not provide authoritative reward feedback for SAC training. "
                "Use control_backend=sim or sim_to_real_shadow."
            )

        require_board_detection = bool(getattr(args, 'require_board_detection', True))
        if require_board_detection:
            print("📡 Enabling board-relative workspace...")
            env.enable_board_tracking()

        # Wait for environment to initialize
        print("   Waiting for environment...")
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)

        # Wait for initial board detection
        if require_board_detection:
            print("\n⏳ Waiting for ArUco board detection...")
            if not env.wait_for_initial_detection(timeout=10.0):
                print("⚠️  WARNING: No board detected! Training will use default workspace.")
                user_confirm = input("   Continue anyway? (y/n): ").strip().lower()
                if user_confirm != 'y':
                    print("❌ Training cancelled")
                    return
            else:
                print("✅ Board detected - targets will be board-relative")
        else:
            print("\n📡 Board detection: optional (skipped)")

        # Create agent based on selection
        print(f"\n🤖 Creating {args.agent.upper()} agent...")

        # Check if using Neural IK mode
        use_neural_ik = getattr(args, 'use_neural_ik', False)
        neural_ik = None

        if use_neural_ik:
            # Load Neural IK model
            nik_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'neural_ik.pth')
            if not os.path.exists(nik_path):
                print(f"\n❌ Neural IK model not found at: {nik_path}")
                print("   Please run option 6 first to train the Neural IK model!")
                return
            neural_ik = NeuralIK()
            neural_ik.load(nik_path)
            print(f"✅ Neural IK loaded from: {nik_path}")

            # 3D action space: normalized XYZ target position [-1, 1]
            action_dim = 3
            max_action = np.array([1.0, 1.0, 1.0])
            min_action = np.array([-1.0, -1.0, -1.0])
            print(f"   Using 3D Position Control (Neural IK converts to joints)")
        else:
            # 6D action space: absolute joint angles
            JOINT_LIMIT = np.pi / 2  # ±90° = ±1.57 rad
            action_dim = 6
            max_action = np.array([JOINT_LIMIT] * 6)
            min_action = np.array([-JOINT_LIMIT] * 6)
            print(f"   Using 6D Direct Joint Control")

        # Store neural_ik in args for training loop access
        args.neural_ik = neural_ik
        args.pid_controller = None

        if args.agent == 'sac':
            agent = SACAgentGazebo(
                state_dim=16,  # 16D observation
                n_actions=action_dim,
                max_action=max_action,
                min_action=min_action,
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                gamma=GAMMA,
                tau=TAU,
                batch_size=BATCH_SIZE,
                buffer_size=BUFFER_SIZE,
                auto_entropy_tuning=True
            )
            mode_str = "Neural IK 3D" if use_neural_ik else "Direct 6D"
            print(f"SAC Agent initialized ({mode_str} Control):")
            print(f"  State dim: 16, Action dim: {action_dim}")

        else:
             # Fallback or error if somehow another agent is passed (though parser restricts it)
             raise ValueError(f"Unknown agent: {args.agent}. Only 'sac' is supported.")

        # Override agent's checkpoint directory to be mode-specific
        # This ensures 3D (neural_ik) and 6D (direct) models are saved separately
        if use_neural_ik:
            agent.checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'{args.agent}_neural_ik')
        else:
            agent.checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'{args.agent}_direct')
        os.makedirs(agent.checkpoint_dir, exist_ok=True)
        print(f"  Checkpoint dir: {agent.checkpoint_dir}")

        # Ask to load existing replay buffer
        # Use mode-specific buffer patterns (3D neuralIK vs 6D direct are incompatible)
        mode_suffix = f"{args.agent}{'_neuralIK' if use_neural_ik else '_direct'}"
        load_buffer = input("\n📦 Load existing replay buffer? (y/n): ").strip().lower()
        if load_buffer == 'y':
            # Find available buffers for THIS MODE - prioritize BEST over FINAL
            import glob
            best_buffers = sorted(glob.glob(f"training_results/pkl/*best*{mode_suffix}*.pkl"), key=os.path.getmtime, reverse=True)
            final_buffers = sorted(glob.glob(f"training_results/pkl/*final*{mode_suffix}*.pkl"), key=os.path.getmtime, reverse=True)

            # Best buffers first, then final buffers
            buffer_files = best_buffers + final_buffers

            if buffer_files:
                print(f"   Found {len(best_buffers)} best buffers, {len(final_buffers)} final buffers")

                # Show top options
                if best_buffers:
                    print(f"   [BEST]  {best_buffers[0]}")
                if final_buffers:
                    print(f"   [FINAL] {final_buffers[0]}")

                # Default to best buffer if available, else final
                default_buffer = best_buffers[0] if best_buffers else final_buffers[0]
                buffer_path = input(f"   Enter path (Enter = {os.path.basename(default_buffer)}): ").strip()
                if buffer_path == '':
                    buffer_path = default_buffer
            else:
                print("   No buffer files found in training_results/pkl/")
                print("   Example: training_results/pkl/replay_buffer_best_20251231_143000.pkl")
                buffer_path = input("   Enter path (Enter = skip): ").strip()

            if buffer_path and os.path.exists(buffer_path):
                try:
                    agent.replay_buffer.load(buffer_path)
                    print(f"   ✅ Loaded replay buffer from: {buffer_path}")
                    print(f"   Buffer size: {agent.replay_buffer.size()}")
                except Exception as e:
                    print(f"   ❌ Failed to load buffer: {e}")
            elif buffer_path:
                print(f"   ❌ Buffer file not found: {buffer_path}")

        # Automatically try to load pre-trained models
        # This allows continuing training from previous checkpoint
        # Use agent.checkpoint_dir which was set based on mode (neural_ik vs direct)
        checkpoint_dir = agent.checkpoint_dir

        # Try to load models: best first, then fallback to latest
        # NOTE: SAC has dual critics (critic1, critic2) - the SAC agent's load_models()
        # automatically infers critic paths from actor path, so we only check actor
        best_actor_path = os.path.join(checkpoint_dir, f'actor_{args.agent}_best.pth')
        latest_actor_path = _latest_file(checkpoint_dir, 'actor_*_best.pth')
        if latest_actor_path is None:
            latest_actor_path = _latest_file(checkpoint_dir, 'actor_*.pth')

        # Choose best if exists, otherwise latest
        actor_path = best_actor_path if os.path.exists(best_actor_path) else latest_actor_path

        if actor_path and os.path.exists(actor_path):
            try:
                # SAC agent's load_models() infers critic1/critic2/alpha paths from actor path
                agent.load_models(actor_path)
                print(f"\n✅ Loaded pre-trained models from: {checkpoint_dir}")
                print(f"   Actor: {os.path.basename(actor_path)}")
                # Show inferred critic paths
                critic1_path = actor_path.replace('actor_', 'critic1_')
                if os.path.exists(critic1_path):
                    print(f"   Critic1: {os.path.basename(critic1_path)}")
                    print(f"   Critic2: {os.path.basename(actor_path.replace('actor_', 'critic2_'))}")
            except Exception as e:
                print(f"\n⚠️  Failed to load models: {e}")
                print("   Starting with untrained agent")
        else:
            print(f"\n📝 No pre-trained models found in {checkpoint_dir}/")
            print("   Starting with untrained agent")
        # ============================================================
        # LOAD PREVIOUS TRAINING RESULTS (for continuing plots)
        # ============================================================
        previous_results = None
        load_results = input("\n📊 Load previous training results? (y/n): ").strip().lower()
        if load_results == 'y':
            import glob
            import pickle

            # Find available training results files for THIS MODE
            pkl_search_dir = "training_results/pkl"
            results_files = sorted(glob.glob(f"{pkl_search_dir}/training_results*{mode_suffix}*.pkl"),
                                   key=os.path.getmtime, reverse=True)

            if results_files:
                print(f"   Found {len(results_files)} training results files:")
                for i, f in enumerate(results_files[:5]):  # Show top 5
                    print(f"   [{i+1}] {os.path.basename(f)}")

                default_file = results_files[0]
                results_path = input(f"   Enter path (Enter = {os.path.basename(default_file)}): ").strip()
                if results_path == '':
                    results_path = default_file

                if os.path.exists(results_path):
                    try:
                        with open(results_path, 'rb') as f:
                            previous_results = pickle.load(f)
                        print(f"   ✅ Loaded training results from: {results_path}")
                        print(f"   Previous episodes: {len(previous_results.get('episode_rewards', []))}")
                    except Exception as e:
                        print(f"   ❌ Failed to load results: {e}")
                        previous_results = None
                else:
                    print(f"   ❌ File not found: {results_path}")
            else:
                print(f"   ❌ No training results files found in {pkl_search_dir}/")

        # Training statistics - initialize from previous results if available
        if previous_results:
            episode_rewards = previous_results.get('episode_rewards', [])
            episode_successes = previous_results.get('episode_successes', [])
            episode_min_distances = previous_results.get('episode_min_distances', [])
            episode_steps = previous_results.get('episode_steps', [])
            actor_losses = previous_results.get('actor_losses', [])
            critic_losses = previous_results.get('critic_losses', [])

            # Load ALL-TIME best metrics (for cross-session comparison)
            best_min_distance = previous_results.get('best_min_distance', float('inf'))
            best_success_rate = previous_results.get('best_success_rate', 0.0)
            best_avg_reward = previous_results.get('best_avg_reward', -float('inf'))

            # If not saved before, calculate from data
            if best_min_distance == float('inf') and episode_min_distances:
                best_min_distance = min(episode_min_distances)
            if best_success_rate == 0.0 and episode_successes:
                best_success_rate = sum(episode_successes) / len(episode_successes)
            if best_avg_reward == -float('inf') and episode_rewards:
                best_avg_reward = max(episode_rewards)

            print(f"   📈 Continuing from episode {len(episode_rewards)}")
            print(f"   🏆 All-time best: Distance={best_min_distance*100:.2f}cm, Success={best_success_rate*100:.1f}%, Reward={best_avg_reward:.2f}")
        else:
            episode_rewards = []
            episode_successes = []
            episode_min_distances = []
            episode_steps = []  # Track steps per episode
            actor_losses = []
            critic_losses = []
            best_min_distance = float('inf')
            best_success_rate = 0.0
            best_avg_reward = -float('inf')

        # Create results directory structure
        results_dir = "training_results"
        csv_dir = f"{results_dir}/csv"
        pkl_dir = f"{results_dir}/pkl"
        png_dir = f"{results_dir}/png"
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(pkl_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print(f"\n📊 Training configuration:")
        print(f"   Episodes: {args.episodes}")
        print(f"   Max steps per episode: {args.max_steps}")
        print(f"   HER: {'Enabled' if HER_ENABLED else 'Disabled'} (k={HER_K})")
        print(f"   Results directory: {results_dir}")

        # Drawing visualization DISABLED for RL training (only for options 7/8)
        # from geometry_msgs.msg import Point
        # from std_srvs.srv import Empty
        pen_pub = None
        reset_line_client = None
        # print(f\"   ✏️ Drawing visualization enabled\")

        def finalize_training(interrupted=False):
            if len(episode_rewards) == 0:
                print("   No episodes completed, skipping saving/plotting.")
                return

            # Training complete - comprehensive summary
            print("\n" + "="*70)
            if interrupted:
                print("⚠️  TRAINING INTERRUPTED BY USER!")
            else:
                print("🎉 TRAINING COMPLETED!")
            print("="*70)

            # Overall statistics
            overall_avg_reward = np.mean(episode_rewards)
            overall_success_rate = np.mean(episode_successes)
            overall_avg_min_dist = np.mean(episode_min_distances)
            best_min_dist = min(episode_min_distances)

            print(f"\n📊 Overall Statistics ({len(episode_rewards)} episodes):")
            print(f"   Average Reward: {overall_avg_reward:.2f}")
            print(f"   Success Rate: {overall_success_rate*100:.1f}%")
            print(f"   Average Min Distance: {overall_avg_min_dist*100:.2f}cm")
            print(f"   Best Min Distance: {best_min_dist*100:.2f}cm")
            print(f"   Best Episode Reward: {max(episode_rewards):.2f}")
            print(f"   Worst Episode Reward: {min(episode_rewards):.2f}")

            # Loss statistics (if available)
            if actor_losses and any(l is not None for l in actor_losses):
                valid_actor_losses = [l for l in actor_losses if l is not None]
                valid_critic_losses = [l for l in critic_losses if l is not None]
                if valid_actor_losses:
                    print(f"\n📉 Training Losses:")
                    print(f"   Average Actor Loss: {np.mean(valid_actor_losses):.4f}")
                    print(f"   Average Critic Loss: {np.mean(valid_critic_losses):.4f}")

            # Plot training statistics (with distance data)
            # Create mode suffix for filenames (e.g., 'sac_neuralIK')
            curr_mode_suffix = f"{args.agent}{'_neuralIK' if use_neural_ik else '_direct'}"
            plot_training_stats(episode_rewards, episode_successes, episode_min_distances,
                               actor_losses, critic_losses, png_dir, csv_dir, timestamp, curr_mode_suffix, episode_steps)

            # Save final model
            agent.save_models()
            agent.replay_buffer.save(f'{pkl_dir}/replay_buffer_final_{curr_mode_suffix}_{timestamp}.pkl')
            print(f"\n💾 Final model saved")

            # Save training results (for continuing in future sessions)
            import pickle
            training_results = {
                'episode_rewards': episode_rewards,
                'episode_successes': episode_successes,
                'episode_min_distances': episode_min_distances,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses,
                # All-time best metrics (for cross-session comparison)
                'best_min_distance': best_min_distance,
                'best_success_rate': best_success_rate,
                'best_avg_reward': best_avg_reward
            }
            results_file = f'{pkl_dir}/training_results_{curr_mode_suffix}_{timestamp}.pkl'
            with open(results_file, 'wb') as f:
                pickle.dump(training_results, f)
            print(f"💾 Training results saved to: {results_file}")
            print(f"   Total episodes: {len(episode_rewards)}")

            # Final cleanup - mode-specific, keep only best and final buffers
            # Clean only THIS mode's buffers (4 periodic, 1 best, 1 final)
            cleanup_old_files(pkl_dir, f"replay_buffer_ep*{curr_mode_suffix}*.pkl", 4)  # Keep 4 periodic
            cleanup_old_files(pkl_dir, f"replay_buffer_best*{curr_mode_suffix}*.pkl", 1)  # Keep only 1 best
            cleanup_old_files(pkl_dir, f"replay_buffer_final*{curr_mode_suffix}*.pkl", 1)  # Keep only 1 final
            cleanup_old_files(pkl_dir, f"training_results*{curr_mode_suffix}*.pkl", 3)  # Keep 3 most recent results
            cleanup_old_files(png_dir, f"training_plot_{curr_mode_suffix}_*.png", 3)
            cleanup_old_files(csv_dir, f"training_data_{curr_mode_suffix}_*.csv", 3)
            print(f"🧹 Cleaned up old {curr_mode_suffix} files")

        # Training loop
        print("\n🚀 Starting training...\n")

        for episode in range(args.episodes):
            episode_start = time.time()

            # Reset environment
            state = env.reset_environment()

            # Reset drawing line at start of episode (only if enabled)
            if reset_line_client is not None and reset_line_client.wait_for_service(timeout_sec=0.5):
                from std_srvs.srv import Empty
                reset_line_client.call_async(Empty.Request())

            # Publish initial position (only if enabled)
            if pen_pub is not None and state is not None:
                from geometry_msgs.msg import Point
                ee = state[6:9]
                pen_pub.publish(Point(x=float(ee[0]), y=float(ee[1]), z=float(ee[2])))

            # Spin to process callbacks
            for _ in range(10):
                rclpy.spin_once(env, timeout_sec=0.1)

            if state is None:
                print(f"Episode {episode+1}: Failed to reset environment")
                continue

            # Episode buffer for HER
            episode_buffer = []
            episode_reward = 0.0
            episode_success = False

            # Reset PID controller for new episode
            if getattr(args, 'pid_controller', None) is not None:
                args.pid_controller.reset()

            # Episode loop
            min_distance = float('inf')

            for step in range(args.max_steps):
                # Select action
                action = agent.select_action(state, evaluate=False)

                # Extract current positions from state (before action)
                # State format: 6 joints + 3 EE + 3 target + 3 dist + 1 dist_3d + 1 ik + 6 vels
                ee_pos_before = state[6:9] if len(state) >= 9 else None
                target_pos = state[9:12] if len(state) >= 12 else None

                print(f"\n  ═══ Step {step+1}/{args.max_steps} ═══")
                if ee_pos_before is not None and target_pos is not None:
                    dist_before = np.linalg.norm(ee_pos_before - target_pos)
                    print(f"  📍 BEFORE: EE=[{ee_pos_before[0]:.4f}, {ee_pos_before[1]:.4f}, {ee_pos_before[2]:.4f}]")
                    print(f"  🎯 TARGET: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
                    print(f"  📏 Distance: {dist_before*100:.2f}cm")

                # Convert action if using Neural IK
                neural_ik = getattr(args, 'neural_ik', None)
                pid_controller = getattr(args, 'pid_controller', None)

                if neural_ik is not None:
                    # Task workspace in BASE_LINK frame
                    # Board at base_link X≈-0.50, Y≈0, Z≈0.56 (world X=0.50)
                    TASK_POS_MIN = np.array([-0.55, -0.10, 0.25])  # base_link coords
                    TASK_POS_MAX = np.array([-0.30,  0.10, 0.60])  # around the board

                    # ======= RESIDUAL RL: PID + SAC =======
                    if pid_controller is not None and ee_pos_before is not None and target_pos is not None:
                        # PID computes normalized baseline action toward target
                        pid_action = pid_controller.compute_normalized(
                            ee_pos_before, target_pos, TASK_POS_MIN, TASK_POS_MAX
                        )

                        # SAC outputs correction in [-1, 1]
                        sac_correction = action  # Already selected above

                        # Combine: PID baseline + small SAC correction (10%)
                        RESIDUAL_ALPHA = 0.1  # SAC contributes 10%
                        combined_action = pid_action + RESIDUAL_ALPHA * sac_correction
                        combined_action = np.clip(combined_action, -1.0, 1.0)

                        # Convert to XYZ target
                        target_xyz = (combined_action + 1) / 2 * (TASK_POS_MAX - TASK_POS_MIN) + TASK_POS_MIN
                        print(f"  🎛️  PID: [{pid_action[0]:.2f}, {pid_action[1]:.2f}, {pid_action[2]:.2f}]")
                        print(f"  🧠 SAC: [{sac_correction[0]:.2f}, {sac_correction[1]:.2f}, {sac_correction[2]:.2f}] × 0.1")
                    else:
                        # Pure SAC (no PID)
                        target_xyz = (action + 1) / 2 * (TASK_POS_MAX - TASK_POS_MIN) + TASK_POS_MIN

                    # Use Neural IK to get joint angles
                    joints_action = neural_ik.predict(target_xyz)
                    print(f"  🎯 Target: [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")
                    # Execute with joint angles
                    next_state, reward, done, info = env.step(joints_action)
                else:
                    # Direct 6D joint control
                    next_state, reward, done, info = env.step(action)

                # Spin to process callbacks
                for _ in range(5):
                    rclpy.spin_once(env, timeout_sec=0.1)

                # Extract positions after action
                if next_state is not None and len(next_state) >= 12:
                    ee_pos_after = next_state[6:9]
                    target_pos_after = next_state[9:12]
                    distance = np.linalg.norm(ee_pos_after - target_pos_after)
                    min_distance = min(min_distance, distance)

                    # Movement
                    if ee_pos_before is not None:
                        ee_movement = np.linalg.norm(ee_pos_after - ee_pos_before)
                        print(f"  📍 AFTER:  EE=[{ee_pos_after[0]:.4f}, {ee_pos_after[1]:.4f}, {ee_pos_after[2]:.4f}]")
                        print(f"  📏 EE moved: {ee_movement*100:.2f}cm")

                    print(f"  📏 Distance: {distance*100:.2f}cm (min: {min_distance*100:.2f}cm)")
                    print(f"  💰 Reward: {reward:.3f}")

                    if done and reward >= 0:  # Sparse: 0 = success
                        print(f"  🎉🎉🎉 SUCCESS! Goal reached! 🎉🎉🎉")

                    # Publish pen position for drawing line (only if enabled)
                    if pen_pub is not None:
                        from geometry_msgs.msg import Point
                        pen_pub.publish(Point(x=float(ee_pos_after[0]), y=float(ee_pos_after[1]), z=float(ee_pos_after[2])))

                if next_state is None:
                    print(f"   Step {step+1}: State unavailable, skipping")
                    break

                # Store transition
                goal = state[9:12]  # Target position from state
                episode_buffer.append((state, action, reward, next_state, done, goal))

                episode_reward += reward

                # Check success (reward is +100 on success)
                if done and reward >= 0:  # Sparse: 0 = success
                    episode_success = True

                state = next_state

                if done:
                    break

            # Store original transitions and apply HER augmentation
            if len(episode_buffer) > 0:
                # Unpack episode buffer into separate lists
                obs_list = [t[0] for t in episode_buffer]
                actions_list = [t[1] for t in episode_buffer]
                next_obs_list = [t[3] for t in episode_buffer]

                # Store original transitions first
                for transition in episode_buffer:
                    state_t, action_t, reward_t, next_state_t, done_t, _ = transition
                    agent.store_transition(state_t, action_t, reward_t, next_state_t, done_t)

                # HER augmentation - calls agent.remember() internally
                if HER_ENABLED:
                    her_augmentation(
                        agent=agent,
                        obs_list=obs_list,
                        actions_list=actions_list,
                        next_obs_list=next_obs_list,
                        k=HER_K,
                        strategy=HER_STRATEGY,
                        goal_threshold=GOAL_THRESHOLD
                    )

            # Training (after enough episodes)
            if episode >= LEARNING_STARTS:
                for _ in range(OPT_STEPS_PER_EPISODE):
                    actor_loss, critic_loss = agent.train()

                    # Store losses for plotting (only store last update per episode)
                    if _ == OPT_STEPS_PER_EPISODE - 1:
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
            else:
                actor_losses.append(None)
                critic_losses.append(None)

            # Log episode results
            episode_rewards.append(episode_reward)
            episode_successes.append(1.0 if episode_success else 0.0)
            episode_min_distances.append(min_distance)  # Track min distance
            episode_steps.append(step + 1)  # Track steps per episode

            # Calculate statistics (ALL episodes, not just last 10)
            avg_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_successes)
            avg_min_dist = np.mean(episode_min_distances)

            episode_time = time.time() - episode_start

            print(f"Episode {episode+1}/{args.episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"MinDist: {min_distance*100:.1f}cm | "
                  f"Success: {'✓' if episode_success else '✗'} | "
                  f"AvgReward: {avg_reward:.2f} | "
                  f"SuccessRate: {success_rate*100:.0f}% | "
                  f"Time: {episode_time:.1f}s")

            # Save best model (priority: distance > success_rate > reward)
            if episode >= MIN_EPISODES:
                is_new_best = False
                reason = ""

                # Priority 1: Best minimum distance (lower is better)
                if min_distance < best_min_distance:
                    is_new_best = True
                    reason = f"Best distance: {min_distance*100:.2f}cm (was {best_min_distance*100:.2f}cm)"
                    best_min_distance = min_distance
                # Priority 2: Best success rate (higher is better)
                elif success_rate > best_success_rate:
                    is_new_best = True
                    reason = f"Best success rate: {success_rate*100:.1f}% (was {best_success_rate*100:.1f}%)"
                    best_success_rate = success_rate
                # Priority 3: Best average reward (higher is better)
                elif avg_reward > best_avg_reward:
                    is_new_best = True
                    reason = f"Best avg reward: {avg_reward:.2f} (was {best_avg_reward:.2f})"
                    best_avg_reward = avg_reward

                if is_new_best:
                    agent.save_models()
                    agent.replay_buffer.save(f'{pkl_dir}/replay_buffer_best_{mode_suffix}_{timestamp}.pkl')
                    print(f"   💾 New best model! {reason}")

            # Periodic saves
            if (episode + 1) % SAVE_INTERVAL == 0:
                agent.save_models(episode=episode+1)
                agent.replay_buffer.save(f'{pkl_dir}/replay_buffer_ep{episode+1}_{mode_suffix}_{timestamp}.pkl')
                print(f"   💾 Checkpoint saved (episode {episode+1})")

        finalize_training(interrupted=False)
        print(f"\n✅ Training complete! Trained for {len(episode_rewards)} episodes.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        finalize_training(interrupted=True)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.destroy_node()
            except Exception as e:
                print(f"⚠️  Error destroying environment: {e}")
        if ros_initialized:
            try:
                rclpy.shutdown()
            except Exception:
                pass  # Ignore shutdown errors (RCL context already shutdown)


def plot_training_stats(episode_rewards, episode_successes, episode_min_distances, actor_losses, critic_losses, png_dir, csv_dir, timestamp, mode_suffix='', episode_steps=None):
    """Plot training statistics with cumulative moving averages including distance

    Args:
        episode_steps: List of steps per episode (optional, for steps-to-reach graph)
    """
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Calculate cumulative average (tracks all episodes up to current point)
    def cumulative_avg(data):
        return [np.mean(data[:i+1]) for i in range(len(data))]

    reward_avg = cumulative_avg(episode_rewards)
    success_avg = cumulative_avg(episode_successes)
    distance_avg = cumulative_avg(episode_min_distances)

    # Convert distances to cm
    distances_cm = [d * 100 for d in episode_min_distances]
    distance_avg_cm = [d * 100 for d in distance_avg]

    # Calculate steps average if available
    steps_avg = cumulative_avg(episode_steps) if episode_steps else None

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Title with mode info
    title = f'Training Statistics - {mode_suffix.upper().replace("_", " + ")}' if mode_suffix else 'Training Statistics'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Episode Rewards (top-left)
    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', linewidth=1.5, label='Episode Reward')
    ax.plot(episodes, reward_avg, color='darkblue', linewidth=3.0, label='Cumulative Average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Success Rate with X/O markers (top-center)
    ax = axes[0, 1]
    success_pct = np.array(success_avg) * 100

    # Separate success and fail episodes
    success_eps = [ep for ep, s in zip(episodes, episode_successes) if s == 1]
    fail_eps = [ep for ep, s in zip(episodes, episode_successes) if s == 0]
    success_y = [100 for _ in success_eps]  # Success at 100%
    fail_y = [0 for _ in fail_eps]  # Fail at 0%

    # Plot O for success, X for fail
    ax.scatter(success_eps, success_y, marker='o', color='green', s=30, alpha=0.6, label='Success')
    ax.scatter(fail_eps, fail_y, marker='x', color='red', s=30, alpha=0.6, label='Fail')
    # Moving average line
    ax.plot(episodes, success_pct, color='darkgreen', linewidth=3.0, label='20-Ep Average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success (1) / Fail (0)', fontsize=12)
    ax.set_title('Episode Success/Fail with Moving Average', fontsize=14, fontweight='bold')
    ax.set_ylim([-5, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Min Distance to Target (top-right)
    ax = axes[0, 2]
    ax.plot(episodes, distances_cm, alpha=0.3, color='orange', linewidth=1.5, label='Episode Min Distance')
    ax.plot(episodes, distance_avg_cm, color='darkorange', linewidth=3.0, label='Cumulative Average')
    ax.axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Goal (0.75cm)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Distance (cm)', fontsize=12)
    ax.set_title('Min Distance to Target', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Combined Training Losses (bottom-left)
    ax = axes[1, 0]
    valid_actor = [(i+1, l) for i, l in enumerate(actor_losses) if l is not None]
    valid_critic = [(i+1, l) for i, l in enumerate(critic_losses) if l is not None]

    if valid_actor:
        actor_eps, actor_vals = zip(*valid_actor)
        ax.plot(actor_eps, actor_vals, color='blue', linewidth=1.5, alpha=0.8, label='Actor Loss')
    if valid_critic:
        critic_eps, critic_vals = zip(*valid_critic)
        ax.plot(critic_eps, critic_vals, color='orange', linewidth=1.5, alpha=0.8, label='Critic Loss')

    if valid_actor or valid_critic:
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', fontsize=12)
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')

    # Plot 5: Steps to Reach Target (bottom-center)
    ax = axes[1, 1]
    if episode_steps and len(episode_steps) > 0:
        ax.plot(episodes[:len(episode_steps)], episode_steps, alpha=0.3, color='purple', linewidth=1.5, label='Steps per Episode')
        ax.plot(episodes[:len(steps_avg)], steps_avg, color='darkviolet', linewidth=3.0, label='Cumulative Average')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Steps', fontsize=12)
        ax.set_title('Steps to Reach Target', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Steps Data', ha='center', va='center', fontsize=12)
        ax.set_title('Steps to Reach Target', fontsize=14, fontweight='bold')

    # Plot 6: Combined Summary (bottom-right)
    ax = axes[1, 2]
    ax.axis('off')
    steps_text = f"  • Avg Steps: {np.mean(episode_steps):.1f}" if episode_steps else "  • Avg Steps: N/A"
    summary_text = f"""
📊 Training Summary
━━━━━━━━━━━━━━━━━━━━

Episodes: {len(episode_rewards)}

Rewards:
  • Final Avg: {reward_avg[-1]:.2f}
  • Best: {max(episode_rewards):.2f}

Success Rate:
  • Final: {success_pct[-1]:.1f}%

Distance to Target:
  • Final Avg: {distance_avg_cm[-1]:.2f}cm
  • Best: {min(distances_cm):.2f}cm

Steps:
{steps_text}
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()

    # Save plot with mode suffix in filename
    filename_suffix = f'_{mode_suffix}' if mode_suffix else ''
    plot_path = f'{png_dir}/training_plot{filename_suffix}_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Training plot saved to: {plot_path}")

    # Save CSV with mode suffix
    import csv
    csv_path = f'{csv_dir}/training_data{filename_suffix}_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Success', 'MinDistance_cm', 'Actor_Loss', 'Critic_Loss'])
        for i in range(len(episode_rewards)):
            actor_loss = actor_losses[i] if i < len(actor_losses) and actor_losses[i] is not None else ''
            critic_loss = critic_losses[i] if i < len(critic_losses) and critic_losses[i] is not None else ''
            min_dist = episode_min_distances[i] * 100 if i < len(episode_min_distances) else ''
            writer.writerow([
                i+1,
                f'{episode_rewards[i]:.3f}',
                int(episode_successes[i]),
                f'{min_dist:.3f}' if min_dist != '' else '',
                f'{actor_loss:.6f}' if actor_loss != '' else '',
                f'{critic_loss:.6f}' if critic_loss != '' else ''
            ])

    print(f"📊 Training data saved to: {csv_path}")


def plot_drawing_stats(episode_rewards, waypoints_reached, shape_completions,
                       actor_losses, critic_losses, episode_trajectories,
                       target_waypoints, mode_suffix='drawing'):
    """
    Plot training statistics for drawing task.

    Creates 6 subplots:
    1. Episode Rewards
    2. Waypoints Reached per Episode (Y: 0-30, X: episode)
    3. Shape Completion Rate
    4. Training Losses
    5. Trajectory Visualization vs Target Triangle
    6. Summary Stats
    """
    import matplotlib.pyplot as plt
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directories
    png_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'png')
    csv_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'csv')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    episodes = list(range(1, len(episode_rewards) + 1))

    # Cumulative averages
    def cumulative_avg(data):
        return [np.mean(data[:i+1]) for i in range(len(data))]

    reward_avg = cumulative_avg(episode_rewards)
    waypoints_avg = cumulative_avg(waypoints_reached)

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    title = f'Drawing Training Statistics - {mode_suffix.upper().replace("_", " + ")}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Episode Rewards (top-left)
    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', linewidth=1.5, label='Episode Reward')
    ax.plot(episodes, reward_avg, color='darkblue', linewidth=3.0, label='Cumulative Average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Waypoints Reached (top-center)
    ax = axes[0, 1]
    # Get total waypoints from config
    from drawing.drawing_config import TOTAL_WAYPOINTS
    total_wp = TOTAL_WAYPOINTS
    ax.scatter(episodes, waypoints_reached, marker='o', color='green', s=40, alpha=0.6, label='Waypoints')
    ax.plot(episodes, waypoints_avg, color='darkgreen', linewidth=3.0, label='Cumulative Average')
    ax.axhline(y=total_wp, color='gold', linestyle='--', linewidth=2, label=f'Target ({total_wp})')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Waypoints Reached', fontsize=12)
    ax.set_title('Waypoints Reached per Episode', fontsize=14, fontweight='bold')
    ax.set_ylim([-1, total_wp + 2])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Shape Completion Rate (top-right)
    ax = axes[0, 2]
    completion_pct = [100.0 if c else 0.0 for c in shape_completions]
    completion_avg = cumulative_avg([1.0 if c else 0.0 for c in shape_completions])
    completion_avg_pct = [c * 100 for c in completion_avg]

    # O for complete, X for incomplete
    complete_eps = [ep for ep, c in zip(episodes, shape_completions) if c]
    incomplete_eps = [ep for ep, c in zip(episodes, shape_completions) if not c]

    ax.scatter(complete_eps, [100]*len(complete_eps), marker='o', color='green', s=40, alpha=0.6, label='Complete')
    ax.scatter(incomplete_eps, [0]*len(incomplete_eps), marker='x', color='red', s=40, alpha=0.6, label='Incomplete')
    ax.plot(episodes, completion_avg_pct, color='darkgreen', linewidth=3.0, label='Completion Rate %')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Completion (%)', fontsize=12)
    ax.set_title('Shape Completion Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([-5, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Training Losses (bottom-left)
    ax = axes[1, 0]
    valid_actor = [(i+1, l) for i, l in enumerate(actor_losses) if l is not None]
    valid_critic = [(i+1, l) for i, l in enumerate(critic_losses) if l is not None]

    if valid_actor:
        actor_eps, actor_vals = zip(*valid_actor)
        ax.plot(actor_eps, actor_vals, color='blue', linewidth=1.5, alpha=0.8, label='Actor Loss')
    if valid_critic:
        critic_eps, critic_vals = zip(*valid_critic)
        ax.plot(critic_eps, critic_vals, color='orange', linewidth=1.5, alpha=0.8, label='Critic Loss')

    if valid_actor or valid_critic:
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', fontsize=12)
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')

    # Plot 5: 3D Trajectory Visualization (bottom-center)
    # Style: Fixed target triangle (orange line) vs Actual trajectory (blue points)
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(2, 3, 5, projection='3d')

    # FIXED target triangle (15cm triangle at Y=20cm, centered at X=0, Z=25cm)
    import math
    size_cm = 15.0  # 15cm triangle (matches training)
    height_cm = size_cm * math.sqrt(3) / 2  # ~13cm
    cx, cy, cz = 0.0, 20.0, 25.0  # Center in cm (Y is the plane)

    # Triangle corners (in cm) - X, Y, Z
    triangle_x = [cx - size_cm/2, cx, cx + size_cm/2, cx - size_cm/2]
    triangle_y = [cy, cy, cy, cy]  # All same Y (drawing plane)
    triangle_z = [cz - height_cm/3, cz + 2*height_cm/3, cz - height_cm/3, cz - height_cm/3]

    # Draw fixed target triangle (orange)
    ax.plot(triangle_x, triangle_y, triangle_z, 'o-', color='orange', linewidth=3,
            markersize=10, label='Target Triangle', zorder=10)

    # Draw actual trajectory from ALL episodes
    if episode_trajectories and len(episode_trajectories) > 0:
        # Scatter plot for ALL episodes (light blue) to show density
        all_x, all_y, all_z = [], [], []
        for traj in episode_trajectories:
            if traj and len(traj) > 0:
                for pt in traj:
                    all_x.append(pt[0] * 100)
                    all_y.append(pt[1] * 100)
                    all_z.append(pt[2] * 100)

        if all_x:
            ax.scatter(all_x, all_y, all_z, c='blue', alpha=0.3, s=5, label='Actual Path')

    ax.set_xlabel('X (cm)', fontsize=10)
    ax.set_ylabel('Y (cm)', fontsize=10)
    ax.set_zlabel('Z (cm)', fontsize=10)
    ax.set_title('3D Trajectory vs Target', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

    # Plot 6: Summary Stats (bottom-right)
    ax = axes[1, 2]
    ax.axis('off')

    num_complete = sum(shape_completions)
    completion_rate = 100.0 * num_complete / len(shape_completions) if shape_completions else 0
    best_waypoints = max(waypoints_reached) if waypoints_reached else 0
    avg_waypoints = np.mean(waypoints_reached) if waypoints_reached else 0

    summary_text = f"""
📊 Drawing Training Summary
━━━━━━━━━━━━━━━━━━━━━━━━━

Episodes: {len(episode_rewards)}

Rewards:
  • Final Avg: {reward_avg[-1]:.2f}
  • Best: {max(episode_rewards):.2f}

Waypoints:
  • Best: {best_waypoints}/{total_wp}
  • Avg: {avg_waypoints:.1f}/{total_wp}

Shape Completion:
  • Completed: {num_complete}/{len(shape_completions)}
  • Rate: {completion_rate:.1f}%
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()

    # Save plot
    plot_path = f'{png_dir}/drawing_training_{mode_suffix}_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Drawing training plot saved to: {plot_path}")

    # Save CSV
    import csv
    csv_path = f'{csv_dir}/drawing_training_{mode_suffix}_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Waypoints_Reached', 'Shape_Complete', 'Actor_Loss', 'Critic_Loss'])
        for i in range(len(episode_rewards)):
            actor_loss = actor_losses[i] if i < len(actor_losses) and actor_losses[i] is not None else ''
            critic_loss = critic_losses[i] if i < len(critic_losses) and critic_losses[i] is not None else ''
            writer.writerow([
                i+1,
                f'{episode_rewards[i]:.3f}',
                waypoints_reached[i],
                int(shape_completions[i]),
                f'{actor_loss:.6f}' if actor_loss != '' else '',
                f'{critic_loss:.6f}' if critic_loss != '' else ''
            ])

    print(f"📊 Drawing training data saved to: {csv_path}")


def evaluate(env, agent, num_episodes=3):
    """Evaluate agent without exploration noise"""
    total_reward = 0.0
    total_success = 0.0

    for ep in range(num_episodes):
        state = env.reset_environment()

        # Spin to process callbacks
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)

        if state is None:
            continue

        ep_reward = 0.0
        ep_success = False

        for step in range(10):
            action = agent.select_action(state, evaluate=True)  # No noise
            next_state, reward, done, info = env.step(action)

            # Spin
            for _ in range(5):
                rclpy.spin_once(env, timeout_sec=0.1)

            if next_state is None:
                break

            ep_reward += reward

            if done and reward > 5.0:
                ep_success = True

            state = next_state

            if done:
                break

        total_reward += ep_reward
        total_success += (1.0 if ep_success else 0.0)

    avg_reward = total_reward / num_episodes
    avg_success = total_success / num_episodes

    return avg_reward, avg_success


def manual_control_mode(control_backend=None):
    """
    Manual control mode - enter joint angles to move robot.
    Uses the RL environment for robot communication.
    """
    print("\n" + "=" * 70)
    print("🎮 MANUAL CONTROL MODE")
    print("=" * 70)
    print("Commands:")
    print("  Enter 6 joint angles in DEGREES: e.g., '0 0 45 0 0 0'")
    print("  (Paste from filtered_step_log.txt 'CMD' line)")
    print("  'home' or 'h' - Move to home position (0,0,0,0,0,0)")
    print("  'up' - Move arm up (0,45,45,0,0,0)")
    print("  'forward' - Extend forward (0,30,60,0,-30,0)")
    print("  'draw' - Toggle drawing mode (publishes pen position)")
    print("  'reset' - Reset drawing line in Gazebo")
    print("  'fk' - Show current FK position")
    print("  'quit' or 'q' - Exit manual mode")
    print("=" * 70)

    env = None
    ros_initialized = False

    try:
        # Initialize ROS2
        rclpy.init()
        ros_initialized = True

        # Create environment
        print("\n📦 Creating environment...")

        # Use RLEnvironment but logging implies we want to verify drawing consistency
        env = RLEnvironment(
            max_episode_steps=100,
            goal_tolerance=0.01,
            control_backend=control_backend,
        )

        # Wait for initialization
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)

        print("✅ Environment ready!")
        if getattr(env, 'digital_twin_enabled', False):
            print("🔄 Digital twin sync: ON (sim_to_real_shadow)")
            print("   Safe whole-move commands can mirror/replay on the Pi")
        else:
            print("🔄 Digital twin sync: OFF")

        # Import FK for position calculation
        from rl.fk_ik_utils import fk
        from geometry_msgs.msg import Point

        # Create pen position publisher for drawing line
        pen_pub = env.create_publisher(Point, '/drawing/pen_position', 10)
        drawing_enabled = True  # Start with drawing enabled
        print("✏️  Drawing mode: ON (pen position will be published)")

        # Publish initial position so first movement draws a line
        init_state = env.get_state()
        if init_state is not None and drawing_enabled:
            ee = init_state[6:9]
            pen_msg = Point(x=float(ee[0]), y=float(ee[1]), z=float(ee[2]))
            pen_pub.publish(pen_msg)
            print(f"✏️  Initial position: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")

        while True:
            try:
                # Show current state
                state = env.get_state()
                if state is not None:
                    current_joints_rad = state[:6]
                    current_joints_deg = np.degrees(current_joints_rad)
                    ee_pos = state[6:9]
                    print(f"\n📍 Current joints (deg): [{current_joints_deg[0]:.1f}, {current_joints_deg[1]:.1f}, "
                          f"{current_joints_deg[2]:.1f}, {current_joints_deg[3]:.1f}, {current_joints_deg[4]:.1f}, "
                          f"{current_joints_deg[5]:.1f}]")
                    print(f"📍 Current EE (Actual): ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")

                cmd = input("\n🤖 Enter command: ").strip().lower()

                if cmd in ['quit', 'q', 'exit']:
                    print("👋 Exiting manual mode...")
                    break

                elif cmd in ['home', 'h']:
                    joints_deg = [0, 0, 0, 0, 0, 0]
                    print("🏠 Moving to home position...")

                elif cmd == 'up':
                    joints_deg = [0, 45, 45, 0, 0, 0]
                    print("⬆️ Moving arm up...")

                elif cmd == 'forward':
                    joints_deg = [0, 30, 60, 0, -30, 0]
                    print("➡️ Extending forward...")

                elif cmd == 'draw':
                    drawing_enabled = not drawing_enabled
                    status = "ON" if drawing_enabled else "OFF"
                    print(f"✏️  Drawing mode: {status}")
                    continue

                elif cmd == 'reset':
                    # Reset the drawing line
                    from std_srvs.srv import Empty
                    reset_client = env.create_client(Empty, '/drawing/reset_line')
                    if reset_client.wait_for_service(timeout_sec=1.0):
                        reset_client.call_async(Empty.Request())
                        print("🔄 Drawing line reset!")
                    else:
                        print("⚠️  Reset service not available")
                    continue

                elif cmd == 'fk':
                    if state is not None:
                        fk_pos = fk(current_joints_rad)
                        print(f"📊 Calculated FK: ({fk_pos[0]:.4f}, {fk_pos[1]:.4f}, {fk_pos[2]:.4f})")
                    continue

                else:
                    # Try to parse as joint angles
                    try:
                        parts = cmd.replace(',', ' ').split()
                        if len(parts) != 6:
                            print("❌ Need exactly 6 joint angles (in degrees)")
                            continue
                        joints_deg = [float(p) for p in parts]
                    except ValueError:
                        print("❌ Invalid input. Enter 6 numbers or a command.")
                        continue

                # Convert to radians
                joints_rad = np.radians(joints_deg)

                # Check for clipping (warn user)
                clipped_rad = np.clip(joints_rad, -np.pi, np.pi)
                if not np.allclose(joints_rad, clipped_rad):
                    print(f"⚠️  WARNING: Input angles clipped to ±180° limits!")
                    print(f"   Input (deg): {joints_deg}")
                    print(f"   Clipped (deg): {np.degrees(clipped_rad)}")

                joints_rad = clipped_rad

                # Show EXPECTED FK vs CURRENT
                try:
                    target_fk = fk(joints_rad)
                    print(f"🎯 Expected Target FK: ({target_fk[0]:.4f}, {target_fk[1]:.4f}, {target_fk[2]:.4f})")
                except Exception as e:
                    print(f"⚠️ FK error: {e}")

                # Execute movement
                print(f"🚀 Moving to: {[f'{d:.1f}°' for d in joints_deg]}")
                next_state, reward, done, info = env.step(joints_rad)

                # Wait for settling (Longer wait for manual verification)
                print("⏳ Settling...")
                time.sleep(1.5)  # Let robot fully reach target
                for _ in range(30): # Spin to update TF/joint states
                    rclpy.spin_once(env, timeout_sec=0.1)

                # Re-read state AFTER settling (not mid-trajectory)
                settled_state = env.get_state()
                if settled_state is not None:
                     final_ee = settled_state[6:9]
                     dist_err = np.linalg.norm(final_ee - target_fk)
                     print(f"📍 Resulting EE:     ({final_ee[0]:.4f}, {final_ee[1]:.4f}, {final_ee[2]:.4f})")
                     print(f"📏 Error (FK vs TF): {dist_err*100:.2f} cm")
                     if dist_err > 0.02:
                         print("⚠️  Large discrepancy! Check physics/collisions/limits.")
                     else:
                         print("✅ FK matches TF2!")

                # Publish pen position if drawing enabled
                if drawing_enabled:
                    new_state = env.get_state()
                    if new_state is not None:
                        ee = new_state[6:9]
                        pen_msg = Point(x=float(ee[0]), y=float(ee[1]), z=float(ee[2]))
                        pen_pub.publish(pen_msg)
                        print(f"✏️  Drew at: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")

                print("✅ Movement complete!")

            except KeyboardInterrupt:
                print("\n👋 Interrupted. Exiting...")
                break

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "=" * 70)
        print("Manual control mode exited.")
        print("=" * 70)

        if env is not None:
            try:
                env.destroy_node()
            except:
                pass
        if ros_initialized:
            try:
                rclpy.shutdown()
            except:
                pass


def show_menu():
    """Display interactive training menu"""
    print("\n" + "="*70)
    print("🎮 TRAINING MENU")
    print("="*70)
    print("1. 🎮 Manual Test Mode (Verify environment)")
    print("2. 🤖 SAC Training (6-DOF Direct Control)")
    print("3. 🧠 SAC Training + Neural IK (3D Position Control)")
    print("4. 🧠 Train Neural IK Model")
    print("5. 🖋️ Drawing Task Training (SAC 6D Direct)")
    print("6. 🖋️ Drawing Task Training (SAC + Neural IK)")
    print("7. 🎛️ PID Tuning (RL-Optimized PID Gains)")
    print("8. 🚀 Digital Twin Realtime Mirror (Gazebo -> Pi realtime)")
    print("9. 🤖 Pi-Only Training (no Gazebo, direct Pi control)")
    print("="*70)

    choice = input("Select option (1-9): ").strip()
    return choice


def get_training_params():
    """Get training parameters interactively"""
    print("\n📊 Training Configuration")
    print("="*70)

    # Episodes
    episodes_input = input(f"Number of episodes (default {NUM_EPISODES}): ").strip()
    episodes = int(episodes_input) if episodes_input else NUM_EPISODES

    # Max steps
    steps_input = input(f"Max steps per episode (default {MAX_STEPS_PER_EPISODE}): ").strip()
    max_steps = int(steps_input) if steps_input else MAX_STEPS_PER_EPISODE

    print(f"\n✅ Configuration:")
    print(f"   Episodes: {episodes}")
    print(f"   Max steps: {max_steps}")
    print("="*70)

    return episodes, max_steps


def get_drawing_params():
    """Get drawing training parameters interactively"""
    print("\n🖋️ Drawing Training Configuration")
    print("="*70)

    # Import config values for display
    from drawing.drawing_config import SHAPE_TYPE, TOTAL_WAYPOINTS, POINTS_PER_EDGE

    print(f"  Shape: {SHAPE_TYPE} ({TOTAL_WAYPOINTS} waypoints, {POINTS_PER_EDGE} per edge)")
    print("  Each step = 1 attempt to reach current waypoint")
    print("  When waypoint reached → next waypoint becomes target")
    print("  Episode ends: all waypoints reached OR max steps exceeded")
    print("-"*70)
    print("  State: 18D = 6 joints + 3 EE + 3 target + 3 dist + 3 other")
    print("="*70)

    # Episodes (default higher for drawing)
    episodes_input = input("Number of episodes (default 100): ").strip()
    episodes = int(episodes_input) if episodes_input else 100

    # Max steps = ideally 1-2 per waypoint, but allow exploration buffer
    # 3 waypoints now, so allow min 5 steps, default 100
    steps_input = input("Max steps per episode (default 100, min 5): ").strip()
    max_steps = int(steps_input) if steps_input else 100
    max_steps = max(5, max_steps)  # Enforce minimum 5 steps

    print(f"\n✅ Drawing Configuration:")
    print(f"   Episodes: {episodes}")
    print(f"   Max steps: {max_steps} ({TOTAL_WAYPOINTS} waypoints, min 5 steps)")
    print(f"   State dim: 18")
    print("="*70)

    return episodes, max_steps


def _latest_matching_file(directory: str, pattern: str):
    import glob

    files = glob.glob(os.path.join(directory, pattern), recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def prompt_pid_backend() -> str:
    """Prompt for the PID tuning backend."""
    env_default = os.environ.get('VISUAL_SERVOING_CONTROL_BACKEND', 'sim')
    try:
        default_backend = resolve_control_backend(env_default)
    except Exception:
        default_backend = 'sim'

    print("\n🔧 PID Control Backend:")
    print("  a. sim")
    print("  b. sim_to_real_shadow")
    print("  c. real_replay")
    raw = input(f"Select (a/b/c, default={default_backend}): ").strip().lower()
    mapping = {
        'a': 'sim',
        'b': 'sim_to_real_shadow',
        'c': 'real_replay',
        'sim': 'sim',
        'sim_to_real_shadow': 'sim_to_real_shadow',
        'real_replay': 'real_replay',
        '': default_backend,
    }
    return resolve_control_backend(mapping.get(raw, raw))


def prompt_pid_replay_paths(mode: str):
    """Prompt for saved replay artifact + gains file paths."""
    pkl_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'pkl')
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoints')

    artifact_default = _latest_matching_file(pkl_dir, f'pid_best_artifact_*_{mode}_*.pkl')
    gains_default = _latest_matching_file(checkpoint_root, f'**/best_gains*.json')

    print("\n📦 Real Replay Inputs")
    if artifact_default:
        artifact_hint = os.path.basename(artifact_default)
    else:
        artifact_hint = 'required'
    artifact_path = input(f"Artifact path (Enter={artifact_hint}): ").strip()
    if not artifact_path and artifact_default:
        artifact_path = artifact_default

    if gains_default:
        gains_hint = os.path.basename(gains_default)
    else:
        gains_hint = 'optional'
    gains_path = input(f"Gains path (Enter={gains_hint}): ").strip()
    if not gains_path and gains_default:
        gains_path = gains_default

    return artifact_path, gains_path


def train_drawing(args):
    """
    Training loop for drawing task using DrawingEnvironment.
    """

    # Import config values
    from drawing.drawing_config import SHAPE_TYPE

    print("="*70)
    print(f"🖋️ Drawing Training - {SHAPE_TYPE.capitalize()} Trajectory")
    print("="*70)

    env = None
    ros_initialized = False

    try:
        # Initialize ROS2
        rclpy.init()
        ros_initialized = True

        # Import DrawingEnvironment (uses dense waypoints)
        from rl.drawing_environment import DrawingEnvironment
        from drawing.shape_generator import ShapeGenerator

        # Create drawing environment
        print("\n📦 Creating Drawing Environment...")

        # Import config values
        from drawing.drawing_config import (
            SHAPE_TYPE, SHAPE_SIZE, X_PLANE, WAYPOINT_TOLERANCE,
            POINTS_PER_EDGE, TOTAL_WAYPOINTS
        )

        env = DrawingEnvironment(
            max_episode_steps=args.max_steps,
            waypoint_tolerance=WAYPOINT_TOLERANCE,
            shape_type=SHAPE_TYPE,  # Uses points_per_edge from config
            shape_size=SHAPE_SIZE,
            x_plane=X_PLANE,
            use_dynamic_workspace=bool(getattr(args, 'require_board_detection', True)),
            control_backend=getattr(args, 'control_backend', None),
        )
        if not env.motion_backend.supports_reward_feedback:
            raise RuntimeError(
                "The selected backend does not provide authoritative reward feedback for SAC drawing training. "
                "Use control_backend=sim or sim_to_real_shadow."
            )

        # Wait for environment
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)

        # Wait for ArUco board detection
        if getattr(args, 'require_board_detection', True):
            print("\n⏳ Waiting for ArUco board detection...")
            if not env.wait_for_initial_detection(timeout_sec=10.0):
                print("⚠️  WARNING: No board detected! Shapes will use default position.")
                user_confirm = input("   Continue anyway? (y/n): ").strip().lower()
                if user_confirm != 'y':
                    print("❌ Training cancelled")
                    return
            else:
                print("✅ Board detected - shapes will be board-relative")
        else:
            print("\n📡 Board detection: optional (skipped)")

        print("✅ Drawing Environment ready!")
        print(f"   Shape: {SHAPE_TYPE} ({TOTAL_WAYPOINTS} waypoints, {POINTS_PER_EDGE} per edge)")
        print(f"   Size: {SHAPE_SIZE*100:.0f}cm | Tolerance: ±{WAYPOINT_TOLERANCE*100:.0f}cm")

        # Create SAC agent
        use_neural_ik = getattr(args, 'use_neural_ik', False)

        if use_neural_ik:
            # Load Neural IK
            nik_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'neural_ik.pth')
            if not os.path.exists(nik_path):
                print(f"\n❌ Neural IK model not found at: {nik_path}")
                print("   Please run option 6 first!")
                return
            neural_ik = NeuralIK()
            neural_ik.load(nik_path)
            args.neural_ik = neural_ik
            action_dim = 3
            max_action = np.array([1.0, 1.0, 1.0])
            min_action = np.array([-1.0, -1.0, -1.0])
            print("✅ Using Neural IK (3D Position Control)")
        else:
            args.neural_ik = None
            JOINT_LIMIT = np.pi / 2
            action_dim = 6
            max_action = np.array([JOINT_LIMIT] * 6)
            min_action = np.array([-JOINT_LIMIT] * 6)
            print("✅ Using 6D Direct Joint Control")

        # Extended state space for drawing (18D)
        agent = SACAgentGazebo(
            state_dim=18,  # 6 joints + 3 EE + 3 target + 3 dist + 3 other
            n_actions=action_dim,
            max_action=max_action,
            min_action=min_action,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            gamma=GAMMA,
            tau=TAU,
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            auto_entropy_tuning=True
        )

        # Set checkpoint directory
        mode_str = "neuralIK" if use_neural_ik else "direct"
        agent.checkpoint_dir = os.path.join(
            os.path.dirname(__file__), 'checkpoints', f'sac_drawing_{mode_str}'
        )
        os.makedirs(agent.checkpoint_dir, exist_ok=True)
        print(f"   Checkpoint dir: {agent.checkpoint_dir}")

        # ============================================================
        # LOAD REPLAY BUFFER (same structure as reaching options 2-5)
        # ============================================================
        mode_suffix = f"sac_drawing_{mode_str}"
        load_buffer = input("\n📦 Load existing replay buffer? (y/n): ").strip().lower()
        if load_buffer == 'y':
            # Find available buffers for THIS MODE - prioritize BEST over FINAL
            import glob
            pkl_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'pkl')
            os.makedirs(pkl_dir, exist_ok=True)

            best_buffers = sorted(glob.glob(f"{pkl_dir}/*best*{mode_suffix}*.pkl"), key=os.path.getmtime, reverse=True)
            final_buffers = sorted(glob.glob(f"{pkl_dir}/*final*{mode_suffix}*.pkl"), key=os.path.getmtime, reverse=True)

            # Best buffers first, then final buffers
            buffer_files = best_buffers + final_buffers

            if buffer_files:
                print(f"   Found {len(best_buffers)} best buffers, {len(final_buffers)} final buffers")

                # Show top options
                if best_buffers:
                    print(f"   [BEST]  {os.path.basename(best_buffers[0])}")
                if final_buffers:
                    print(f"   [FINAL] {os.path.basename(final_buffers[0])}")

                # Default to best buffer if available, else final
                default_buffer = best_buffers[0] if best_buffers else final_buffers[0]
                buffer_path = input(f"   Enter path (Enter = {os.path.basename(default_buffer)}): ").strip()
                if buffer_path == '':
                    buffer_path = default_buffer

                if buffer_path and os.path.exists(buffer_path):
                    try:
                        agent.replay_buffer.load(buffer_path)
                        print(f"   ✅ Loaded replay buffer from: {buffer_path}")
                        print(f"   Buffer size: {agent.replay_buffer.size()}")
                    except Exception as e:
                        print(f"   ❌ Failed to load buffer: {e}")
                elif buffer_path:
                    print(f"   ❌ Buffer file not found: {buffer_path}")
            else:
                print(f"   No buffer files found for {mode_suffix} in training_results/pkl/")

        # ============================================================
        # LOAD PRE-TRAINED MODELS (same structure as reaching options 2-5)
        # ============================================================
        checkpoint_dir = agent.checkpoint_dir

        # Try to load models: best first, then fallback to latest
        def _latest_file(directory, pattern):
            import glob
            files = glob.glob(os.path.join(directory, pattern))
            return max(files, key=os.path.getmtime) if files else None

        best_actor_path = os.path.join(checkpoint_dir, 'actor_sac_best.pth')
        latest_actor_path = _latest_file(checkpoint_dir, 'actor_*_best.pth')
        if latest_actor_path is None:
            latest_actor_path = _latest_file(checkpoint_dir, 'actor_*.pth')

        # Choose best if exists, otherwise latest
        actor_path = best_actor_path if os.path.exists(best_actor_path) else latest_actor_path

        if actor_path and os.path.exists(actor_path):
            try:
                agent.load_models(actor_path)
                print(f"\n✅ Loaded pre-trained models from: {checkpoint_dir}")
                print(f"   Actor: {os.path.basename(actor_path)}")
                # Show inferred critic paths
                critic1_path = actor_path.replace('actor_', 'critic1_')
                if os.path.exists(critic1_path):
                    print(f"   Critic1: {os.path.basename(critic1_path)}")
                    print(f"   Critic2: {os.path.basename(actor_path.replace('actor_', 'critic2_'))}")
            except Exception as e:
                print(f"\n⚠️ Failed to load models: {e}")
                print("   Starting with untrained agent")
        else:
            print(f"\n📝 No pre-trained models found in {checkpoint_dir}/")
            print("   Starting with untrained agent")

        # Pre-flight check: spawn the shape once so user can verify
        print("\n" + "="*70)
        print("👀 PRE-FLIGHT CHECK: Spawning shape in Gazebo...")
        env.reset_environment()
        for _ in range(20):
            rclpy.spin_once(env, timeout_sec=0.1)

        input("   Please verify the shape is correctly spawned in Gazebo. Press ENTER to start training...")
        print("="*70)

        def finalize_drawing_training(interrupted=False):
            if len(episode_rewards) == 0:
                print("   No episodes completed, skipping saving/plotting.")
                return

            # Close step log file if not closed
            if not step_log_file.closed:
                step_log_file.close()
                print(f"📝 Step log saved: {step_log_path}")

            from drawing.drawing_config import TOTAL_WAYPOINTS
            total_wp = TOTAL_WAYPOINTS

            print("\n" + "="*70)
            if interrupted:
                print("⚠️  Drawing training interrupted by user!")
            else:
                print("🎉 Drawing training complete!")
            print(f"   Best waypoints: {max(waypoints_completed)}/{total_wp}")
            print("="*70)

            agent.save_models()

            # Save replay buffer for future training (same location as reaching)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pkl_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'pkl')
            os.makedirs(pkl_dir, exist_ok=True)

            # Save both "best" and "final" buffers
            buffer_base = f"replay_buffer_best_{mode_suffix}_{timestamp}.pkl"
            buffer_path = os.path.join(pkl_dir, buffer_base)
            try:
                agent.replay_buffer.save(buffer_path)
                print(f"💾 Saved replay buffer: {buffer_path}")
                print(f"   Buffer size: {agent.replay_buffer.size()} transitions")
            except Exception as e:
                print(f"⚠️ Failed to save buffer: {e}")

            # Plot training statistics
            plot_suffix = f"sac_drawing_{mode_str}"
            plot_drawing_stats(
                episode_rewards=episode_rewards,
                waypoints_reached=waypoints_completed,
                shape_completions=shape_completions,
                actor_losses=actor_losses,
                critic_losses=critic_losses,
                episode_trajectories=episode_trajectories,
                target_waypoints=target_waypoints,
                mode_suffix=plot_suffix
            )

            # Final cleanup - mode-specific, keep only best and final buffers
            cleanup_old_files(pkl_dir, f"replay_buffer_ep*{mode_suffix}*.pkl", 4)  # Keep 4 periodic
            cleanup_old_files(pkl_dir, f"replay_buffer_best*{mode_suffix}*.pkl", 1)  # Keep only 1 best
            cleanup_old_files(pkl_dir, f"replay_buffer_final*{mode_suffix}*.pkl", 1)  # Keep only 1 final

            # Clean up png and csv and step_logs in training_results/
            png_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'png')
            csv_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'csv')
            step_log_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'step_logs')
            cleanup_old_files(png_dir, f"drawing_training_{plot_suffix}_*.png", 3)
            cleanup_old_files(csv_dir, f"drawing_training_{plot_suffix}_*.csv", 3)
            cleanup_old_files(step_log_dir, "step_log_*.jsonl", 3)
            print(f"🧹 Cleaned up old {mode_suffix} files")

        # Training loop
        print(f"\n🚀 Starting drawing training ({args.episodes} episodes)...\n")

        # Create step log file for detailed step-by-step logging
        import json
        from datetime import datetime
        step_log_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'step_logs')
        os.makedirs(step_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_log_path = os.path.join(step_log_dir, f'step_log_{timestamp}.jsonl')
        step_log_file = open(step_log_path, 'w')
        print(f"📝 Step log: {step_log_path}")

        # Data tracking for plotting
        episode_rewards = []
        waypoints_completed = []
        shape_completions = []
        episode_trajectories = []
        actor_losses = []
        critic_losses = []

        # Get target waypoints for plotting
        target_waypoints = env.waypoints if hasattr(env, 'waypoints') else None

        for episode in range(args.episodes):
            state = env.reset_environment()

            for _ in range(10):
                rclpy.spin_once(env, timeout_sec=0.1)

            if state is None:
                print(f"Episode {episode+1}: Failed to reset")
                continue

            episode_reward = 0.0
            min_distance = float('inf')
            episode_trajectory = []  # Track EE positions this episode

            for step in range(args.max_steps):
                # Get state info before action (18D state layout)
                # [0-5] joints, [6-8] EE, [9-11] target, [12-14] dist, [15] dist3d, [16] progress, [17] remaining
                ee_pos_before = state[6:9] if len(state) >= 9 else None
                target_pos = state[9:12] if len(state) >= 12 else None
                wp_reached_before = 0  # Will get from info after step

                action = agent.select_action(state, evaluate=False)

                print(f"\n  ═══ Step {step+1}/{args.max_steps} ═══")
                if ee_pos_before is not None and target_pos is not None:
                    dist_before = np.linalg.norm(ee_pos_before - target_pos)
                    print(f"  📍 EE:     [{ee_pos_before[0]:.4f}, {ee_pos_before[1]:.4f}, {ee_pos_before[2]:.4f}]")
                    print(f"  🎯 Target: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
                    print(f"  📏 Distance: {dist_before*100:.2f}cm")

                # Convert action if using Neural IK
                if args.neural_ik is not None:
                    # FIXED: Treat waypoint as single target (like Options 4-5)
                    # Agent outputs delta direction, we move toward waypoint

                    # Get current waypoint target from state
                    waypoint = target_pos  # state[9:12] = current waypoint

                    # Action is delta scaling (how much to move toward waypoint)
                    # Action = 1 means full step, 0 = no move, -1 = away
                    STEP_SIZE = 0.15  # 15cm max step (matches triangle edge spacing)

                    # Compute direction to waypoint
                    direction = waypoint - ee_pos_before
                    distance = np.linalg.norm(direction)

                    if distance > 0.001:  # Avoid division by zero
                        direction_norm = direction / distance
                        # Action scales how much we move in that direction
                        # action[0] = forward/back, action[1-2] = fine adjustment
                        move_amount = (action[0] + 1) / 2 * STEP_SIZE  # 0 to 15cm
                        fine_adjust = action[1:3] * 0.02  # ±2cm lateral

                        # Target = EE + movement toward waypoint + fine adjustment
                        delta = direction_norm * move_amount
                        target_xyz = ee_pos_before + delta
                        target_xyz[0] += fine_adjust[0]  # X adjustment
                        target_xyz[2] += fine_adjust[1]  # Z adjustment
                    else:
                        target_xyz = waypoint  # Already at waypoint

                    # Clamp to safe bounds (base_link frame)
                    target_xyz = np.clip(target_xyz,
                                         [-0.55, -0.15, 0.10],
                                         [-0.20,  0.15, 0.65])

                    # Neural IK converts target position to joints
                    joints_action = args.neural_ik.predict(target_xyz)
                    print(f"  🎯 Waypoint: [{waypoint[0]:.3f}, {waypoint[1]:.3f}, {waypoint[2]:.3f}]")
                    print(f"  🧠 IK Target: [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")
                    next_state, reward, done, info = env.step(joints_action)
                else:
                    next_state, reward, done, info = env.step(action)

                for _ in range(5):
                    rclpy.spin_once(env, timeout_sec=0.1)

                if next_state is None:
                    print("  ❌ State unavailable")
                    break

                # Log after action (18D state: EE at [6:9])
                ee_pos_after = next_state[6:9]
                dist_after = info.get('distance', 0)
                wp_idx = info.get('waypoint_index', 0)
                wp_total = info.get('total_waypoints', 30)
                wp_reached = info.get('waypoints_reached', 0)

                print(f"  📍 AFTER: [{ee_pos_after[0]:.4f}, {ee_pos_after[1]:.4f}, {ee_pos_after[2]:.4f}]")
                print(f"  📏 Dist: {dist_after*100:.2f}cm | WP: {wp_idx}/{wp_total} | Reached: {wp_reached}")
                print(f"  💰 Reward: {reward:.3f}")

                # Log step data to file
                step_data = {
                    'episode': episode + 1,
                    'step': step + 1,
                    'joints': state[0:6].tolist() if len(state) >= 6 else [],
                    'ee_before': ee_pos_before.tolist() if ee_pos_before is not None else [],
                    'ee_after': ee_pos_after.tolist(),
                    'target': target_pos.tolist() if target_pos is not None else [],
                    'action': action.tolist() if hasattr(action, 'tolist') else list(action),
                    'dist_before_cm': float(dist_before * 100) if 'dist_before' in dir() else 0,
                    'dist_after_cm': float(dist_after * 100),
                    'waypoint_idx': wp_idx,
                    'waypoint_total': wp_total,
                    'waypoints_reached': wp_reached,
                    'reward': float(reward),
                    'done': done,
                    'shape_complete': info.get('shape_complete', False)
                }
                step_log_file.write(json.dumps(step_data) + '\n')
                step_log_file.flush()  # Ensure data is written immediately

                min_distance = min(min_distance, dist_after)

                if wp_reached > wp_reached_before:
                    print(f"  ✅ WAYPOINT {wp_idx} REACHED!")
                    wp_reached_before = wp_reached

                if info.get('shape_complete', False):
                    print(f"  🎨🎨🎨 SHAPE COMPLETE! 🎨🎨🎨")

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Track trajectory for plotting
                episode_trajectory.append(ee_pos_after.copy())

                episode_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(episode_reward)
            wp_reached = info.get('waypoints_reached', 0)
            waypoints_completed.append(wp_reached)
            shape_complete = info.get('shape_complete', False)
            shape_completions.append(shape_complete)
            episode_trajectories.append(episode_trajectory)

            # Train agent and track losses
            ep_actor_loss = None
            ep_critic_loss = None
            if episode >= 5:
                for _ in range(20):
                    losses = agent.train()
                    if losses and len(losses) >= 2:
                        ep_actor_loss = losses[0]
                        ep_critic_loss = losses[1]
            actor_losses.append(ep_actor_loss)
            critic_losses.append(ep_critic_loss)

            # Log
            shape_complete = info.get('shape_complete', False)
            status = "🎨 COMPLETE!" if shape_complete else f"WP: {wp_reached}/{wp_total}"
            print(f"Episode {episode+1}/{args.episodes} | "
                  f"Reward: {episode_reward:.1f} | {status}")

            # Save best
            if shape_complete or (episode > 10 and wp_reached >= max(waypoints_completed)):
                agent.save_models()

        finalize_drawing_training(interrupted=False)
        print(f"\n✅ Drawing training complete! Trained for {len(episode_rewards)} episodes.")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        finalize_drawing_training(interrupted=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.destroy_node()
            except:
                pass
        if ros_initialized:
            try:
                rclpy.shutdown()
            except:
                pass

def _run_pid_real_replay(mode='reaching', replay_artifact_path=None, replay_gains_path=None):
    """Replay a saved PID-tuning artifact on the real robot backend with detailed logging and plotting."""
    import json
    import pickle
    import matplotlib.pyplot as plt
    from datetime import datetime

    base_env = None
    ros_initialized = False

    if not replay_artifact_path:
        raise ValueError("real_replay requires a saved PID replay artifact path")
    if not os.path.exists(replay_artifact_path):
        raise FileNotFoundError(f"Replay artifact not found: {replay_artifact_path}")

    try:
        rclpy.init()
        ros_initialized = True

        print("\n📦 Creating real replay environment...")
        base_env = RLEnvironment(
            max_episode_steps=1,
            goal_tolerance=0.01,
            control_backend='real_replay',
        )

        print("   Waiting for hardware state...")
        time.sleep(1.0)
        for _ in range(15):
            rclpy.spin_once(base_env, timeout_sec=0.1)

        with open(replay_artifact_path, 'rb') as f:
            artifact = pickle.load(f)

        artifact_mode = artifact.get('mode', mode)
        if artifact_mode != mode:
            print(f"⚠️  Artifact mode is '{artifact_mode}', overriding requested '{mode}'")
            mode = artifact_mode

        # Retrieve replay trajectory from artifact. Prefer the smoothed nominal replay
        # path if present; fall back to the raw commanded trace for older artifacts.
        commanded_trajectory_list = artifact.get('replay_trajectory_rad', [])
        if not commanded_trajectory_list:
            commanded_trajectory_list = artifact.get('commanded_trajectory_rad', [])
        if not commanded_trajectory_list:
            # Fall back to checking if it's in the replay_plan
            replay_plan = artifact.get('replay_plan', {})
            segments = replay_plan.get('segments', [])
            if not segments:
                print("❌ Artifact has no commanded trajectory or replay plan!")
                return False
            commanded_trajectory = []
            for seg in segments:
                pos_deg_dict = {name: pos for name, pos in zip(seg['joint_names_pi'], seg['positions_deg'])}
                pos_rad = np.zeros(len(base_env.motion_backend.mapper.gazebo_joint_names))
                for gz_idx, gz_name in enumerate(base_env.motion_backend.mapper.gazebo_joint_names):
                    _, pi_name, home_deg, inverted = base_env.motion_backend.mapper.gazebo_lookup[gz_name]
                    if pi_name in pos_deg_dict:
                        pos_rad[gz_idx] = base_env.motion_backend.mapper.pi_deg_to_gazebo_rad(
                            pos_deg_dict[pi_name], home_deg, inverted
                        )
                commanded_trajectory.append(pos_rad)
        else:
            commanded_trajectory = [np.array(cmd) for cmd in commanded_trajectory_list]

        dt = float(artifact.get('trajectory_dt_sec', 0.02))

        # Prompt for parameters
        episodes_input = input("Number of episodes to run (default 5): ").strip()
        episodes = int(episodes_input) if episodes_input else 5

        rate_input = input("Replay rate Hz (default 5.0, lower=safer): ").strip()
        replay_rate = float(rate_input) if rate_input else 5.0

        # Generate replay plan
        new_replay_plan = base_env.motion_backend.mapper.export_pi_replay_plan(
            joint_samples_rad=commanded_trajectory,
            sample_dt=dt,
            joint_limits_low=base_env.gazebo_limits_low,
            joint_limits_high=base_env.gazebo_limits_high,
            replay_rate_hz=replay_rate,
        )

        gains = None
        if replay_gains_path and os.path.exists(replay_gains_path):
            with open(replay_gains_path, 'r') as f:
                gains = json.load(f)

        print("\n▶️ Multi-Episode Deploy to Pi Started")
        print("=" * 70)
        print(f"   Artifact: {replay_artifact_path}")
        print(f"   Mode: {mode}")
        print(f"   Replay rate: {replay_rate} Hz")
        print(f"   Episodes: {episodes}")
        if gains is not None:
            print(f"   Gains file: {replay_gains_path}")
            print(f"   Kp: {np.round(gains.get('Kp', []), 3)}")
            print(f"   Ki: {np.round(gains.get('Ki', []), 3)}")
            print(f"   Kd: {np.round(gains.get('Kd', []), 3)}")
        print("=" * 70)

        # Setup Logging
        log_dir = os.path.join(os.getcwd(), 'training_results', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f"deploy_replay_log_{timestamp}.txt")

        with open(log_path, 'w') as f:
            f.write(f"=== DEPLOY MULTI-EPISODE REPLAY LOG ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Artifact: {replay_artifact_path}\n")
            f.write(f"Replay Rate: {replay_rate} Hz\n")
            f.write(f"Total Episodes: {episodes}\n")
            f.write(f"Total Segments Per Episode: {len(new_replay_plan['segments'])}\n")
            f.write("--------------------------------------------------------------------------------\n")

        episode_rewards = []
        episode_cartesian_mm = []
        episode_avg_wp_mm = []
        episode_max_wp_mm = []
        episode_joint_errors = []
        episode_durations = []
        episode_waypoint_errors = []

        # Accumulate logs for plotting
        commanded_deg_log_all = []
        actual_deg_log_all = [] # list of arrays, one per episode
        time_log = []
        deploy_commanded_joint_traces = []
        deploy_actual_joint_traces = []
        deploy_joint_trace_times = []
        deploy_actual_paths_xyz = []

        target_meta = artifact.get('target_metadata', {})
        target_shape_xyz = None
        if mode == 'drawing':
            target_shape_list = target_meta.get('shape_xyz_waypoints', [])
            if target_shape_list:
                target_shape_xyz = np.asarray(target_shape_list, dtype=np.float64)

        total_segments_sent = 0
        total_segments_acked = 0
        total_segments_in_tolerance = 0
        total_segments_lagging = 0
        deploy_joint_error_tolerance_deg = 2.0
        deploy_max_joint_errors = []
        pi_joint_names = list(base_env.motion_backend.mapper.pi_joint_names)
        pi_joint_meta = {}
        for gz_name, (_, pi_name, home_deg, inverted) in base_env.motion_backend.mapper.gazebo_lookup.items():
            pi_joint_meta[pi_name] = (home_deg, inverted)

        for ep in range(episodes):
            print(f"\n🎬 Starting Episode {ep+1}/{episodes}...")
            # Move to start position first
            start_joint = commanded_trajectory[0]
            print(f"🏠 Homing robot and moving to start position (duration=2.0s)...")
            base_env.motion_backend.home(duration=2.0)
            time.sleep(1.0)
            base_env.motion_backend.move_to_joint_positions(start_joint, duration=2.0)
            time.sleep(2.0)
            for _ in range(15):
                rclpy.spin_once(base_env, timeout_sec=0.1)

            # Get initial actual position
            q_start_actual = np.array(base_env.joint_positions)
            print(f"Start actual joints (deg): {np.degrees(q_start_actual)}")

            # Log commands and actuals for this episode
            commanded_deg_log = []
            actual_deg_log = []
            commanded_joint_rad_log = []
            actual_joint_rad_log = []
            curr_time = 0.0
            episode_time_log = []

            ep_start_time = time.time()

            print(f"▶️ Replaying {len(new_replay_plan['segments'])} segments for Episode {ep+1}...")
            for idx, seg in enumerate(new_replay_plan['segments']):
                # Prepare degrees dict
                positions_deg = {
                    name: float(pos)
                    for name, pos in zip(seg['joint_names_pi'], seg['positions_deg'])
                }
                # Log commanded degrees
                cmd_deg_arr = [positions_deg[name] for name in pi_joint_names]
                commanded_deg_log.append(cmd_deg_arr)
                commanded_joint_rad_log.append(np.array([
                    base_env.motion_backend.mapper.pi_deg_to_gazebo_rad(
                        positions_deg[name],
                        pi_joint_meta[name][0],
                        pi_joint_meta[name][1],
                    )
                    for name in pi_joint_names
                ], dtype=np.float64))

                # Build and publish message
                traj_msg = base_env.motion_backend.mapper.build_pi_trajectory_msg(positions_deg, float(seg['duration_sec']))
                traj_msg.header.stamp = base_env.get_clock().now().to_msg()
                base_env.motion_backend.real_joint_trajectory_pub.publish(traj_msg)
                total_segments_sent += 1

                # Reset tracking variable
                base_env.data_ready = False

                # Sleep for segment duration
                time.sleep(float(seg['duration_sec']) + 0.02)
                for _ in range(10):
                    rclpy.spin_once(base_env, timeout_sec=0.01)

                # Read actual joint state of physical arm and map to degrees
                actual_joint_rad_gazebo = np.array(base_env.joint_positions, dtype=np.float64)
                actual_deg_dict = base_env.motion_backend.mapper.gazebo_positions_to_pi_deg(np.array(base_env.joint_positions))
                actual_deg_arr = [actual_deg_dict[name] for name in pi_joint_names]
                actual_deg_log.append(actual_deg_arr)
                actual_joint_rad_log.append(actual_joint_rad_gazebo.copy())
                joint_error_dict = {
                    name: abs(float(positions_deg[name]) - float(actual_deg_dict[name]))
                    for name in pi_joint_names
                }
                max_joint_error = max(joint_error_dict.values()) if joint_error_dict else 0.0
                mean_segment_joint_error = float(np.mean(list(joint_error_dict.values()))) if joint_error_dict else 0.0
                deploy_max_joint_errors.append(max_joint_error)

                if base_env.data_ready:
                    total_segments_acked += 1
                    if max_joint_error <= deploy_joint_error_tolerance_deg:
                        total_segments_in_tolerance += 1
                        packet_status = "OK"
                    else:
                        total_segments_lagging += 1
                        packet_status = "LAG"
                else:
                    packet_status = "LOST"

                curr_time += float(seg['duration_sec'])
                episode_time_log.append(curr_time)

                cmd_str = ", ".join(f"{k}={positions_deg[k]:.1f}°" for k in base_env.motion_backend.mapper.pi_joint_names)
                actual_str = ", ".join(f"{k}={actual_deg_dict[k]:.1f}°" for k in base_env.motion_backend.mapper.pi_joint_names)
                err_str = ", ".join(f"{k}={joint_error_dict[k]:.1f}°" for k in base_env.motion_backend.mapper.pi_joint_names)

                log_line = (
                    f"[Ep {ep+1}/{episodes} | SEG {idx+1}/{len(new_replay_plan['segments'])}] "
                    f"Cmd: [{cmd_str}] | "
                    f"Actual: [{actual_str}] | "
                    f"Err: [{err_str}] | "
                    f"MaxErr: {max_joint_error:.2f}° | "
                    f"MeanErr: {mean_segment_joint_error:.2f}° | "
                    f"Status: {packet_status} | "
                    f"dur={seg['duration_sec']:.2f}s\n"
                )
                print(log_line.strip())
                with open(log_path, 'a') as f:
                    f.write(log_line)

            # Convert to numpy arrays
            commanded_deg_log = np.array(commanded_deg_log)
            actual_deg_log = np.array(actual_deg_log)
            error_log = commanded_deg_log - actual_deg_log

            # Save for final plot
            if not time_log:
                time_log = episode_time_log
                commanded_deg_log_all = commanded_deg_log
            actual_deg_log_all.append(actual_deg_log)
            deploy_commanded_joint_traces.append([q.tolist() for q in commanded_joint_rad_log])
            deploy_actual_joint_traces.append([
                np.array([
                    base_env.motion_backend.mapper.pi_deg_to_gazebo_rad(
                        actual_deg_row[name],
                        pi_joint_meta[name][0],
                        pi_joint_meta[name][1],
                    )
                    for name in pi_joint_names
                ], dtype=np.float64).tolist()
                for actual_deg_row in [
                    {name: row[idx] for idx, name in enumerate(pi_joint_names)}
                    for row in actual_deg_log
                ]
            ])
            deploy_joint_trace_times.append(list(episode_time_log))

            # Compute metrics for this episode
            from rl.fk_ik_utils import fk
            q_final_actual = np.array(base_env.joint_positions)
            pi_xyz_final = np.array(fk(q_final_actual.tolist(), raw=True))

            if mode == 'drawing':
                target_xyz = np.array(target_meta.get('shape_xyz_waypoints', []))
                target_xyz_final = np.array(target_xyz[-1]) if len(target_xyz) > 0 else np.zeros(3)
            else:
                target_xyz_final = np.array(target_meta.get('target_xyz', np.zeros(3)))

            pi_cartesian_miss_mm = float(np.linalg.norm(pi_xyz_final - target_xyz_final) * 1000.0)
            joint_diff = float(np.mean(np.abs(error_log)))
            ep_duration = time.time() - ep_start_time

            episode_cartesian_mm.append(pi_cartesian_miss_mm)
            episode_joint_errors.append(joint_diff)
            episode_durations.append(ep_duration)

            if mode == 'drawing' and actual_joint_rad_log and target_shape_xyz is not None:
                actual_path_xyz = np.array(
                    [fk(np.asarray(q, dtype=np.float64).tolist(), raw=True) for q in actual_joint_rad_log],
                    dtype=np.float64,
                )
                deploy_actual_paths_xyz.append(actual_path_xyz)
                avg_wp_mm, max_wp_mm, waypoint_errors_mm = _compute_resampled_waypoint_errors(
                    actual_path_xyz,
                    target_shape_xyz,
                )
                episode_avg_wp_mm.append(avg_wp_mm)
                episode_max_wp_mm.append(max_wp_mm)
                episode_waypoint_errors.append(waypoint_errors_mm)
            else:
                episode_avg_wp_mm.append(None)
                episode_max_wp_mm.append(None)
                episode_waypoint_errors.append(None)

            # Print option 7 style progress line
            if mode == 'drawing' and episode_avg_wp_mm[-1] is not None:
                print(f"\nEp {ep+1:4d}/{episodes} | "
                      f"Duration: {ep_duration:.1f}s | "
                      f"EndMiss: {pi_cartesian_miss_mm:5.1f}mm | "
                      f"AvgWp: {episode_avg_wp_mm[-1]:5.1f}mm "
                      f"MaxWp: {episode_max_wp_mm[-1]:5.1f}mm | "
                      f"MeanJointErr: {joint_diff:.2f}° | "
                      f"Hz: {replay_rate:.1f}")
            else:
                print(f"\nEp {ep+1:4d}/{episodes} | "
                      f"Duration: {ep_duration:.1f}s | "
                      f"CartesianMiss: {pi_cartesian_miss_mm:5.1f}mm | "
                      f"MeanJointErr: {joint_diff:.2f}° | "
                      f"Hz: {replay_rate:.1f}")

            # Log episode summary
            if mode == 'drawing' and episode_avg_wp_mm[-1] is not None:
                ep_summary = (
                    f"\n--- Episode {ep+1} Summary ---\n"
                    f"Duration: {ep_duration:.2f}s | End Miss: {pi_cartesian_miss_mm:.2f} mm | "
                    f"Avg WP Miss: {episode_avg_wp_mm[-1]:.2f} mm | "
                    f"Max WP Miss: {episode_max_wp_mm[-1]:.2f} mm | "
                    f"Mean Joint Error: {joint_diff:.2f}°\n"
                    f"--------------------------------------------------------------------------------\n"
                )
            else:
                ep_summary = (
                    f"\n--- Episode {ep+1} Summary ---\n"
                    f"Duration: {ep_duration:.2f}s | Cartesian Miss: {pi_cartesian_miss_mm:.2f} mm | "
                    f"Mean Joint Error: {joint_diff:.2f}°\n"
                    f"--------------------------------------------------------------------------------\n"
                )
            with open(log_path, 'a') as f:
                f.write(ep_summary)

        # Final Session Stats
        loss_pct = ((total_segments_sent - total_segments_acked) / total_segments_sent) * 100.0 if total_segments_sent > 0 else 0.0
        lag_pct = (total_segments_lagging / total_segments_sent) * 100.0 if total_segments_sent > 0 else 0.0
        tolerance_pct = (total_segments_in_tolerance / total_segments_sent) * 100.0 if total_segments_sent > 0 else 0.0
        mean_cartesian = np.mean(episode_cartesian_mm)
        std_cartesian = np.std(episode_cartesian_mm)
        mean_joint_err = np.mean(episode_joint_errors)
        mean_duration = np.mean(episode_durations)
        valid_avg_wp = [v for v in episode_avg_wp_mm if v is not None]
        valid_max_wp = [v for v in episode_max_wp_mm if v is not None]

        final_summary_lines = [
            "",
            "==================================================",
            "📊 DEPLOY SESSION COMPLETE SUMMARY",
            "==================================================",
            f"Total Episodes Run      : {episodes}",
            f"Avg Cartesian Error     : {mean_cartesian:.2f} ± {std_cartesian:.2f} mm",
            f"Avg Joint Position Error: {mean_joint_err:.2f}°",
            f"Avg Episode Duration    : {mean_duration:.2f} s",
        ]
        if mode == 'drawing' and valid_avg_wp:
            final_summary_lines.append(
                f"Avg Waypoint Miss       : {np.mean(valid_avg_wp):.2f} ± {np.std(valid_avg_wp):.2f} mm"
            )
            final_summary_lines.append(
                f"Avg Max Waypoint Miss   : {np.mean(valid_max_wp):.2f} mm"
            )
        final_summary_lines += [
            f"Telemetry Miss Rate     : {loss_pct:.1f}% ({total_segments_sent - total_segments_acked}/{total_segments_sent} segments)",
            f"In-Tolerance Rate       : {tolerance_pct:.1f}% ({total_segments_in_tolerance}/{total_segments_sent} segments, <= {deploy_joint_error_tolerance_deg:.1f}°)",
            f"Lag/Error Rate          : {lag_pct:.1f}% ({total_segments_lagging}/{total_segments_sent} segments, > {deploy_joint_error_tolerance_deg:.1f}°)",
            f"Log saved to            : {log_path}",
            "==================================================",
            "",
        ]
        final_summary = "\n".join(final_summary_lines)
        print(final_summary)
        with open(log_path, 'a') as f:
            f.write(final_summary)

        # Save PKL results
        results_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'pkl')
        png_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'png')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        deploy_results_path = os.path.join(results_dir, f'deploy_results_{mode}_{timestamp}.pkl')

        deploy_results = {
            'source_artifact': replay_artifact_path,
            'source_mode': mode,
            'pi_replay_rate_hz': replay_rate,
            'total_episodes': episodes,
            'avg_cartesian_error_mm': mean_cartesian,
            'std_cartesian_error_mm': std_cartesian,
            'avg_joint_error_deg': mean_joint_err,
            'packet_loss_pct': loss_pct,
            'joint_error_tolerance_deg': deploy_joint_error_tolerance_deg,
            'in_tolerance_rate_pct': tolerance_pct,
            'lag_error_rate_pct': lag_pct,
            'segments_in_tolerance': total_segments_in_tolerance,
            'segments_lagging': total_segments_lagging,
            'deploy_max_joint_errors_deg': deploy_max_joint_errors,
            'deploy_timestamp': timestamp,
            'episode_cartesian_errors': episode_cartesian_mm,
            'episode_avg_wp_mm': episode_avg_wp_mm,
            'episode_max_wp_mm': episode_max_wp_mm,
            'episode_joint_errors': episode_joint_errors,
            'commanded_deg': commanded_deg_log_all.tolist(),
            'actual_deg_all_episodes': [act.tolist() for act in actual_deg_log_all],
            'time_log': time_log,
            'deploy_commanded_joint_traces': deploy_commanded_joint_traces,
            'deploy_actual_joint_traces': deploy_actual_joint_traces,
            'deploy_joint_trace_times': deploy_joint_trace_times,
            'deploy_actual_paths_xyz': [path.tolist() for path in deploy_actual_paths_xyz],
            'target_shape_xyz': target_shape_xyz.tolist() if target_shape_xyz is not None else None,
            'episode_waypoint_errors': episode_waypoint_errors,
        }
        with open(deploy_results_path, 'wb') as f:
            pickle.dump(deploy_results, f)
        print(f"💾 Deploy results saved to: {deploy_results_path}")

        if mode == 'drawing' and deploy_actual_paths_xyz and target_shape_xyz is not None:
            valid_wp_indices = [idx for idx, value in enumerate(episode_avg_wp_mm) if value is not None]
            if valid_wp_indices:
                representative_idx = min(valid_wp_indices, key=lambda idx: episode_avg_wp_mm[idx])
            else:
                representative_idx = 0
            _plot_drawing_trajectory(
                all_paths=deploy_actual_paths_xyz,
                best_path=deploy_actual_paths_xyz[representative_idx],
                target_shape=target_shape_xyz,
                png_dir=png_dir,
                timestamp=timestamp,
                plot_stem='deploy_trajectory',
                title_prefix='Deploy Replay (Drawing): Trajectory Quality',
            )
            _plot_pid_joint_tracking(
                commanded_traces=[deploy_commanded_joint_traces[idx] for idx in range(len(deploy_actual_joint_traces))],
                actual_traces=deploy_actual_joint_traces,
                time_traces=deploy_joint_trace_times,
                joint_names=pi_joint_names,
                png_dir=png_dir,
                timestamp=timestamp,
                representative_idx=representative_idx,
                avg_wp_mm=episode_avg_wp_mm,
                max_wp_mm=episode_max_wp_mm,
                plot_stem='deploy_joint_tracking',
                title_prefix='Deploy Joint Tracking',
            )
            _plot_deploy_drawing_summary(
                endpoint_mm=episode_cartesian_mm,
                avg_wp_mm=episode_avg_wp_mm,
                max_wp_mm=episode_max_wp_mm,
                joint_errors=episode_joint_errors,
                durations=episode_durations,
                waypoint_errors=episode_waypoint_errors,
                png_dir=png_dir,
                timestamp=timestamp,
                replay_rate_hz=replay_rate,
            )

        # Save Combined Plot
        plot_path = os.path.join(png_dir, f'deploy_comparison_{mode}_{timestamp}.png')

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Deploy Multi-Episode: {mode.capitalize()} (Pi vs. Sim at {replay_rate}Hz)\n"
                     f"Avg Cartesian Miss: {mean_cartesian:.1f} ± {std_cartesian:.1f}mm | Joint Err: {mean_joint_err:.2f}° avg\n"
                     f"Telemetry Miss: {loss_pct:.1f}% | Total Runs: {episodes}",
                     fontsize=14, fontweight='bold')

        joint_names = base_env.motion_backend.mapper.pi_joint_names
        for idx, (ax, joint_name) in enumerate(zip(axes.ravel(), joint_names)):
            ax.plot(time_log, commanded_deg_log_all[:, idx], 'r--', linewidth=2, label='Commanded (Sim)')
            for ep_idx, actual_deg_log in enumerate(actual_deg_log_all):
                ax.plot(time_log, actual_deg_log[:, idx], alpha=0.5, label=f'Actual Ep {ep_idx+1}')
            ax.set_title(f"Joint: {joint_name}")
            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Angle (deg)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🖼️ Comparison plot saved: {plot_path}")

        # Clean up old files
        cleanup_old_files(os.path.join(results_dir, 'pkl'), f"deploy_results_{mode}_*.pkl", 3)
        cleanup_old_files(os.path.join(results_dir, 'png'), f"deploy_comparison_{mode}_*.png", 3)
        cleanup_old_files(os.path.join(results_dir, 'logs'), "deploy_replay_log_*.txt", 3)

        return True

    finally:
        if base_env is not None:
            try:
                print("\n🏠 Returning robot to home position before exit...")
                base_env.home(duration=2.0)
            except Exception as e:
                print(f"   ⚠️ Could not return home: {e}")
            try:
                base_env.destroy_node()
            except Exception:
                pass
        if ros_initialized:
            try:
                rclpy.shutdown()
            except Exception:
                pass


def train_pid_tuning(
    mode='reaching',
    control_backend=None,
    require_board_detection=False,
    replay_artifact_path=None,
    replay_gains_path=None,
    segment_steps=20,
):
    """
    Train RL agent to optimize PID gains for trajectory tracking.

    Self-contained training function — does NOT call train() or modify
    any existing training infrastructure. Uses its own SAC agent with
    24D state and 18D action dimensions.

    Targets are generated in joint-space (random valid configurations).
    FK (exact URDF math) is used to compute XYZ for visualization only.
    No Neural IK dependency.

    Args:
        mode: 'reaching' or 'drawing' (both use joint-space for now)
    """
    control_backend = resolve_control_backend(control_backend)

    if control_backend == 'real_replay':
        return _run_pid_real_replay(
            mode=mode,
            replay_artifact_path=replay_artifact_path,
            replay_gains_path=replay_gains_path,
        )

    print("\n" + "="*70)
    print(f"🎛️  PID TUNING — RL-Optimized PID Gains ({mode.upper()})")
    print("="*70)
    print(f"Backend: {control_backend}")
    print("Architecture: SAC → PID gains (18D) → position commands → Gazebo")
    print("Episode: observe state → set gains → track trajectory → reward")
    print("Targets: random joint-space → FK for sphere visualization")
    print("="*70)

    # Lazy imports (only loaded for option 7)
    from rl.pid_tuning_env import PIDTuningEnv
    from controllers.pid_joint_controller import PIDJointController

    env = None
    ros_initialized = False

    try:
        # Initialize ROS2
        rclpy.init()
        ros_initialized = True

        # Create base RL environment (handles ROS2 communication)
        print(f"\n📦 Creating base RL environment for {mode}...")
        if mode == 'drawing':
            from rl.drawing_environment import DrawingEnvironment
            from drawing.drawing_config import SHAPE_TYPE, SHAPE_SIZE, WAYPOINT_TOLERANCE
            base_env = DrawingEnvironment(
                max_episode_steps=200,
                waypoint_tolerance=WAYPOINT_TOLERANCE,
                shape_type=SHAPE_TYPE,
                shape_size=SHAPE_SIZE,
                use_dynamic_workspace=require_board_detection,
                control_backend=control_backend,
                mirror_safe_moves=(control_backend != 'sim_to_real_shadow'),
            )
        else:
            base_env = RLEnvironment(
                max_episode_steps=200,
                goal_tolerance=0.01,
                control_backend=control_backend,
                mirror_safe_moves=(control_backend != 'sim_to_real_shadow'),
            )

        # Enable board tracking (DrawingEnvironment does this internally)
        if mode != 'drawing' and require_board_detection:
            print("📡 Enabling board tracking...")
            base_env.enable_board_tracking()

        # Wait for environment to initialize
        print("   Waiting for environment...")
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(base_env, timeout_sec=0.1)

        # Wait for ArUco board detection only when explicitly required
        if require_board_detection:
            print("\n⏳ Waiting for ArUco board detection...")
            if not base_env.wait_for_initial_detection(10.0):
                print("⚠️  No board detected — sphere visualization may be offset")
                print("   (Training still works, targets are in joint space)")
            else:
                print("✅ Board detected — visualization active")
        else:
            print("\n📡 Board detection: optional (skipped)")

        # Create PID tuning environment (wraps base_env)
        # Targets = random joints or drawing shape
        print("\n🎛️  Creating PID Tuning environment...")
        env = PIDTuningEnv(base_env, mode=mode, segment_steps=segment_steps)

        # Get training parameters
        print("\n📊 PID Tuning Configuration")
        print("="*70)

        episodes_input = input(f"Number of episodes (default 500): ").strip()
        episodes = int(episodes_input) if episodes_input else 500

        print(f"\n✅ Configuration:")
        print(f"   Episodes: {episodes}")
        print(f"   State dim: {env.state_dim} (24D)")
        print(f"   Action dim: {env.action_dim} (18D)")
        print(f"   Control backend: {control_backend}")
        print(f"   Require board detection: {require_board_detection}")
        print("="*70)

        # Create SAC agent for PID tuning (different dimensions from reaching/drawing)
        print("\n🤖 Creating SAC agent for PID tuning...")
        agent = SACAgentGazebo(
            state_dim=env.state_dim,     # 24D
            n_actions=env.action_dim,    # 18D
            max_action=np.ones(env.action_dim),
            min_action=-np.ones(env.action_dim),
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.99,
            tau=0.05,
            batch_size=256,
            buffer_size=int(1e6),
            auto_entropy_tuning=True
        )

        # Override checkpoint directory for PID tuning mode
        mode_suffix = f'sac_pid_tuning_{mode}_{control_backend}'
        agent.checkpoint_dir = os.path.join(
            os.path.dirname(__file__), 'checkpoints', mode_suffix
        )
        os.makedirs(agent.checkpoint_dir, exist_ok=True)
        print(f"   Checkpoint dir: {agent.checkpoint_dir}")

        # Explicitly choose warm-start vs fresh start.
        # For changed-controller experiments, loading an older actor can poison the comparison.
        best_actor = os.path.join(agent.checkpoint_dir, 'actor_sac_best.pth')
        fallback_best_actor = os.path.join(
            os.path.dirname(__file__), 'checkpoints', f'sac_pid_tuning_{mode}_sim', 'actor_sac_best.pth'
        )

        model_to_load = None
        if os.path.exists(best_actor):
            model_to_load = best_actor
        elif os.path.exists(fallback_best_actor):
            model_to_load = fallback_best_actor
            print(f"\nℹ️  No local checkpoint found for backend '{control_backend}'. Fallback available from sim: {os.path.basename(fallback_best_actor)}")

        if model_to_load:
            load_model = input("\n🧠 Load pre-trained PID tuning model? (y/n, default=y): ").strip().lower()
            if load_model != 'n':
                try:
                    agent.load_models(model_to_load)
                    print(f"   ✅ Loaded pre-trained PID tuning model from: {model_to_load}")
                except Exception as e:
                    print(f"   ⚠️  Failed to load model: {e}")
                    print("   Starting with untrained agent")
            else:
                print("   📝 Starting fresh (pre-trained PID model not loaded)")
        else:
            print("   📝 No pre-trained model found, starting fresh")

        # Try to load replay buffer
        load_buffer = input("\n📦 Load existing replay buffer? (y/n): ").strip().lower()
        if load_buffer == 'y':
            import glob
            pkl_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'pkl')
            buffer_files = sorted(
                glob.glob(os.path.join(pkl_dir, f"*replay_buffer*{mode_suffix}*.pkl")),
                key=os.path.getmtime, reverse=True
            )
            if buffer_files:
                default_buf = buffer_files[0]
                buf_path = input(f"   Path (Enter={os.path.basename(default_buf)}): ").strip()
                if not buf_path:
                    buf_path = default_buf
                if os.path.exists(buf_path):
                    try:
                        agent.replay_buffer.load(buf_path)
                        print(f"   ✅ Buffer loaded: {agent.replay_buffer.size()} transitions")
                    except Exception as e:
                        print(f"   ❌ Failed: {e}")
            else:
                print("   No buffer files found")

        # Training statistics
        episode_rewards = []
        episode_iaes = []
        episode_norm_iaes = []
        episode_efforts = []
        episode_smooth_delta = []
        episode_smooth_jerk = []
        episode_final_errors = []
        actor_losses = []
        critic_losses = []
        episode_cartesian_mm = []
        episode_max_wp_mm = []
        episode_waypoint_errors = []
        episode_all_paths = []
        episode_target_shapes = []
        episode_commanded_joint_traces = []
        episode_actual_joint_traces = []
        episode_joint_trace_times = []
        guard_events = []
        best_artifact = None
        best_artifact_path = None
        best_gains_path = None
        best_actual_path = None
        best_target_shape = None
        best_plot_episode_idx = None

        best_reward = -float('inf')

        # Results directory
        results_dir = os.path.join(os.path.dirname(__file__), 'training_results')
        pkl_dir = os.path.join(results_dir, 'pkl')
        png_dir = os.path.join(results_dir, 'png')
        csv_dir = os.path.join(results_dir, 'csv')
        os.makedirs(pkl_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        def finalize_pid_tuning(interrupted=False):
            if len(episode_rewards) == 0:
                print("   No episodes completed, skipping saving/plotting.")
                return

            # ================================================================
            # TRAINING COMPLETE
            # ================================================================
            print("\n" + "="*70)
            if interrupted:
                print("⚠️  PID TUNING TRAINING INTERRUPTED BY USER!")
            else:
                print("🎉 PID TUNING TRAINING COMPLETE!")
            print("="*70)

            # Summary
            print(f"\n📊 Summary ({len(episode_rewards)} episodes):")
            print(f"   Average Reward: {np.mean(episode_rewards):.2f}")
            print(f"   Best Reward: {max(episode_rewards):.2f}")
            print(f"   Average IAE: {np.mean(episode_iaes):.4f}")
            print(f"   Average Final Error: {np.mean(np.degrees(episode_final_errors)):.2f}°")

            best_gains = env.get_best_gains()
            if best_gains:
                print(f"\n   🏆 Best PID Gains (episode {best_gains['episode']}):")
                print(f"      Kp: {np.round(best_gains['Kp'], 2)}")
                print(f"      Ki: {np.round(best_gains['Ki'], 3)}")
                print(f"      Kd: {np.round(best_gains['Kd'], 3)}")

            # Save final results
            agent.save_models()
            agent.replay_buffer.save(
                os.path.join(pkl_dir, f'replay_buffer_final_{mode_suffix}_{timestamp}.pkl')
            )

            # Plot PID tuning results
            _plot_pid_tuning_results(
                episode_rewards, episode_iaes, episode_final_errors,
                actor_losses, critic_losses, env.get_gain_history(),
                png_dir, csv_dir, timestamp,
                cartesian_mm=episode_cartesian_mm,
                max_wp_mm=episode_max_wp_mm,
                waypoint_errors=episode_waypoint_errors,
                efforts=episode_efforts,
                normalized_iaes=episode_norm_iaes,
                smooth_deltas=episode_smooth_delta,
                smooth_jerks=episode_smooth_jerk,
                mode=mode,
                mode_suffix=mode_suffix
            )
            if mode == 'drawing':
                nonlocal best_plot_episode_idx, best_actual_path, best_target_shape
                if best_plot_episode_idx is None and episode_cartesian_mm:
                    best_plot_episode_idx = int(np.argmin(episode_cartesian_mm))
                    best_actual_path = episode_all_paths[best_plot_episode_idx] if best_plot_episode_idx < len(episode_all_paths) else None
                    best_target_shape = episode_target_shapes[best_plot_episode_idx] if best_plot_episode_idx < len(episode_target_shapes) else None

                # Save and report worst outliers (highest max waypoint miss)
                try:
                    outlier_k = 5
                    idx_sorted = sorted(
                        range(len(episode_max_wp_mm)),
                        key=lambda i: float(episode_max_wp_mm[i]) if episode_max_wp_mm[i] is not None else -1.0,
                        reverse=True
                    )
                    outliers = []
                    for rank, i in enumerate(idx_sorted[:outlier_k], start=1):
                        outliers.append({
                            'rank': rank,
                            'episode': i + 1,
                            'reward': float(episode_rewards[i]),
                            'iae': float(episode_iaes[i]),
                            'normalized_iae': float(episode_norm_iaes[i]) if i < len(episode_norm_iaes) else None,
                            'effort': float(episode_efforts[i]) if i < len(episode_efforts) else None,
                            'smooth_delta': float(episode_smooth_delta[i]) if i < len(episode_smooth_delta) else None,
                            'smooth_jerk': float(episode_smooth_jerk[i]) if i < len(episode_smooth_jerk) else None,
                            'avg_wp_error_mm': float(episode_cartesian_mm[i]) if i < len(episode_cartesian_mm) else None,
                            'max_wp_error_mm': float(episode_max_wp_mm[i]) if i < len(episode_max_wp_mm) else None,
                            'waypoint_errors_mm': episode_waypoint_errors[i] if i < len(episode_waypoint_errors) else None,
                            'actual_path_xyz': episode_all_paths[i] if i < len(episode_all_paths) else None,
                            'target_shape_xyz': episode_target_shapes[i] if i < len(episode_target_shapes) else None,
                        })

                    import pickle
                    outliers_path = os.path.join(pkl_dir, f'pid_outliers_{mode_suffix}_{timestamp}.pkl')
                    with open(outliers_path, 'wb') as f:
                        pickle.dump(outliers, f)

                    print("\n🔥 Worst Episodes By Max Waypoint Miss")
                    for o in outliers:
                        print(
                            f"  #{o['rank']} ep={o['episode']:4d} "
                            f"maxWP={o['max_wp_error_mm']:.1f}mm avgWP={o['avg_wp_error_mm']:.1f}mm "
                            f"R={o['reward']:.2f} normIAE={o['normalized_iae']:.3f}"
                        )
                    print(f"💾 Outliers saved: {outliers_path}")
                except Exception as e:
                    print(f"⚠️  Outlier report failed: {e}")

                _plot_pid_joint_tracking(
                    commanded_traces=episode_commanded_joint_traces,
                    actual_traces=episode_actual_joint_traces,
                    time_traces=episode_joint_trace_times,
                    joint_names=env.base_env.motion_backend.mapper.pi_joint_names,
                    png_dir=png_dir,
                    timestamp=timestamp,
                    representative_idx=best_plot_episode_idx,
                    rewards=episode_rewards,
                    avg_wp_mm=episode_cartesian_mm,
                    max_wp_mm=episode_max_wp_mm,
                    plot_stem=f'pid_joint_tracking_{mode_suffix}'
                )
                _plot_drawing_trajectory(episode_all_paths, best_actual_path, best_target_shape, png_dir, timestamp, plot_stem=f'pid_trajectory_{mode_suffix}')

            # Save training results for continuation
            import pickle
            results = {
                'episode_rewards': episode_rewards,
                'episode_iaes': episode_iaes,
                'episode_final_errors': episode_final_errors,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses,
                'gain_history': env.get_gain_history(),
                'guard_events': guard_events,
                'control_backend': control_backend,
                'mode': mode,
                'require_board_detection': require_board_detection,
                'best_artifact_path': best_artifact_path,
                'best_gains_path': best_gains_path,
                'episode_commanded_joint_traces': episode_commanded_joint_traces,
                'episode_actual_joint_traces': episode_actual_joint_traces,
                'episode_joint_trace_times': episode_joint_trace_times,
            }
            results_path = os.path.join(pkl_dir, f'training_results_{mode_suffix}_{timestamp}.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"💾 Results saved to: {results_path}")

            # Cleanup old files
            cleanup_old_files(pkl_dir, f"replay_buffer_ep*{mode_suffix}*.pkl", 4)
            cleanup_old_files(pkl_dir, f"replay_buffer_final*{mode_suffix}*.pkl", 1)
            cleanup_old_files(pkl_dir, f"pid_best_artifact*{mode_suffix}*.pkl", 3)
            cleanup_old_files(pkl_dir, f"training_results_{mode_suffix}_*.pkl", 3)
            cleanup_old_files(pkl_dir, f"pid_outliers_{mode_suffix}_*.pkl", 3)
            cleanup_old_files(png_dir, f"pid_tuning_{mode_suffix}_*.png", 3)
            cleanup_old_files(png_dir, f"pid_joint_tracking_{mode_suffix}_*.png", 3)
            cleanup_old_files(png_dir, f"pid_trajectory_{mode_suffix}_*.png", 3)
            cleanup_old_files(csv_dir, f"pid_tuning_{mode_suffix}_*.csv", 3)
            # Clean up raw episode logs in logs/ dir
            logs_dir = os.path.join(results_dir, 'logs')
            cleanup_old_files(logs_dir, f"shadow_pid_episode_log_*.txt", 3)
            cleanup_old_files(logs_dir, f"shadow_pid_episode_start_log_*.txt", 3)
            print(f"🧹 Cleaned up old buffer, result, and plot files for {mode_suffix}")

        print("\n🚀 Starting PID tuning training...\n")

        LEARNING_STARTS = 10
        OPT_STEPS = 32
        SAVE_INTERVAL = 25
        GUARD_HEALTH_AVG_WP_MM = 5.0
        GUARD_HEALTH_MAX_WP_MM = 10.0
        GUARD_HEALTH_NORM_IAE = 12.0
        GUARD_HEALTH_FINAL_ERR_DEG = 1.0
        GUARD_MIN_HEALTHY_STREAK = 3
        GUARD_COOLDOWN_EPISODES = 5
        guard_snapshot = _capture_sac_snapshot(agent)
        guard_snapshot_episode = 0
        healthy_streak = 0
        catastrophic_streak = 0
        cooldown_remaining = 0
        healthy_metrics_window = []
        HEALTHY_WINDOW_MAX = 20

        for episode in range(episodes):
            episode_start = time.time()

            # Reset environment (moves to home, generates target)
            state = env.reset()

            # RL agent selects PID gains
            action = agent.select_action(state, evaluate=(cooldown_remaining > 0))

            # Execute trajectory with selected PID gains
            next_state, reward, done, info = env.step(action)

            avg_wp = float(info.get('avg_wp_error_mm', info.get('cartesian_dist_mm', 0.0)))
            max_wp = float(info.get('max_wp_error_mm', info.get('cartesian_dist_mm', 0.0)))
            norm_iae = float(info.get('normalized_iae', 0.0))
            final_err_deg = float(np.degrees(info['final_error']))

            if healthy_metrics_window:
                avg_roll = float(np.median([m['avg_wp'] for m in healthy_metrics_window]))
                max_roll = float(np.median([m['max_wp'] for m in healthy_metrics_window]))
                norm_roll = float(np.median([m['norm_iae'] for m in healthy_metrics_window]))
                final_roll = float(np.median([m['final_deg'] for m in healthy_metrics_window]))
            else:
                avg_roll = max_roll = norm_roll = final_roll = 0.0

            severe_divergence = (
                avg_wp > max(20.0, 6.0 * max(avg_roll, 1.0)) or
                max_wp > max(60.0, 8.0 * max(max_roll, 1.0)) or
                norm_iae > max(40.0, 6.0 * max(norm_roll, 1.0)) or
                final_err_deg > max(8.0, 6.0 * max(final_roll, 0.25))
            )
            catastrophic = (
                avg_wp > max(10.0, 3.0 * max(avg_roll, 1.0)) or
                max_wp > max(25.0, 4.0 * max(max_roll, 1.0)) or
                norm_iae > max(20.0, 3.0 * max(norm_roll, 1.0)) or
                final_err_deg > max(3.0, 4.0 * max(final_roll, 0.25))
            )
            healthy = (
                avg_wp <= GUARD_HEALTH_AVG_WP_MM and
                max_wp <= GUARD_HEALTH_MAX_WP_MM and
                norm_iae <= GUARD_HEALTH_NORM_IAE and
                final_err_deg <= GUARD_HEALTH_FINAL_ERR_DEG
            )

            should_store = True
            trigger_recovery = False

            if episode >= LEARNING_STARTS and catastrophic:
                should_store = False
                catastrophic_streak += 1
                healthy_streak = 0

                if severe_divergence or catastrophic_streak >= 2:
                    trigger_recovery = True
            else:
                catastrophic_streak = 0
                if healthy:
                    healthy_streak += 1
                    healthy_metrics_window.append({
                        'avg_wp': avg_wp,
                        'max_wp': max_wp,
                        'norm_iae': norm_iae,
                        'final_deg': final_err_deg,
                    })
                    if len(healthy_metrics_window) > HEALTHY_WINDOW_MAX:
                        healthy_metrics_window.pop(0)

                    if episode >= LEARNING_STARTS and healthy_streak >= GUARD_MIN_HEALTHY_STREAK:
                        guard_snapshot = _capture_sac_snapshot(agent)
                        guard_snapshot_episode = episode + 1
                else:
                    healthy_streak = 0

            if trigger_recovery and guard_snapshot is not None:
                _restore_sac_snapshot(agent, guard_snapshot)
                cooldown_remaining = GUARD_COOLDOWN_EPISODES
                catastrophic_streak = 0
                guard_events.append({
                    'episode': episode + 1,
                    'reward': reward,
                    'avg_wp_mm': avg_wp,
                    'max_wp_mm': max_wp,
                    'norm_iae': norm_iae,
                    'final_error_deg': final_err_deg,
                    'rollback_to_episode': guard_snapshot_episode,
                })
                print(
                    f"   🛟 Divergence guard: rollback to healthy snapshot from ep {guard_snapshot_episode} "
                    f"(ep {episode+1} skipped, cooldown {cooldown_remaining} eps)"
                )

            # Store transition (single-step MDP) unless guard quarantines it
            if should_store:
                agent.store_transition(state, action, reward, next_state, float(done))

            # Training updates
            a_loss, c_loss = None, None
            if episode >= LEARNING_STARTS and cooldown_remaining == 0:
                for _ in range(OPT_STEPS):
                    a_loss, c_loss = agent.train()
            elif cooldown_remaining > 0:
                cooldown_remaining -= 1

            # Log statistics
            episode_rewards.append(reward)
            episode_iaes.append(info['iae'])
            episode_norm_iaes.append(norm_iae)
            episode_efforts.append(info.get('effort', 0.0))
            episode_smooth_delta.append(info.get('normalized_command_delta', 0.0))
            episode_smooth_jerk.append(info.get('normalized_command_jerk', 0.0))
            episode_final_errors.append(info['final_error'])
            actor_losses.append(a_loss)
            critic_losses.append(c_loss)
            episode_cartesian_mm.append(info.get('cartesian_dist_mm', 0.0))
            episode_max_wp_mm.append(info.get('max_wp_error_mm', info.get('cartesian_dist_mm', 0.0)))
            episode_waypoint_errors.append(info.get('waypoint_errors_mm', None))
            episode_commanded_joint_traces.append(info.get('commanded_trajectory_rad', []))
            episode_actual_joint_traces.append(info.get('actual_joint_trace_rad', []))
            episode_joint_trace_times.append(info.get('joint_trace_time_sec', []))

            if mode == 'drawing':
                episode_all_paths.append(info.get('actual_path_xyz', []))
                episode_target_shapes.append(info.get('target_shape_xyz', None))

            episode_time = time.time() - episode_start

            # Print progress
            avg_reward = np.mean(episode_rewards[-50:])
            avg_iae = np.mean(episode_iaes[-50:])

            gains = info['gains']
            kp_mean = np.mean(gains['Kp'])
            ki_mean = np.mean(gains['Ki'])
            kd_mean = np.mean(gains['Kd'])

            if mode == 'drawing':
                print(f"Ep {episode+1:4d}/{episodes} | "
                      f"R: {reward:8.2f} | "
                      f"IAE: {info['iae']:6.1f} | "
                      f"AvgWp: {avg_wp:5.1f}mm MaxWp: {max_wp:5.1f}mm | "
                      f"Kp̄={kp_mean:.2f} Ki̊={ki_mean:.3f} Kd̄={kd_mean:.3f} | "
                      f"{episode_time:.1f}s")
            else:
                print(f"Ep {episode+1:4d}/{episodes} | "
                      f"R: {reward:8.2f} | "
                      f"IAE: {info['iae']:6.3f} | "
                      f"CartesianMiss: {info['cartesian_dist_mm']:5.1f}mm | "
                      f"Kp̄={kp_mean:.2f} Ki̊={ki_mean:.3f} Kd̄={kd_mean:.3f} | "
                      f"{episode_time:.1f}s")

            # Save best model
            if reward > best_reward and episode >= LEARNING_STARTS:
                best_reward = reward
                agent.save_models()
                print(f"   \U0001f4be New best! Reward={reward:.2f}")

                if mode == 'drawing':
                    best_actual_path = info.get('actual_path_xyz', None)
                    best_target_shape = info.get('target_shape_xyz', None)
                best_plot_episode_idx = episode

                # Save best gains
                best_gains = env.get_best_gains()
                if best_gains:
                    import json
                    gains_path = os.path.join(agent.checkpoint_dir, f'best_gains_{mode_suffix}.json')
                    gains_save = {
                        k: v.tolist() if hasattr(v, 'tolist') else v
                        for k, v in best_gains.items()
                    }
                    with open(gains_path, 'w') as f:
                        json.dump(gains_save, f, indent=2)
                    best_gains_path = gains_path

                best_artifact = env.get_last_episode_artifact()
                if best_artifact is not None:
                    import pickle
                    best_artifact_path = os.path.join(
                        pkl_dir, f'pid_best_artifact_{mode_suffix}_{timestamp}.pkl'
                    )
                    with open(best_artifact_path, 'wb') as f:
                        pickle.dump(best_artifact, f)
                    print(f"   💾 Replay artifact saved: {best_artifact_path}")

                    replay_error = best_artifact.get('replay_export_error')
                    auto_best_replay = os.environ.get('PID_SHADOW_REPLAY_BEST', '0').strip().lower() in {'1', 'true', 'yes', 'y'}
                    if replay_error:
                        print(f"   ⚠️ Replay export unavailable: {replay_error}")
                    elif control_backend == 'sim_to_real_shadow' and auto_best_replay:
                        replay_ok = env.replay_artifact(
                            best_artifact,
                            label=f'{mode_suffix}_ep{episode+1}',
                        )
                        if not replay_ok:
                            print("   ⚠️ Shadow replay failed on hardware backend")
                    elif control_backend == 'sim_to_real_shadow':
                        print("   ⏭️ Shadow replay skipped; use exported artifact/JSON for manual Pi-local replay")

            # Periodic saves
            if (episode + 1) % SAVE_INTERVAL == 0:
                agent.save_models(episode=episode+1)
                agent.replay_buffer.save(
                    os.path.join(pkl_dir, f'replay_buffer_ep{episode+1}_{mode_suffix}_{timestamp}.pkl')
                )
                print(f"   💾 Checkpoint saved (episode {episode+1})")

        finalize_pid_tuning(interrupted=False)
        print(f"\n✅ PID tuning training complete! Trained for {len(episode_rewards)} episodes.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        finalize_pid_tuning(interrupted=True)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        auto_home_on_exit = os.environ.get('PID_AUTO_HOME_ON_EXIT', '1').strip().lower() in {'1', 'true', 'yes', 'y'}
        if env is not None and auto_home_on_exit:
            try:
                print("\n🏠 Returning robot to home position before exit...")
                env.base_env.home(duration=2.0)
                time.sleep(2.0)
            except Exception as e:
                print(f"   ⚠️ Could not return home: {e}")
        elif env is not None:
            print("\n⏭️ Auto-home on exit skipped (unset PID_AUTO_HOME_ON_EXIT or set it to 1 to enable)")

        if env is not None and hasattr(env, 'base_env'):
            try:
                env.base_env.destroy_node()
            except Exception:
                pass
        if ros_initialized:
            try:
                rclpy.shutdown()
            except Exception:
                pass


def _plot_pid_tuning_results(rewards, iaes, final_errors, actor_losses, critic_losses,
                             gain_history, png_dir, csv_dir, timestamp,
                             cartesian_mm=None, max_wp_mm=None, waypoint_errors=None,
                             efforts=None, normalized_iaes=None,
                             smooth_deltas=None, smooth_jerks=None,
                             mode='reaching', mode_suffix=None):
    """Plot PID tuning training statistics."""
    episodes = np.arange(1, len(rewards) + 1)
    is_drawing = mode == 'drawing' and cartesian_mm is not None and len(cartesian_mm) > 0

    def cumulative_avg(data):
        return [np.mean(data[:i+1]) for i in range(len(data))]

    def rolling_avg(data, window=20):
        out = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            out.append(np.mean(data[start:i+1]))
        return out

    def rolling_rate(flags, window=50):
        out = []
        for i in range(len(flags)):
            start = max(0, i - window + 1)
            out.append(np.mean(flags[start:i+1]))
        return out

    # Use 3x3 grid for drawing mode, 2x3 for reaching
    if is_drawing:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    mode_label = 'Drawing' if is_drawing else 'Reaching'
    fig.suptitle(f'PID Tuning Training — RL-Optimized Gains ({mode_label})', fontsize=16, fontweight='bold')

    # ── Row 0, Col 0: Rewards ──
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=1.5)
    ax.plot(episodes, cumulative_avg(rewards), color='darkblue', linewidth=3.0, label='Avg Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Row 0, Col 1: IAE ──
    ax = axes[0, 1]
    ax.plot(episodes, iaes, alpha=0.3, color='orange', linewidth=1.5)
    ax.plot(episodes, cumulative_avg(iaes), color='darkorange', linewidth=3.0, label='Avg IAE')
    if normalized_iaes is not None and len(normalized_iaes) == len(iaes):
        ax.plot(episodes, normalized_iaes, alpha=0.15, color='sienna', linewidth=1.0)
        ax.plot(episodes, rolling_avg(normalized_iaes, window=50),
                color='saddlebrown', linewidth=2.5, label='Norm IAE (roll 50)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('IAE (rad·steps)')
    ax.set_title('Integral Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Row 0, Col 2: Final Error ──
    ax = axes[0, 2]
    final_errors_deg = [np.degrees(e) for e in final_errors]
    ax.plot(episodes, final_errors_deg, alpha=0.3, color='red', linewidth=1.5)
    ax.plot(episodes, cumulative_avg(final_errors_deg), color='darkred', linewidth=3.0, label='Avg Error')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Final Error (°)')
    ax.set_title('Final Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Row 1, Col 0: Success Rates / Effort (high-signal) ──
    ax = axes[1, 0]
    if is_drawing and cartesian_mm is not None and max_wp_mm is not None:
        ok_avg = [v <= 5.0 for v in cartesian_mm]
        ok_max = [v <= 10.0 for v in max_wp_mm]
        ax.plot(episodes, rolling_rate(ok_avg, window=50), color='green', linewidth=2.5, label='P(AvgWP<=5mm) roll50')
        ax.plot(episodes, rolling_rate(ok_max, window=50), color='orange', linewidth=2.5, label='P(MaxWP<=10mm) roll50')
        ax.set_ylabel('Rate')
        ax.set_ylim(-0.02, 1.02)
        ax.set_title('Quality Success Rates')
        ax.legend(loc='lower right', fontsize=9)
    elif efforts is not None and len(efforts) == len(rewards):
        ax.plot(episodes, efforts, alpha=0.25, color='gray', linewidth=1.0)
        ax.plot(episodes, rolling_avg(efforts, window=50), color='black', linewidth=2.5, label='Effort roll50')
        ax.set_ylabel('Effort')
        ax.set_title('Control Effort')
        ax.legend()
    else:
        ax.text(0.05, 0.5, "No success-rate data", transform=ax.transAxes)
        ax.set_title('Quality')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)

    # ── Row 1, Col 1: PID Gain Evolution ──
    ax = axes[1, 1]
    if gain_history:
        kp_means = [np.mean(g['Kp']) for g in gain_history]
        ki_means = [np.mean(g['Ki']) for g in gain_history]
        kd_means = [np.mean(g['Kd']) for g in gain_history]
        gh_eps = [g['episode'] for g in gain_history]
        ax.plot(gh_eps, kp_means, color='red', linewidth=2, label='Kp (mean)')
        ax.plot(gh_eps, ki_means, color='green', linewidth=2, label='Ki (mean)')
        ax.plot(gh_eps, kd_means, color='blue', linewidth=2, label='Kd (mean)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Gain Value')
    ax.set_title('PID Gain Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Row 1, Col 2: Summary ──
    ax = axes[1, 2]
    ax.axis('off')
    best_r = max(rewards)
    best_iae = min(iaes)
    summary_lines = [
        f"PID Tuning Summary ({mode_label})",
        f"-------------------------",
        f"",
        f"Episodes: {len(rewards)}",
        f"",
        f"Rewards:",
        f"  - Average: {np.mean(rewards):.2f}",
        f"  - Best: {best_r:.2f}",
        f"",
        f"Tracking Quality:",
        f"  - Best IAE: {best_iae:.4f}",
        f"  - Avg Final Error: {np.mean(final_errors_deg):.2f} deg",
        f"  - Best Final Error: {min(final_errors_deg):.2f} deg",
    ]
    if smooth_deltas is not None and len(smooth_deltas) == len(rewards):
        summary_lines += [
            f"",
            f"Smoothness:",
            f"  - Avg Cmd Delta: {np.mean(smooth_deltas):.2f}",
        ]
        if smooth_jerks is not None and len(smooth_jerks) == len(rewards):
            summary_lines.append(f"  - Avg Cmd Jerk: {np.mean(smooth_jerks):.2f}")
    if is_drawing:
        summary_lines += [
            f"",
            f"Drawing Accuracy:",
            f"  - Avg WP Miss: {np.mean(cartesian_mm):.1f}mm",
            f"  - Best Avg WP: {min(cartesian_mm):.1f}mm",
            f"  - Avg Max WP: {np.mean(max_wp_mm):.1f}mm",
        ]
    summary = "\n".join(summary_lines)
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    # ── Drawing-specific plots (Row 2) ──
    if is_drawing:
        # Row 2, Col 0: Avg Waypoint Miss (mm)
        ax = axes[2, 0]
        ax.plot(episodes, cartesian_mm, alpha=0.3, color='purple', linewidth=1.5)
        ax.plot(episodes, rolling_avg(cartesian_mm), color='darkviolet', linewidth=3.0, label='Rolling Avg (20)')
        ax.axhline(y=5.0, color='green', linestyle='--', alpha=0.5, label='5mm target')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Waypoint Miss (mm)')
        ax.set_title('Per-Waypoint Cartesian Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Row 2, Col 1: Max Waypoint Miss (mm)
        ax = axes[2, 1]
        ax.plot(episodes, max_wp_mm, alpha=0.3, color='crimson', linewidth=1.5)
        ax.plot(episodes, rolling_avg(max_wp_mm), color='darkred', linewidth=3.0, label='Rolling Avg (20)')
        ax.axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='10mm target')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Max Waypoint Miss (mm)')
        ax.set_title('Worst Waypoint Per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Row 2, Col 2: Waypoint error profile (mean + p90) or fallback to spread
        ax = axes[2, 2]
        if waypoint_errors is not None and len(waypoint_errors) > 0:
            import numpy as _np
            max_len = max(len(w) for w in waypoint_errors if w is not None)
            if max_len > 0:
                mat = _np.full((len(waypoint_errors), max_len), _np.nan, dtype=_np.float64)
                for i, w in enumerate(waypoint_errors):
                    if w is None:
                        continue
                    w = _np.array(w, dtype=_np.float64)
                    mat[i, : min(max_len, len(w))] = w[:max_len]
                mean_wp = _np.nanmean(mat, axis=0)
                p90_wp = _np.nanpercentile(mat, 90, axis=0)
                x = _np.arange(1, len(mean_wp) + 1)
                ax.plot(x, mean_wp, color='teal', linewidth=2.5, label='Mean WP error')
                ax.plot(x, p90_wp, color='darkcyan', linewidth=2.5, label='P90 WP error')
                ax.axhline(y=5.0, color='green', linestyle='--', alpha=0.4, label='5mm')
                ax.axhline(y=10.0, color='orange', linestyle='--', alpha=0.4, label='10mm')
                ax.set_xlabel('Waypoint Index')
                ax.set_ylabel('Error (mm)')
                ax.set_title('Waypoint Error Profile')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                waypoint_errors = None
        if waypoint_errors is None:
            spread = [m - a for a, m in zip(cartesian_mm, max_wp_mm)]
            ax.plot(episodes, spread, alpha=0.3, color='teal', linewidth=1.5)
            ax.plot(episodes, rolling_avg(spread), color='darkcyan', linewidth=3.0, label='Rolling Avg (20)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Max - Avg WP Error (mm)')
            ax.set_title('Waypoint Consistency (lower = more uniform)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename_suffix = f'_{mode_suffix}' if mode_suffix else ''
    plot_path = os.path.join(png_dir, f'pid_tuning{filename_suffix}_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 PID tuning plot saved: {plot_path}")

    # Save CSV
    import csv
    csv_path = os.path.join(csv_dir, f'pid_tuning{filename_suffix}_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Episode', 'Reward', 'IAE', 'NormIAE', 'Effort', 'FinalError_deg',
                  'SmoothDelta', 'SmoothJerk',
                  'Kp_mean', 'Ki_mean', 'Kd_mean', 'Actor_Loss', 'Critic_Loss']
        if is_drawing:
            header += ['AvgWpMiss_mm', 'MaxWpMiss_mm']
        writer.writerow(header)

        for i in range(len(rewards)):
            gh = gain_history[i] if i < len(gain_history) else None
            row = [
                i+1,
                f'{rewards[i]:.4f}',
                f'{iaes[i]:.4f}',
                f'{normalized_iaes[i]:.6f}' if normalized_iaes is not None and i < len(normalized_iaes) else '',
                f'{efforts[i]:.6f}' if efforts is not None and i < len(efforts) else '',
                f'{final_errors_deg[i]:.4f}',
                f'{smooth_deltas[i]:.6f}' if smooth_deltas is not None and i < len(smooth_deltas) else '',
                f'{smooth_jerks[i]:.6f}' if smooth_jerks is not None and i < len(smooth_jerks) else '',
                f'{np.mean(gh["Kp"]):.4f}' if gh else '',
                f'{np.mean(gh["Ki"]):.4f}' if gh else '',
                f'{np.mean(gh["Kd"]):.4f}' if gh else '',
                f'{actor_losses[i]:.6f}' if actor_losses[i] is not None else '',
                f'{critic_losses[i]:.6f}' if critic_losses[i] is not None else '',
            ]
            if is_drawing:
                row.append(f'{cartesian_mm[i]:.2f}' if i < len(cartesian_mm) else '')
                row.append(f'{max_wp_mm[i]:.2f}' if i < len(max_wp_mm) else '')
            writer.writerow(row)
    print(f"📊 PID tuning CSV saved: {csv_path}")


def _plot_pid_joint_tracking(commanded_traces, actual_traces, time_traces, joint_names,
                             png_dir, timestamp, representative_idx=None,
                             rewards=None, avg_wp_mm=None, max_wp_mm=None,
                             plot_stem='pid_joint_tracking',
                             title_prefix='PID Joint Tracking'):
    """Create a deploy-style per-joint tracking plot for PID tuning episodes."""
    valid = []
    for ep_idx, (cmd_trace, act_trace) in enumerate(zip(commanded_traces, actual_traces)):
        if not cmd_trace or not act_trace:
            continue
        cmd = np.degrees(np.asarray(cmd_trace, dtype=np.float64))
        act = np.degrees(np.asarray(act_trace, dtype=np.float64))
        if cmd.ndim != 2 or act.ndim != 2 or cmd.shape[1] != len(joint_names) or act.shape[1] != len(joint_names):
            continue
        common_len = min(len(cmd), len(act))
        if common_len < 2:
            continue
        valid.append((ep_idx, cmd[:common_len], act[:common_len]))

    if not valid:
        return

    if representative_idx is None:
        representative_idx = valid[0][0]

    representative = next((item for item in valid if item[0] == representative_idx), valid[0])
    rep_ep_idx, rep_cmd, rep_act = representative

    common_len_all = min(len(cmd) for _, cmd, _ in valid)
    actual_stack = np.stack([act[:common_len_all] for _, _, act in valid], axis=0)
    mean_actual = np.mean(actual_stack, axis=0)

    rep_time = None
    if time_traces and rep_ep_idx < len(time_traces):
        t = np.asarray(time_traces[rep_ep_idx], dtype=np.float64)
        if t.ndim == 1 and len(t) >= len(rep_cmd):
            rep_time = t[:len(rep_cmd)]
    if rep_time is None:
        rep_time = np.arange(len(rep_cmd), dtype=np.float64) * 0.02

    title_parts = [f"{title_prefix} (Representative Episode {rep_ep_idx + 1})"]
    if rewards is not None and rep_ep_idx < len(rewards):
        title_parts.append(f"Reward: {rewards[rep_ep_idx]:.2f}")
    if avg_wp_mm is not None and rep_ep_idx < len(avg_wp_mm):
        if avg_wp_mm[rep_ep_idx] is not None:
            title_parts.append(f"Avg WP: {avg_wp_mm[rep_ep_idx]:.1f}mm")
    if max_wp_mm is not None and rep_ep_idx < len(max_wp_mm):
        if max_wp_mm[rep_ep_idx] is not None:
            title_parts.append(f"Max WP: {max_wp_mm[rep_ep_idx]:.1f}mm")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(" | ".join(title_parts), fontsize=14, fontweight='bold')

    for joint_idx, (ax, joint_name) in enumerate(zip(axes.ravel(), joint_names)):
        for ep_idx, _, act in valid:
            ax.plot(rep_time[:common_len_all], act[:common_len_all, joint_idx],
                    color='gray', alpha=0.10, linewidth=0.8)
        ax.plot(rep_time[:common_len_all], mean_actual[:, joint_idx],
                color='black', alpha=0.85, linewidth=1.8, label='Mean actual')
        ax.plot(rep_time, rep_cmd[:, joint_idx], 'r--', linewidth=2.0, label='Representative command')
        ax.plot(rep_time, rep_act[:, joint_idx], color='tab:blue', linewidth=2.0, label='Representative actual')
        ax.set_title(f'Joint: {joint_name}')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Angle (deg)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(png_dir, f'{plot_stem}_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 PID joint tracking plot saved: {plot_path}")


def _plot_drawing_trajectory(all_paths, best_path, target_shape, png_dir, timestamp,
                             plot_stem='pid_trajectory',
                             title_prefix='PID Tuning (Drawing): Trajectory Quality'):
    """Plot the actual tracked trajectory versus the target shape (2D + 3D)."""
    if best_path is None or target_shape is None or len(best_path) == 0:
        return
    import numpy as _np

    def _resample(path_xyz, n=220):
        path_xyz = _np.asarray(path_xyz, dtype=_np.float64)
        if len(path_xyz) < 2:
            return None
        t_old = _np.linspace(0.0, 1.0, len(path_xyz))
        t_new = _np.linspace(0.0, 1.0, n)
        out = _np.zeros((n, 3), dtype=_np.float64)
        for k in range(3):
            out[:, k] = _np.interp(t_new, t_old, path_xyz[:, k])
        return out

    # Inputs are in meters (FK output). Convert to mm once, deterministically.
    target_m = _np.asarray(target_shape, dtype=_np.float64)
    best_m = _np.asarray(best_path, dtype=_np.float64)
    target = target_m * 1000.0
    best = best_m * 1000.0

    valid_paths = []
    for p in all_paths:
        if p is None or len(p) < 2:
            continue
        rp = _resample(_np.asarray(p, dtype=_np.float64), n=220)  # meters
        if rp is not None:
            valid_paths.append(rp * 1000.0)  # mm

    best_r = _resample(best_m, n=220)
    if best_r is not None:
        best_r = best_r * 1000.0
    else:
        best_r = best

    # Compute mean + band in Y/Z/X
    if valid_paths:
        stack = _np.stack(valid_paths, axis=0)
        mean = _np.mean(stack, axis=0)
        p10 = _np.percentile(stack, 10, axis=0)
        p90 = _np.percentile(stack, 90, axis=0)
    else:
        mean = best_r
        p10 = best_r
        p90 = best_r

    fig = plt.figure(figsize=(24, 7))
    fig.suptitle(title_prefix, fontsize=16, fontweight='bold')
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    # Panel 1: Y-Z with percentile band (removes redundancy vs arrow plot)
    ty, tz = target[:, 1], target[:, 2]
    target_y_closed = _np.append(ty, ty[0])
    target_z_closed = _np.append(tz, tz[0])

    ax1.plot(target_y_closed, target_z_closed, color='blue', linestyle='--', linewidth=2.5, label='Target Shape', zorder=10)
    if valid_paths:
        for p in valid_paths[::max(1, len(valid_paths)//60)]:
            ax1.plot(p[:, 1], p[:, 2], color='gray', alpha=0.06, linewidth=0.8)
    ax1.fill_betweenx(mean[:, 2], p10[:, 1], p90[:, 1], color='orange', alpha=0.15, label='P10–P90 band')
    ax1.plot(mean[:, 1], mean[:, 2], color='darkorange', linewidth=3.0, label='Mean Trajectory', zorder=11)
    ax1.plot(best_r[:, 1], best_r[:, 2], color='red', alpha=0.9, linewidth=2.0, label='Best Episode', zorder=12)
    ax1.scatter(ty, tz, color='blue', s=25, zorder=13)
    ax1.set_xlabel('Y (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_title(f'Board Plane (Y-Z)  |  Episodes: {len(valid_paths)}')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend(loc='lower left', fontsize=9)

    # Panel 2: X drift vs progress (this is what 2D Y-Z hides)
    ax2.plot(mean[:, 0], color='black', linewidth=2.5, label='Mean X')
    ax2.fill_between(_np.arange(len(mean)), p10[:, 0], p90[:, 0], color='gray', alpha=0.2, label='P10–P90 X')
    ax2.plot(best_r[:, 0], color='red', alpha=0.9, linewidth=2.0, label='Best X')
    ax2.axhline(y=_np.mean(target[:, 0]), color='blue', linestyle='--', alpha=0.6, label='Target X mean')
    ax2.set_xlabel('Progress (resampled)')
    ax2.set_ylabel('X (mm)')
    ax2.set_title('Off-Plane Drift (X over time)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Panel 3: 3D view (readable defaults)
    ax3.plot(target[:, 0], target[:, 1], target[:, 2], color='blue', linestyle='--', linewidth=2.0, label='Target')
    ax3.plot(best_r[:, 0], best_r[:, 1], best_r[:, 2], color='red', linewidth=2.0, label='Best')
    ax3.scatter(target[:, 0], target[:, 1], target[:, 2], color='blue', s=18)
    ax3.scatter(best_r[0, 0], best_r[0, 1], best_r[0, 2], color='green', s=70, marker='*', label='Start')
    ax3.scatter(best_r[-1, 0], best_r[-1, 1], best_r[-1, 2], color='purple', s=70, marker='X', label='End')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.set_title('3D Trajectory')
    ax3.view_init(elev=18, azim=-55)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(png_dir, f'{plot_stem}_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Trajectory plot saved: {plot_path}")


def _compute_resampled_waypoint_errors(path_xyz, target_shape_xyz):
    """Compare an actual path against ordered target waypoints after progress resampling."""
    path = np.asarray(path_xyz, dtype=np.float64)
    target = np.asarray(target_shape_xyz, dtype=np.float64)
    if path.ndim != 2 or target.ndim != 2 or path.shape[1] != 3 or target.shape[1] != 3:
        return None, None, None
    if len(path) < 2 or len(target) < 2:
        return None, None, None

    t_old = np.linspace(0.0, 1.0, len(path))
    t_new = np.linspace(0.0, 1.0, len(target))
    resampled = np.zeros_like(target)
    for axis in range(3):
        resampled[:, axis] = np.interp(t_new, t_old, path[:, axis])

    errors_mm = np.linalg.norm(resampled - target, axis=1) * 1000.0
    return float(np.mean(errors_mm)), float(np.max(errors_mm)), errors_mm.tolist()


def _plot_deploy_drawing_summary(endpoint_mm, avg_wp_mm, max_wp_mm, joint_errors, durations,
                                 waypoint_errors, png_dir, timestamp, replay_rate_hz):
    """Create a training-style deploy summary focused on drawing accuracy."""
    if not endpoint_mm:
        return

    episodes = np.arange(1, len(endpoint_mm) + 1)

    def rolling_avg(values, window=20):
        out = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            out.append(np.mean(values[start:i+1]))
        return out

    def rolling_rate(flags, window=50):
        out = []
        for i in range(len(flags)):
            start = max(0, i - window + 1)
            out.append(np.mean(flags[start:i+1]))
        return out

    valid_avg_wp = [v for v in avg_wp_mm if v is not None]
    valid_max_wp = [v for v in max_wp_mm if v is not None]

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(
        f'Deploy Replay — Drawing Quality (Pi at {replay_rate_hz:.1f}Hz)',
        fontsize=16,
        fontweight='bold',
    )

    ax = axes[0, 0]
    ax.plot(episodes, endpoint_mm, alpha=0.3, color='tab:blue', linewidth=1.5)
    ax.plot(episodes, rolling_avg(endpoint_mm), color='navy', linewidth=3.0, label='Rolling Avg (20)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Endpoint Miss (mm)')
    ax.set_title('Final Endpoint Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    avg_series = [np.nan if v is None else v for v in avg_wp_mm]
    ax.plot(episodes, avg_series, alpha=0.3, color='purple', linewidth=1.5)
    if valid_avg_wp:
        ax.plot(episodes, rolling_avg([np.nanmean(avg_series[:i+1]) if np.isnan(avg_series[i]) else avg_series[i] for i in range(len(avg_series))]),
                alpha=0.0)
        ax.plot(episodes, np.array([np.nanmean(avg_series[max(0, i - 19):i+1]) for i in range(len(avg_series))]),
                color='darkviolet', linewidth=3.0, label='Rolling Avg (20)')
    ax.axhline(y=5.0, color='green', linestyle='--', alpha=0.5, label='5mm target')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Waypoint Miss (mm)')
    ax.set_title('Per-Waypoint Cartesian Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    max_series = [np.nan if v is None else v for v in max_wp_mm]
    ax.plot(episodes, max_series, alpha=0.3, color='crimson', linewidth=1.5)
    if valid_max_wp:
        ax.plot(episodes, np.array([np.nanmean(max_series[max(0, i - 19):i+1]) for i in range(len(max_series))]),
                color='darkred', linewidth=3.0, label='Rolling Avg (20)')
    ax.axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='10mm target')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Max Waypoint Miss (mm)')
    ax.set_title('Worst Waypoint Per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes, joint_errors, alpha=0.3, color='tab:brown', linewidth=1.5)
    ax.plot(episodes, rolling_avg(joint_errors), color='saddlebrown', linewidth=3.0, label='Rolling Avg (20)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Joint Error (deg)')
    ax.set_title('Joint Tracking Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, durations, alpha=0.3, color='tab:gray', linewidth=1.5)
    ax.plot(episodes, rolling_avg(durations), color='black', linewidth=3.0, label='Rolling Avg (20)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration (s)')
    ax.set_title('Episode Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ok_avg = [(v is not None and v <= 5.0) for v in avg_wp_mm]
    ok_max = [(v is not None and v <= 10.0) for v in max_wp_mm]
    ax.plot(episodes, rolling_rate(ok_avg, window=50), color='green', linewidth=2.5, label='P(AvgWP<=5mm) roll50')
    ax.plot(episodes, rolling_rate(ok_max, window=50), color='orange', linewidth=2.5, label='P(MaxWP<=10mm) roll50')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('Quality Success Rates')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    valid_wp_rows = [row for row in waypoint_errors if row]
    if valid_wp_rows:
        max_len = max(len(row) for row in valid_wp_rows)
        mat = np.full((len(valid_wp_rows), max_len), np.nan, dtype=np.float64)
        for row_idx, row in enumerate(valid_wp_rows):
            arr = np.asarray(row, dtype=np.float64)
            mat[row_idx, : min(max_len, len(arr))] = arr[:max_len]
        mean_wp = np.nanmean(mat, axis=0)
        p90_wp = np.nanpercentile(mat, 90, axis=0)
        x = np.arange(1, len(mean_wp) + 1)
        ax.plot(x, mean_wp, color='teal', linewidth=2.5, label='Mean WP error')
        ax.plot(x, p90_wp, color='darkcyan', linewidth=2.5, label='P90 WP error')
        ax.axhline(y=5.0, color='green', linestyle='--', alpha=0.4, label='5mm')
        ax.axhline(y=10.0, color='orange', linestyle='--', alpha=0.4, label='10mm')
        ax.set_xlabel('Waypoint Index')
        ax.set_ylabel('Error (mm)')
        ax.set_title('Waypoint Error Profile')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.05, 0.5, 'No waypoint profile', transform=ax.transAxes)
        ax.set_title('Waypoint Error Profile')

    ax = axes[2, 1]
    paired = [(avg_wp_mm[i], joint_errors[i]) for i in range(len(avg_wp_mm)) if avg_wp_mm[i] is not None]
    if paired:
        avg_vals = np.array([item[0] for item in paired], dtype=np.float64)
        joint_vals = np.array([item[1] for item in paired], dtype=np.float64)
        ax.scatter(avg_vals, joint_vals, alpha=0.7, color='tab:blue')
        ax.set_xlabel('Avg Waypoint Miss (mm)')
        ax.set_ylabel('Mean Joint Error (deg)')
        ax.set_title('Accuracy vs Joint Error')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.05, 0.5, 'No paired drawing metrics', transform=ax.transAxes)
        ax.set_title('Accuracy vs Joint Error')

    ax = axes[2, 2]
    ax.axis('off')
    summary_lines = [
        'Deploy Summary (Drawing)',
        '------------------------',
        '',
        f'Episodes: {len(endpoint_mm)}',
        f'Replay Rate: {replay_rate_hz:.1f} Hz',
        '',
        f'Avg Endpoint Miss: {np.mean(endpoint_mm):.1f} mm',
        f'Best Endpoint Miss: {np.min(endpoint_mm):.1f} mm',
        f'Avg Joint Error: {np.mean(joint_errors):.2f} deg',
        f'Avg Duration: {np.mean(durations):.2f} s',
    ]
    if valid_avg_wp:
        summary_lines += [
            '',
            f'Avg WP Miss: {np.mean(valid_avg_wp):.1f} mm',
            f'Best Avg WP: {np.min(valid_avg_wp):.1f} mm',
            f'Avg Max WP: {np.mean(valid_max_wp):.1f} mm',
            f'Best Max WP: {np.min(valid_max_wp):.1f} mm',
        ]
    ax.text(
        0.1,
        0.5,
        '\n'.join(summary_lines),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
    )

    plt.tight_layout()
    plot_path = os.path.join(png_dir, f'deploy_tuning_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Deploy tuning plot saved: {plot_path}")


def main():
    """Main entry point with interactive menu or direct CLI"""
    parser = argparse.ArgumentParser(description='Train RL agent for 6-DOF robot arm')
    parser.add_argument('--agent', type=str, default=None, choices=['sac'],
                        help='RL agent to use: sac (skips menu if provided)')
    parser.add_argument('--episodes', type=int, default=None,
                        help=f'Number of training episodes (default: {NUM_EPISODES})')
    parser.add_argument('--max-steps', type=int, default=None,
                        help=f'Max steps per episode (default: {MAX_STEPS_PER_EPISODE})')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint to load (optional)')
    parser.add_argument('--manual', action='store_true',
                        help='Start in manual test mode (skips menu)')
    parser.add_argument('--control-backend', type=str, default=None,
                        choices=sorted(SUPPORTED_CONTROL_BACKENDS),
                        help='Motion backend: sim, sim_to_real_shadow, or real_replay')
    
    # Direct CLI options
    parser.add_argument('--option', '-o', type=str, default=None,
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                        help='Direct menu option choice (1-9)')
    parser.add_argument('--sub-option', '-s', type=str, default=None,
                        help='Sub-option choice (e.g., a, b, c)')
    parser.add_argument('--pid-backend', type=str, default=None,
                        choices=['sim', 'sim_to_real_shadow', 'real_replay', 'pi_direct'],
                        help='PID tuning control backend')
    parser.add_argument('--require-board', type=str, default=None,
                        choices=['y', 'n', 'yes', 'no'],
                        help='Require live board detection (y/n)')
    parser.add_argument('--replay-artifact', type=str, default=None,
                        help='Path to replay artifact .pkl')
    parser.add_argument('--replay-gains', type=str, default=None,
                        help='Path to replay gains')
    parser.add_argument('--segment-steps', type=int, default=None,
                        help='Drawing segment steps (default: 20)')
    parser.add_argument('--mirror-rate', type=float, default=None,
                        help='Mirror publish rate Hz (default: 10)')
    parser.add_argument('--mirror-deadband', type=float, default=None,
                        help='Mirror deadband degrees (default: 0.5)')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of FK samples for training Neural IK Model')

    args = parser.parse_args()

    # If manual mode flag is set or option 1 is selected
    if args.manual or args.option == '1':
        backend = args.control_backend if args.control_backend else 'sim'
        manual_control_mode(control_backend=backend)
        return

    # If agent is specified via command line (legacy behavior)
    if args.agent is not None and args.option is None:
        if args.episodes is None:
            args.episodes = NUM_EPISODES
        if args.max_steps is None:
            args.max_steps = MAX_STEPS_PER_EPISODE
        train(args)
        return

    # Determine choice: from CLI option or interactive menu
    if args.option is not None:
        choice = args.option
    else:
        choice = show_menu()

    if choice == '1':
        # Run inline manual test mode
        backend = args.control_backend if args.control_backend else 'sim'
        manual_control_mode(control_backend=backend)
        return  # Exit after manual mode
    elif choice == '2':
        args.agent = 'sac'
        if args.episodes is not None and args.max_steps is not None:
            episodes, max_steps = args.episodes, args.max_steps
        else:
            episodes, max_steps = get_training_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train(args)
    elif choice == '3':
        args.agent = 'sac'
        args.use_neural_ik = True
        if args.episodes is not None and args.max_steps is not None:
            episodes, max_steps = args.episodes, args.max_steps
        else:
            episodes, max_steps = get_training_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train(args)
    elif choice == '4':
        # Train Neural IK model
        print("\n" + "="*70)
        print("🧠 Training Neural IK Model")
        print("="*70)

        # Ask for number of samples
        if args.n_samples is not None:
            n_samples = args.n_samples
        else:
            try:
                n_samples_input = input("Number of FK samples (default 500000): ").strip()
                if n_samples_input == '':
                    n_samples = 500000
                else:
                    n_samples = int(n_samples_input)
            except ValueError:
                print("Invalid input, using default 500000")
                n_samples = 500000

        nik = NeuralIK()
        positions, joints = nik.generate_training_data(n_samples=n_samples)
        nik.train(positions, joints, epochs=100)
        save_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'neural_ik.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nik.save(save_path)
        print("\n✅ Neural IK training complete! Now you can use options 4 or 5.")
    elif choice == '5':
        # Drawing Training (SAC) - 6D Direct
        print("\n🖋️ Drawing Training (SAC 6D Direct)")
        args.agent = 'sac'
        args.use_neural_ik = False
        args.drawing_mode = True
        if args.episodes is not None and args.max_steps is not None:
            episodes, max_steps = args.episodes, args.max_steps
        else:
            episodes, max_steps = get_drawing_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train_drawing(args)
    elif choice == '6':
        # Drawing Training (SAC + Neural IK) - 3D Position
        print("\n🖋️ Drawing Training (SAC + Neural IK 3D)")
        args.agent = 'sac'
        args.use_neural_ik = True
        args.drawing_mode = True
        if args.episodes is not None and args.max_steps is not None:
            episodes, max_steps = args.episodes, args.max_steps
        else:
            episodes, max_steps = get_drawing_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train_drawing(args)
    elif choice == '7':
        # PID Tuning (RL-Optimized PID Gains) — Sub-menu
        print("\n🎛️ PID Tuning Mode:")
        if args.sub_option is not None:
            sub = args.sub_option.lower()
        else:
            print("  a. 📍 Reaching (Random joint targets)")
            print("  b. 🖋️  Drawing (Shape waypoints)")
            sub = input("Select (a/b, default=a): ").strip().lower()
        
        mode = 'drawing' if sub == 'b' else 'reaching'

        if args.pid_backend is not None:
            try:
                control_backend = resolve_control_backend(args.pid_backend)
            except Exception:
                control_backend = 'sim'
        else:
            control_backend = prompt_pid_backend()

        if args.require_board is not None:
            require_board_detection = args.require_board.lower() in ('y', 'yes')
        else:
            require_board_detection = input(
                "Require live board detection? (y/N): "
            ).strip().lower() == 'y'

        replay_artifact_path = None
        replay_gains_path = None
        if control_backend == 'real_replay':
            if args.replay_artifact is not None or args.replay_gains is not None:
                replay_artifact_path = args.replay_artifact
                replay_gains_path = args.replay_gains
            else:
                replay_artifact_path, replay_gains_path = prompt_pid_replay_paths(mode)

        segment_steps = 20
        if args.segment_steps is not None:
            segment_steps = args.segment_steps
        elif mode == 'drawing':
            steps_input = input("Drawing segment steps (default 20, lower = faster): ").strip()
            if steps_input:
                try:
                    segment_steps = int(steps_input)
                except ValueError:
                    pass

        train_pid_tuning(
            mode=mode,
            control_backend=control_backend,
            require_board_detection=require_board_detection,
            replay_artifact_path=replay_artifact_path,
            replay_gains_path=replay_gains_path,
            segment_steps=segment_steps,
        )
    elif choice == '8':
        # 🚀 Digital Twin Realtime Mirror
        print("\n🚀 Digital Twin Realtime Mirror:")
        if args.mirror_rate is not None:
            rate_hz = args.mirror_rate
        else:
            rate_input = input("Mirror publish rate Hz (default 10): ").strip()
            try:
                rate_hz = float(rate_input) if rate_input else 10.0
            except ValueError:
                rate_hz = 10.0

        if args.mirror_deadband is not None:
            deadband = args.mirror_deadband
        else:
            deadband_input = input("Mirror deadband degrees (default 0.5): ").strip()
            try:
                deadband = float(deadband_input) if deadband_input else 0.5
            except ValueError:
                deadband = 0.5

        if args.sub_option is not None:
            sub = args.sub_option.lower()
        else:
            print("\nSelect training mode to run WITH mirror:")
            print("  a. 📍 PID Reaching")
            print("  b. 🖋️  PID Drawing      ← khuyến nghị test đầu tiên")
            print("  c. 🎮 Manual Control")
            sub = input("Select (a/b/c, default=b): ").strip().lower()
            if not sub:
                sub = 'b'

        mode = 'drawing' if sub == 'b' else 'reaching'

        # Initialize defaults for train_pid_tuning parameters
        require_board_detection = False
        segment_steps = 20

        if sub in ('a', 'b'):
            if args.require_board is not None:
                require_board_detection = args.require_board.lower() in ('y', 'yes')
            else:
                require_board_detection = input(
                    "Require live board detection? (y/N): "
                ).strip().lower() == 'y'

            if sub == 'b':
                if args.segment_steps is not None:
                    segment_steps = args.segment_steps
                else:
                    steps_input = input("Drawing segment steps (default 20, lower = faster): ").strip()
                    if steps_input:
                        try:
                            segment_steps = int(steps_input)
                        except ValueError:
                            pass

        # Resolve path to sim_to_pi_mirror.py
        import subprocess
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        mirror_script = os.path.join(scripts_dir, "digital_twin", "sim_to_pi_mirror.py")

        print(f"\n🔄 Starting sim-to-real mirror in background ({rate_hz} Hz, deadband {deadband}°)...")
        mirror_proc = None
        try:
            mirror_proc = subprocess.Popen(
                [sys.executable, mirror_script, "--rate-hz", str(rate_hz), "--deadband-deg", str(deadband)]
            )
            time.sleep(1.0) # Let it print starting logs

            if sub == 'c':
                manual_control_mode(control_backend='sim')
            else:
                train_pid_tuning(
                    mode=mode,
                    control_backend='sim',
                    require_board_detection=require_board_detection,
                    replay_artifact_path=None,
                    replay_gains_path=None,
                    segment_steps=segment_steps,
                )
        finally:
            if mirror_proc is not None:
                print("\n🛑 Stopping sim-to-real mirror...")
                mirror_proc.terminate()
                try:
                    mirror_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    mirror_proc.kill()
                    mirror_proc.wait()
                print("✅ Mirror stopped.")
    elif choice == '9':
        # 🤖 Pi-Only Training (no Gazebo)
        print("\n🤖 Pi-Only Training (no Gazebo):")
        print("   Laptop giao tiếp trực tiếp với Pi qua ROS 2 DDS.")
        print("   Pi phải đang chạy wicom_roboarm node trên cùng mạng.")
        print("   End-effector position được tính bằng FK (không cần TF2/Gazebo).")
        print()
        if args.sub_option is not None:
            sub = args.sub_option.lower()
        else:
            print("  a. 📍 PID Reaching (Random joint targets)")
            print("  b. 🖋️  PID Drawing (Shape waypoints) ← khuyến nghị")
            print("  c. 🎮 Manual Control")
            sub = input("Select (a/b/c, default=b): ").strip().lower()
            if not sub:
                sub = 'b'

        mode = 'drawing' if sub == 'b' else 'reaching'
        segment_steps = 20

        if sub in ('a', 'b'):
            if sub == 'b':
                if args.segment_steps is not None:
                    segment_steps = args.segment_steps
                else:
                    steps_input = input("Drawing segment steps (default 20, lower = faster): ").strip()
                    if steps_input:
                        try:
                            segment_steps = int(steps_input)
                        except ValueError:
                            pass

            train_pid_tuning(
                mode=mode,
                control_backend='pi_direct',
                require_board_detection=False,
                replay_artifact_path=None,
                replay_gains_path=None,
                segment_steps=segment_steps,
            )
        elif sub == 'c':
            manual_control_mode(control_backend='pi_direct')
        else:
            print("❌ Invalid sub-option")
    else:
        print("❌ Invalid choice! Exiting...")


if __name__ == '__main__':
    main()
