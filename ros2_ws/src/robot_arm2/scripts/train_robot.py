#!/usr/bin/env python3
"""
Main RL Training Script for 6-DOF Robot Arm
Trains TD3+HER agent to reach target positions on drawing surface

Usage:
    python3 train_robot.py --episodes 500 --max-steps 10
"""

import rclpy
import numpy as np
import argparse
import time
import os
from datetime import datetime

# Import RL components
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl.rl_environment import RLEnvironment
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgentGazebo
from utils.her import her_augmentation
from rl.neural_ik import NeuralIK
from rl.pid_controller import PIDController

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Episode settings
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 10
LEARNING_STARTS = 10  # Start training after this many episodes

# Training settings (GitHub style)
OPT_STEPS_PER_EPISODE = 64  # GitHub: 64 gradient updates per episode
SAVE_INTERVAL = 25  # Save models every N episodes
EVAL_INTERVAL = 10  # Evaluate (without noise) every N episodes
MIN_EPISODES = 25  # Minimum episodes before allowing 'best' model save

# HER settings
HER_ENABLED = True
HER_K = 4  # Number of HER samples per timestep
HER_STRATEGY = 'future'  # 'future' or 'final'

# Reward settings (SPARSE like GitHub project)
# 0 for success, -1 for failure - let HER do the work
GOAL_THRESHOLD = 0.0075  # 0.75cm threshold for success (tighter)
SUCCESS_REWARD = 0.0   # Sparse: 0 for success
STEP_PENALTY = -1.0    # Sparse: -1 for failure

# TD3/SAC hyperparameters (GitHub style - higher LRs)
ACTOR_LR = 0.001       # GitHub: 0.001 (was 3e-4)
CRITIC_LR = 0.002      # GitHub: 0.002 (was 1e-4)
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)
BATCH_OPT_STEPS = 64   # GitHub: 64 optimization steps per episode

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


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    """Main training function"""
    print("="*70)
    print("TD3+HER Training for 6-DOF Robot Arm")
    print("="*70)
    
    env = None  # Initialize env to prevent unbound error in finally block
    ros_initialized = False
    
    try:
        # Initialize ROS2
        rclpy.init()
        ros_initialized = True
        
        # Create environment
        print("\n📦 Creating RL environment...")
        env = RLEnvironment(
            max_episode_steps=args.max_steps,
            goal_tolerance=GOAL_THRESHOLD
        )
        
        # Wait for environment to initialize
        print("   Waiting for environment...")
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)
        
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
        
        # PID Controller DISABLED - caused worse results (4.5% vs 26% success rate)
        # The PID was making the pen swing horizontally instead of reaching target
        args.pid_controller = None
        # if use_neural_ik:
        #     args.pid_controller = PIDController(Kp=2.0, Ki=0.0, Kd=0.1)
        #     print(f"   ✅ PID Controller enabled for Residual RL (Kp=2.0, Kd=0.1)")
        # else:
        #     args.pid_controller = None
        
        if args.agent == 'td3':
            agent = TD3Agent(
                state_dim=16,  # 16D observation
                action_dim=action_dim,
                max_action=max_action,
                min_action=min_action,
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                gamma=GAMMA,
                tau=TAU,
                batch_size=BATCH_SIZE,
                buffer_size=BUFFER_SIZE
            )
            mode_str = "Neural IK 3D" if use_neural_ik else "Direct 6D"
            print(f"TD3 Agent initialized ({mode_str} Control):")
            print(f"  State dim: 16, Action dim: {action_dim}")
            print(f"  Device: {agent.device}")
            print(f"  Buffer size: {BUFFER_SIZE}, Batch size: {BATCH_SIZE}")
        
        elif args.agent == 'sac':
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
            raise ValueError(f"Unknown agent: {args.agent}. Choose 'td3' or 'sac'")
        
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
                    # NARROWED TASK WORKSPACE (12x15x12 cm)
                    TASK_POS_MIN = np.array([-0.06, -0.30, 0.16])  # X±6, Y-30to-15, Z16to28
                    TASK_POS_MAX = np.array([0.06, -0.15, 0.28])   # 12x15x12 cm
                    
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
        
        # Training complete - comprehensive summary
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETED!")
        print("="*70)
        
        # Overall statistics
        overall_avg_reward = np.mean(episode_rewards)
        overall_success_rate = np.mean(episode_successes)
        overall_avg_min_dist = np.mean(episode_min_distances)
        best_min_dist = min(episode_min_distances)
        
        print(f"\n📊 Overall Statistics ({args.episodes} episodes):")
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
        # Create mode suffix for filenames (e.g., 'sac_neuralIK' or 'td3_direct')
        mode_suffix = f"{args.agent}{'_neuralIK' if use_neural_ik else '_direct'}"
        plot_training_stats(episode_rewards, episode_successes, episode_min_distances, 
                           actor_losses, critic_losses, png_dir, csv_dir, timestamp, mode_suffix)
        
        # Save final model
        agent.save_models()
        agent.replay_buffer.save(f'{pkl_dir}/replay_buffer_final_{mode_suffix}_{timestamp}.pkl')
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
        results_file = f'{pkl_dir}/training_results_{mode_suffix}_{timestamp}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(training_results, f)
        print(f"💾 Training results saved to: {results_file}")
        print(f"   Total episodes: {len(episode_rewards)}")
        
        # Final cleanup - mode-specific, keep only best and final buffers
        # Clean only THIS mode's buffers (4 periodic, 1 best, 1 final)
        cleanup_old_files(pkl_dir, f"replay_buffer_ep*{mode_suffix}*.pkl", 4)  # Keep 4 periodic
        cleanup_old_files(pkl_dir, f"replay_buffer_best*{mode_suffix}*.pkl", 1)  # Keep only 1 best
        cleanup_old_files(pkl_dir, f"replay_buffer_final*{mode_suffix}*.pkl", 1)  # Keep only 1 final
        cleanup_old_files(pkl_dir, f"training_results*{mode_suffix}*.pkl", 3)  # Keep 3 most recent results
        print(f"🧹 Cleaned up old {mode_suffix} buffer files")
        
        print(f"\n✅ Training complete! Trained for {args.episodes} episodes.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
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


def plot_training_stats(episode_rewards, episode_successes, episode_min_distances, actor_losses, critic_losses, png_dir, csv_dir, timestamp, mode_suffix=''):
    """Plot training statistics with cumulative moving averages including distance"""
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
    
    # Create figure with 3x2 subplots
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
    
    # Plot 2: Success Rate (top-center)
    ax = axes[0, 1]
    success_pct = np.array(episode_successes) * 100
    success_avg_pct = np.array(success_avg) * 100
    ax.plot(episodes, success_pct, alpha=0.3, color='green', linewidth=1.5, label='Episode Success')
    ax.plot(episodes, success_avg_pct, color='darkgreen', linewidth=3.0, label='Cumulative Average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Min Distance to Target (top-right) - NEW!
    ax = axes[0, 2]
    ax.plot(episodes, distances_cm, alpha=0.3, color='orange', linewidth=1.5, label='Episode Min Distance')
    ax.plot(episodes, distance_avg_cm, color='darkorange', linewidth=3.0, label='Cumulative Average')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Goal (1cm)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Distance (cm)', fontsize=12)
    ax.set_title('Min Distance to Target', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Actor Loss (bottom-left)
    ax = axes[1, 0]
    valid_actor = [(i+1, l) for i, l in enumerate(actor_losses) if l is not None]
    if valid_actor:
        actor_eps, actor_vals = zip(*valid_actor)
        ax.plot(actor_eps, actor_vals, color='red', linewidth=2.5, label='Actor Loss')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Actor Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center', fontsize=12)
        ax.set_title('Actor Loss', fontsize=14, fontweight='bold')
    
    # Plot 5: Critic Loss (bottom-center)
    ax = axes[1, 1]
    valid_critic = [(i+1, l) for i, l in enumerate(critic_losses) if l is not None]
    if valid_critic:
        critic_eps, critic_vals = zip(*valid_critic)
        ax.plot(critic_eps, critic_vals, color='purple', linewidth=2.5, label='Critic Loss')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Critic Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center', fontsize=12)
        ax.set_title('Critic Loss', fontsize=14, fontweight='bold')
    
    # Plot 6: Combined Summary (bottom-right)
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
📊 Training Summary
━━━━━━━━━━━━━━━━━━━━

Episodes: {len(episode_rewards)}

Rewards:
  • Final Avg: {reward_avg[-1]:.2f}
  • Best: {max(episode_rewards):.2f}

Success Rate:
  • Final: {success_avg_pct[-1]:.1f}%

Distance to Target:
  • Final Avg: {distance_avg_cm[-1]:.2f}cm
  • Best: {min(distances_cm):.2f}cm
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


def manual_control_mode():
    """
    Manual control mode - enter joint angles to move robot.
    Uses the RL environment for robot communication.
    """
    print("\n" + "=" * 70)
    print("🎮 MANUAL CONTROL MODE")
    print("=" * 70)
    print("Commands:")
    print("  Enter 6 joint angles in DEGREES: e.g., '0 0 45 0 0 0'")
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
        env = RLEnvironment(max_episode_steps=100, goal_tolerance=0.01)
        
        # Wait for initialization
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)
        
        print("✅ Environment ready!")
        
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
                    print(f"📍 EE position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
                
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
                        print(f"📊 FK Position: ({fk_pos[0]:.4f}, {fk_pos[1]:.4f}, {fk_pos[2]:.4f})")
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
                
                # Convert to radians and clip to limits
                joints_rad = np.radians(joints_deg)
                joints_rad = np.clip(joints_rad, -np.pi/2, np.pi/2)
                
                # Show target FK
                try:
                    target_fk = fk(joints_rad)
                    print(f"🎯 Target FK: ({target_fk[0]:.4f}, {target_fk[1]:.4f}, {target_fk[2]:.4f})")
                except Exception as e:
                    print(f"⚠️ FK error: {e}")
                
                # Execute movement
                print(f"🚀 Moving to: {[f'{d:.1f}°' for d in joints_deg]}")
                next_state, reward, done, info = env.step(joints_rad)
                
                # Spin to process
                for _ in range(10):
                    rclpy.spin_once(env, timeout_sec=0.1)
                
                time.sleep(0.5)
                
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
    print("1. Manual Test Mode")
    print("2. RL Training Mode (TD3) - 6D Direct Joint Control")
    print("3. RL Training Mode (SAC) - 6D Direct Joint Control")
    print("-"*70)
    print("4. RL Training Mode (TD3 + Neural IK) - 3D Position Control ⚡")
    print("5. RL Training Mode (SAC + Neural IK) - 3D Position Control ⚡")
    print("-"*70)
    print("6. Train Neural IK Model (run first for options 4-5)")
    print("-"*70)
    print("🖋️  DRAWING TRAINING (30 waypoints, continuous trajectory)")
    print("7. Drawing Training (SAC) - 6D Direct")
    print("8. Drawing Training (SAC + Neural IK) - 3D Position ⚡")
    print("="*70)
    
    choice = input("Select option (1-8): ").strip()
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
    print("  Triangle: 30 waypoints (10 per edge)")
    print("  Each step = 1 attempt to reach current waypoint")
    print("  When waypoint reached → next waypoint becomes target")
    print("  Episode ends: all 30 reached OR max steps exceeded")
    print("-"*70)
    print("  State: 18D = 6 joints + 3 EE + 3 target + 3 dist + 3 other")
    print("="*70)
    
    # Episodes (default higher for drawing)
    episodes_input = input("Number of episodes (default 100): ").strip()
    episodes = int(episodes_input) if episodes_input else 100
    
    # Max steps = ideally 1 per waypoint, but allow some buffer
    # 30 waypoints = 30 steps minimum, 50 = safer buffer
    steps_input = input("Max steps per episode (default 50, min 30 for 30 waypoints): ").strip()
    max_steps = int(steps_input) if steps_input else 50
    max_steps = max(30, max_steps)  # Enforce minimum
    
    print(f"\n✅ Drawing Configuration:")
    print(f"   Episodes: {episodes}")
    print(f"   Max steps: {max_steps} (30 waypoints need ≥30 steps)")
    print(f"   State dim: 18")
    print("="*70)
    
    return episodes, max_steps


def train_drawing(args):
    """
    Training loop for drawing task using DrawingEnvironment.
    Uses dense_triangle with 30 waypoints for continuous trajectory.
    """
    print("="*70)
    print("🖋️ Drawing Training - Continuous Triangle Trajectory")
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
        env = DrawingEnvironment(
            max_episode_steps=args.max_steps,
            waypoint_tolerance=0.01,  # 1cm tolerance
            shape_type='dense_triangle',  # Will need to update DrawingEnvironment
            shape_size=0.15,  # 15cm triangle
            y_plane=0.20
        )
        
        # Wait for environment
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)
        
        print("✅ Drawing Environment ready!")
        print(f"   Shape: dense_triangle (30 waypoints)")
        print(f"   Tolerance: ±1cm")
        
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
        
        # Training loop
        print(f"\n🚀 Starting drawing training ({args.episodes} episodes)...\n")
        
        episode_rewards = []
        waypoints_completed = []
        
        for episode in range(args.episodes):
            state = env.reset_environment()
            
            for _ in range(10):
                rclpy.spin_once(env, timeout_sec=0.1)
            
            if state is None:
                print(f"Episode {episode+1}: Failed to reset")
                continue
            
            episode_reward = 0.0
            min_distance = float('inf')
            
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
                    STEP_SIZE = 0.05  # 5cm max step
                    
                    # Compute direction to waypoint
                    direction = waypoint - ee_pos_before
                    distance = np.linalg.norm(direction)
                    
                    if distance > 0.001:  # Avoid division by zero
                        direction_norm = direction / distance
                        # Action scales how much we move in that direction
                        # action[0] = forward/back, action[1-2] = fine adjustment
                        move_amount = (action[0] + 1) / 2 * STEP_SIZE  # 0 to 5cm
                        fine_adjust = action[1:3] * 0.02  # ±2cm lateral
                        
                        # Target = EE + movement toward waypoint + fine adjustment
                        delta = direction_norm * move_amount
                        target_xyz = ee_pos_before + delta
                        target_xyz[0] += fine_adjust[0]  # X adjustment
                        target_xyz[2] += fine_adjust[1]  # Z adjustment
                    else:
                        target_xyz = waypoint  # Already at waypoint
                    
                    # Clamp to safe bounds
                    target_xyz = np.clip(target_xyz, 
                                         [-0.15, 0.10, 0.05],
                                         [0.15, 0.30, 0.45])
                    
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
                
                min_distance = min(min_distance, dist_after)
                
                if wp_reached > wp_reached_before:
                    print(f"  ✅ WAYPOINT {wp_idx} REACHED!")
                    wp_reached_before = wp_reached
                
                if info.get('shape_complete', False):
                    print(f"  🎨🎨🎨 SHAPE COMPLETE! 🎨🎨🎨")
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            wp_reached = info.get('waypoints_reached', 0)
            waypoints_completed.append(wp_reached)
            
            # Train agent
            if episode >= 5:
                for _ in range(20):
                    agent.train()
            
            # Log
            shape_complete = info.get('shape_complete', False)
            status = "🎨 COMPLETE!" if shape_complete else f"WP: {wp_reached}/30"
            print(f"Episode {episode+1}/{args.episodes} | "
                  f"Reward: {episode_reward:.1f} | {status}")
            
            # Save best
            if shape_complete or (episode > 10 and wp_reached >= max(waypoints_completed)):
                agent.save_models()
        
        print("\n" + "="*70)
        print("🎉 Drawing training complete!")
        print(f"   Best waypoints: {max(waypoints_completed)}/30")
        print("="*70)
        
        agent.save_models()
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
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


def main():
    """Main entry point with interactive menu"""
    parser = argparse.ArgumentParser(description='Train RL agent for 6-DOF robot arm')
    parser.add_argument('--agent', type=str, default=None, choices=['td3', 'sac'],
                        help='RL agent to use: td3 or sac (skips menu if provided)')
    parser.add_argument('--episodes', type=int, default=None,
                        help=f'Number of training episodes (default: {NUM_EPISODES})')
    parser.add_argument('--max-steps', type=int, default=None,
                        help=f'Max steps per episode (default: {MAX_STEPS_PER_EPISODE})')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint to load (optional)')
    parser.add_argument('--manual', action='store_true',
                        help='Start in manual test mode (skips menu)')
    
    args = parser.parse_args()
    
    # If manual mode flag is set
    if args.manual:
        manual_test_mode()
        return
    
    # If agent is specified via command line, skip menu
    if args.agent is not None:
        # Use command-line values or defaults
        if args.episodes is None:
            args.episodes = NUM_EPISODES
        if args.max_steps is None:
            args.max_steps = MAX_STEPS_PER_EPISODE
        train(args)
        return
    
    # Show interactive menu
    choice = show_menu()
    
    if choice == '1':
        # Run inline manual test mode
        manual_control_mode()
        return  # Exit after manual mode
    elif choice == '2':
        args.agent = 'td3'
        # Get training parameters interactively
        episodes, max_steps = get_training_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train(args)
    elif choice == '3':
        args.agent = 'sac'
        # Get training parameters interactively
        episodes, max_steps = get_training_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train(args)
    elif choice == '4':
        args.agent = 'td3'
        args.use_neural_ik = True
        # Get training parameters interactively
        episodes, max_steps = get_training_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train(args)
    elif choice == '5':
        args.agent = 'sac'
        args.use_neural_ik = True
        # Get training parameters interactively
        episodes, max_steps = get_training_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train(args)
    elif choice == '6':
        # Train Neural IK model
        print("\n" + "="*70)
        print("🧠 Training Neural IK Model")
        print("="*70)
        
        # Ask for number of samples
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
    elif choice == '7':
        # Drawing Training (SAC) - 6D Direct
        print("\n🖋️ Drawing Training (SAC 6D Direct)")
        args.agent = 'sac'
        args.use_neural_ik = False
        args.drawing_mode = True
        episodes, max_steps = get_drawing_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train_drawing(args)
    elif choice == '8':
        # Drawing Training (SAC + Neural IK) - 3D Position
        print("\n🖋️ Drawing Training (SAC + Neural IK 3D)")
        args.agent = 'sac'
        args.use_neural_ik = True
        args.drawing_mode = True
        episodes, max_steps = get_drawing_params()
        args.episodes = episodes
        args.max_steps = max_steps
        train_drawing(args)
    else:
        print("❌ Invalid choice! Exiting...")


if __name__ == '__main__':
    main()
