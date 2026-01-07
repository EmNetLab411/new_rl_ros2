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

# Training settings
OPT_STEPS_PER_EPISODE = 40  # Gradient updates per episode
SAVE_INTERVAL = 25  # Save models every N episodes
EVAL_INTERVAL = 10  # Evaluate (without noise) every N episodes
MIN_EPISODES = 25  # Minimum episodes before allowing 'best' model save

# HER settings
HER_ENABLED = True
HER_K = 4  # Number of HER samples per timestep
HER_STRATEGY = 'future'  # 'future' or 'final'

# Reward settings
GOAL_THRESHOLD = 0.01  # 7.5mm threshold for success
SUCCESS_REWARD = 10.0
STEP_PENALTY = -0.1

# TD3 hyperparameters
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)

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


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    """Main training function"""
    print("="*70)
    print("TD3+HER Training for 6-DOF Robot Arm")
    print("="*70)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
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
        
        # Agent configuration for ABSOLUTE JOINT CONTROL
        # Action: 6D ABSOLUTE joint angles (±π/2 rad = ±90°)
        # Agent outputs target joint positions, not deltas
        # State: 18D observation from environment
        JOINT_LIMIT = np.pi / 2  # ±90° = ±1.57 rad
        
        if args.agent == 'td3':
            agent = TD3Agent(
                state_dim=16,  # 16D observation (no key_velocities)
                action_dim=6,  # 6 absolute joint angles
                max_action=np.array([JOINT_LIMIT] * 6),   # +90°
                min_action=np.array([-JOINT_LIMIT] * 6),  # -90°
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                gamma=GAMMA,
                tau=TAU,
                batch_size=BATCH_SIZE,
                buffer_size=BUFFER_SIZE
            )
            print(f"TD3 Agent initialized (Absolute Joint Control):")
            print(f"  State dim: 16, Action dim: 6 (absolute joint angles)")
            print(f"  Joint range: ±{np.degrees(JOINT_LIMIT):.0f}° per joint")
            print(f"  Device: {agent.device}")
            print(f"  Buffer size: {BUFFER_SIZE}, Batch size: {BATCH_SIZE}")
        
        elif args.agent == 'sac':
            agent = SACAgentGazebo(
                state_dim=16,  # 16D observation (no key_velocities)
                n_actions=6,   # 6 absolute joint angles
                max_action=np.array([JOINT_LIMIT] * 6),   # +90°
                min_action=np.array([-JOINT_LIMIT] * 6),  # -90°
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                gamma=GAMMA,
                tau=TAU,
                batch_size=BATCH_SIZE,
                buffer_size=BUFFER_SIZE,
                auto_entropy_tuning=True
            )
            print(f"SAC Agent initialized (Absolute Joint Control):")
            print(f"  State dim: 16, Action dim: 6 (absolute joint angles)")
            print(f"  Joint range: ±{np.degrees(JOINT_LIMIT):.0f}° per joint")
        
        else:
            raise ValueError(f"Unknown agent: {args.agent}. Choose 'td3' or 'sac'")
        
        # Ask to load existing replay buffer
        load_buffer = input("\n📦 Load existing replay buffer? (y/n): ").strip().lower()
        if load_buffer == 'y':
            # Find available buffers - prioritize BEST over FINAL
            import glob
            best_buffers = sorted(glob.glob("training_results/pkl/*best*.pkl"), key=os.path.getmtime, reverse=True)
            final_buffers = sorted(glob.glob("training_results/pkl/*final*.pkl"), key=os.path.getmtime, reverse=True)
            
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
        
        # Automatically try to load pre-trained models (like old code)
        # This allows continuing training from previous checkpoint
        checkpoint_dir = f"checkpoints/{args.agent}_gazebo"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        actor_path = os.path.join(checkpoint_dir, f'actor_{args.agent}_best.pth')
        critic_path = os.path.join(checkpoint_dir, f'critic_{args.agent}_best.pth')
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            try:
                agent.load_models(actor_path, critic_path)
                print(f"\n✅ Loaded pre-trained models from: {checkpoint_dir}")
                print(f"   Actor: {os.path.basename(actor_path)}")
                print(f"   Critic: {os.path.basename(critic_path)}")
            except Exception as e:
                print(f"\n⚠️  Failed to load models: {e}")
                print("   Starting with untrained agent")
        else:
            print(f"\n📝 No pre-trained models found in {checkpoint_dir}/")
            print("   Starting with untrained agent")
        
        # Training statistics
        episode_rewards = []
        episode_successes = []
        episode_min_distances = []  # Track min distance to target each episode
        best_avg_reward = -float('inf')
        actor_losses = []
        critic_losses = []
        
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
        
        # Training loop
        print("\n🚀 Starting training...\n")
        
        for episode in range(args.episodes):
            episode_start = time.time()
            
            # Reset environment
            state = env.reset_environment()
            
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
                
                # Execute action
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
                    
                    if done and reward > 5.0:
                        print(f"  🎉🎉🎉 SUCCESS! Goal reached! 🎉🎉🎉")
                
                if next_state is None:
                    print(f"   Step {step+1}: State unavailable, skipping")
                    break
                
                # Store transition
                goal = state[9:12]  # Target position from state
                episode_buffer.append((state, action, reward, next_state, done, goal))
                
                episode_reward += reward
                
                # Check success
                if done and reward > 5.0:  # Goal reached
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
            
            # Save best model
            if episode >= MIN_EPISODES and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_models()
                agent.replay_buffer.save(f'{pkl_dir}/replay_buffer_best_{timestamp}.pkl')
                print(f"   💾 New best model saved! Avg reward: {best_avg_reward:.2f}")
            
            # Periodic saves
            if (episode + 1) % SAVE_INTERVAL == 0:
                agent.save_models(episode=episode+1)
                agent.replay_buffer.save(f'{pkl_dir}/replay_buffer_ep{episode+1}_{timestamp}.pkl')
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
        plot_training_stats(episode_rewards, episode_successes, episode_min_distances, actor_losses, critic_losses, png_dir, csv_dir, timestamp)
        
        # Save final model
        agent.save_models()
        agent.replay_buffer.save(f'{pkl_dir}/replay_buffer_final_{timestamp}.pkl')
        print(f"\n💾 Final model saved")
        
        # Final cleanup - keep only best and final buffers, clean periodic ones
        cleanup_old_files(pkl_dir, "replay_buffer_ep*.pkl", MAX_BUFFER_FILES)
        cleanup_old_files(pkl_dir, "replay_buffer_best*.pkl", 1)  # Keep only 1 best
        cleanup_old_files(pkl_dir, "replay_buffer_final*.pkl", 1)  # Keep only 1 final
        print(f"🧹 Cleaned up old buffer files")
        
        print(f"\n✅ Training complete! Trained for {args.episodes} episodes.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.destroy_node()
        rclpy.shutdown()


def plot_training_stats(episode_rewards, episode_successes, episode_min_distances, actor_losses, critic_losses, png_dir, csv_dir, timestamp):
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
    fig.suptitle('Training Statistics', fontsize=16, fontweight='bold')
    
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
    
    # Save plot
    plot_path = f'{png_dir}/training_plot_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plot saved to: {plot_path}")
    
    # Save CSV
    import csv
    csv_path = f'{csv_dir}/training_data_{timestamp}.csv'
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


def show_menu():
    """Display interactive training menu"""
    print("\n" + "="*70)
    print("🎮 TRAINING MENU")
    print("="*70)
    print("1. Manual Test Mode")
    print("2. RL Training Mode (TD3)")
    print("3. RL Training Mode (SAC)")
    print("="*70)
    
    choice = input("Select option (1-3): ").strip()
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
        # Run manual test mode from control_robot.py
        import subprocess
        print("\n🎮 Starting Manual Test Mode...")
        print("=" * 70)
        script_path = os.path.join(os.path.dirname(__file__), 'control_robot.py')
        if os.path.exists(script_path):
            subprocess.run(['python3', script_path])
        else:
            print(f"❌ control_robot.py not found at {script_path}")
        print("\n" + "=" * 70)
        print("Manual test mode exited.")
        print("=" * 70)
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
    else:
        print("❌ Invalid choice! Exiting...")


if __name__ == '__main__':
    main()
