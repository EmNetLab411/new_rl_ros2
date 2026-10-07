#!/usr/bin/env python3
"""
Backend adapters for Gazebo, sim-to-real shadow, and real replay control.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.action import ActionClient

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


SUPPORTED_CONTROL_BACKENDS = {
    'sim',
    'sim_to_real_shadow',
    'real_replay',
    'pi_direct',
}

GAZEBO_JOINT_NAMES = [
    'Revolute 20',
    'Revolute 22',
    'Revolute 23',
    'Revolute 26',
    'Revolute 28',
    'Revolute 30',
]

GAZEBO_TO_PI_JOINT_MAP = [
    ("Revolute 20", "base", 90.0, False),
    ("Revolute 22", "shoulder", 90.0, False),
    ("Revolute 23", "elbow", 90.0, False),
    # Gazebo q4=0 corresponds to the physical servo's 90deg neutral because
    # the URDF already bakes in the -90deg wrist mount orientation.
    ("Revolute 26", "wrist_roll", 90.0, False),
    ("Revolute 28", "wrist_pitch", 90.0, False),
    ("Revolute 30", "pen", 90.0, False),
]

PI_SERVO_MIN_DEG = 0.0
PI_SERVO_MAX_DEG = 180.0

SERVO_SPEED_PER_60DEG = [
    0.18,
    0.18,
    0.17,
    0.12,
    0.12,
    0.12,
]

MIN_TRAJECTORY_DURATION = 0.3
TRAJECTORY_MARGIN = 0.15
PI_REPLAY_RATE_HZ = 25.0
PI_SHADOW_REPLAY_HZ = float(os.environ.get('PI_SHADOW_REPLAY_HZ', '5.0'))
PI_REPLAY_MIN_SEGMENT_SEC = 1.0 / PI_REPLAY_RATE_HZ


def resolve_control_backend(requested: Optional[str] = None) -> str:
    """Resolve the control backend from explicit input or environment."""
    candidate = (requested or '').strip().lower()
    if not candidate:
        candidate = os.environ.get('VISUAL_SERVOING_CONTROL_BACKEND', '').strip().lower()

    legacy_mode = os.environ.get('VISUAL_SERVOING_DIGITAL_TWIN_MODE', '').strip().lower()
    if not candidate and legacy_mode == 'sim_to_real':
        candidate = 'sim_to_real_shadow'

    if candidate == 'sim_to_real':
        candidate = 'sim_to_real_shadow'

    if not candidate:
        candidate = 'sim'

    if candidate not in SUPPORTED_CONTROL_BACKENDS:
        raise ValueError(
            f"Unsupported control backend '{candidate}'. "
            f"Expected one of {sorted(SUPPORTED_CONTROL_BACKENDS)}."
        )
    return candidate


def _duration_msg(duration_sec: float) -> Duration:
    sec = int(duration_sec)
    nanosec = int((duration_sec - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)


class GazeboPiMapper:
    """Shared Gazebo<->Pi joint/state mapping and replay downsampling helpers."""

    def __init__(self):
        self.gazebo_joint_names = list(GAZEBO_JOINT_NAMES)
        self.gazebo_to_pi = list(GAZEBO_TO_PI_JOINT_MAP)
        self.pi_joint_names = [item[1] for item in self.gazebo_to_pi]
        self.pi_lookup = {item[1]: item for item in self.gazebo_to_pi}
        self.gazebo_lookup = {item[0]: item for item in self.gazebo_to_pi}

    def gazebo_rad_to_pi_deg(self, gazebo_rad: float, home_deg: float, inverted: bool) -> float:
        offset_deg = np.degrees(gazebo_rad)
        if inverted:
            offset_deg = -offset_deg
        return float(np.clip(home_deg + offset_deg, PI_SERVO_MIN_DEG, PI_SERVO_MAX_DEG))

    def pi_deg_to_gazebo_rad(self, pi_deg: float, home_deg: float, inverted: bool) -> float:
        offset_deg = float(pi_deg) - float(home_deg)
        if inverted:
            offset_deg = -offset_deg
        return np.radians(offset_deg)

    def gazebo_positions_to_pi_deg(self, target_positions: np.ndarray) -> Dict[str, float]:
        target_positions = np.asarray(target_positions, dtype=np.float64)
        gz_lookup = {
            joint_name: float(target_positions[idx])
            for idx, joint_name in enumerate(self.gazebo_joint_names[: len(target_positions)])
        }
        result = {}
        for gz_name, pi_name, home_deg, inverted in self.gazebo_to_pi:
            if gz_name not in gz_lookup:
                continue
            result[pi_name] = self.gazebo_rad_to_pi_deg(gz_lookup[gz_name], home_deg, inverted)
        return result

    def pi_joint_state_to_gazebo(
        self, msg: JointState
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        positions = [0.0] * len(self.gazebo_joint_names)
        velocities = [0.0] * len(self.gazebo_joint_names)
        found_all = True
        msg_lookup = {name: idx for idx, name in enumerate(msg.name)}

        for gz_idx, gz_name in enumerate(self.gazebo_joint_names):
            _, pi_name, home_deg, inverted = self.gazebo_lookup[gz_name]
            if pi_name not in msg_lookup:
                found_all = False
                continue

            jidx = msg_lookup[pi_name]
            pi_value = float(msg.position[jidx])
            if abs(pi_value) < 6.3:
                pi_value = np.degrees(pi_value)
            positions[gz_idx] = self.pi_deg_to_gazebo_rad(pi_value, home_deg, inverted)
            if len(msg.velocity) > jidx:
                velocities[gz_idx] = float(msg.velocity[jidx])

        return np.array(positions, dtype=np.float64), np.array(velocities, dtype=np.float64), found_all

    def build_pi_trajectory_msg(self, positions_deg: Dict[str, float], duration_sec: float) -> JointTrajectory:
        traj = JointTrajectory()
        point = JointTrajectoryPoint()

        for pi_name in self.pi_joint_names:
            if pi_name not in positions_deg:
                continue
            traj.joint_names.append(pi_name)
            point.positions.append(float(positions_deg[pi_name]))

        point.time_from_start = _duration_msg(float(duration_sec))
        traj.points = [point]
        return traj

    def build_pi_trajectory_from_gazebo(
        self, target_positions: np.ndarray, duration_sec: float
    ) -> JointTrajectory:
        return self.build_pi_trajectory_msg(
            self.gazebo_positions_to_pi_deg(target_positions),
            duration_sec,
        )

    def export_pi_replay_plan(
        self,
        joint_samples_rad: List[np.ndarray],
        sample_dt: float,
        joint_limits_low: np.ndarray,
        joint_limits_high: np.ndarray,
        replay_rate_hz: float = PI_REPLAY_RATE_HZ,
        min_segment_sec: float = PI_REPLAY_MIN_SEGMENT_SEC,
    ) -> Dict:
        """Downsample high-rate PID commands into Pi-safe timed segments."""
        if sample_dt <= 0.0:
            raise ValueError(f"sample_dt must be positive, got {sample_dt}")

        samples = [np.asarray(s, dtype=np.float64) for s in joint_samples_rad]
        if not samples:
            raise ValueError("No joint samples available for replay export")

        low = np.asarray(joint_limits_low, dtype=np.float64)
        high = np.asarray(joint_limits_high, dtype=np.float64)
        for idx, sample in enumerate(samples):
            if sample.shape != low.shape:
                raise ValueError(
                    f"Sample #{idx} shape {sample.shape} does not match limits shape {low.shape}"
                )
            if np.any(sample < (low - 1e-6)) or np.any(sample > (high + 1e-6)):
                raise ValueError(f"Sample #{idx} exceeds joint limits")

        downsample_stride = max(1, int(round((1.0 / replay_rate_hz) / sample_dt)))
        segment_indices = list(range(downsample_stride - 1, len(samples), downsample_stride))
        final_index = len(samples) - 1
        if not segment_indices:
            segment_indices = [final_index]
        elif segment_indices[-1] != final_index:
            remaining_steps = final_index - segment_indices[-1]
            if (remaining_steps * sample_dt) < min_segment_sec:
                segment_indices[-1] = final_index
            else:
                segment_indices.append(final_index)

        # Determine lead steps for lag compensation.
        # Can be set via environment variable:
        # - PI_REPLAY_LAG_COMP_STEPS: explicitly sets the number of lead steps.
        # - PI_REPLAY_LAG_COMP_SECONDS: sets target seconds to compensate (default: 0.2s).
        lead_steps = 0
        lag_steps_env = os.environ.get('PI_REPLAY_LAG_COMP_STEPS')
        if lag_steps_env is not None:
            try:
                lead_steps = max(0, int(lag_steps_env))
            except ValueError:
                pass
        else:
            lag_sec_env = os.environ.get('PI_REPLAY_LAG_COMP_SECONDS', '0.2')
            try:
                lag_sec = max(0.0, float(lag_sec_env))
                lead_steps = int(round(lag_sec * replay_rate_hz))
            except ValueError:
                pass

        raw_segments = []
        prev_idx = -1
        for sample_idx in segment_indices:
            segment_steps = sample_idx - prev_idx
            duration_sec = segment_steps * sample_dt
            if duration_sec < min_segment_sec - 1e-9:
                raise ValueError(
                    f"Replay segment duration {duration_sec:.4f}s is below minimum {min_segment_sec:.4f}s"
                )

            positions_rad = samples[sample_idx]
            raw_segments.append({
                'sample_index': int(sample_idx),
                'duration_sec': float(duration_sec),
                'positions_rad': positions_rad,
            })
            prev_idx = sample_idx

        # Extract the original start position before any lead shift
        original_start_rad = samples[0]
        original_start_deg_dict = self.gazebo_positions_to_pi_deg(original_start_rad)
        original_start_deg = [original_start_deg_dict[name] for name in self.pi_joint_names]

        segments = []
        num_segs = len(raw_segments)
        for i in range(num_segs):
            # Lead shift: command target of (i + lead_steps) early
            target_idx = min(i + lead_steps, num_segs - 1)
            compensated_rad = raw_segments[target_idx]['positions_rad']
            compensated_deg = self.gazebo_positions_to_pi_deg(compensated_rad)

            segments.append({
                'sample_index': raw_segments[i]['sample_index'],
                'duration_sec': raw_segments[i]['duration_sec'],
                'positions_rad': compensated_rad.tolist(),
                'positions_deg': [compensated_deg[name] for name in self.pi_joint_names],
                'joint_names_pi': list(self.pi_joint_names),
            })

        return {
            'source_dt_sec': float(sample_dt),
            'replay_rate_hz': float(replay_rate_hz),
            'min_segment_sec': float(min_segment_sec),
            'downsample_stride': int(downsample_stride),
            'segment_count': len(segments),
            'segments': segments,
            'original_start_joint_deg': original_start_deg,
            'lead_steps_applied': lead_steps,
        }


class MotionBackendBase:
    """Common interface for control backends."""

    joint_state_topic = '/joint_states'
    supports_reward_feedback = True
    supports_high_rate_streaming = False
    uses_gazebo_model_states = False

    def __init__(self, node):
        self.node = node
        self.mapper = GazeboPiMapper()

    def extract_joint_state(self, msg: JointState) -> Tuple[np.ndarray, np.ndarray, bool]:
        positions = [0.0] * len(self.mapper.gazebo_joint_names)
        velocities = [0.0] * len(self.mapper.gazebo_joint_names)
        found_all = True
        msg_lookup = {name: idx for idx, name in enumerate(msg.name)}

        for idx, joint_name in enumerate(self.mapper.gazebo_joint_names):
            if joint_name not in msg_lookup:
                found_all = False
                continue
            jidx = msg_lookup[joint_name]
            positions[idx] = float(msg.position[jidx])
            velocities[idx] = float(msg.velocity[jidx]) if len(msg.velocity) > jidx else 0.0

        return np.array(positions, dtype=np.float64), np.array(velocities, dtype=np.float64), found_all

    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.array(self.node.joint_positions, dtype=np.float64),
            np.array(self.node.joint_velocities, dtype=np.float64),
        )

    def move_to_joint_positions(self, target_positions: np.ndarray, duration: Optional[float] = None) -> bool:
        raise NotImplementedError

    def stream_joint_positions(self, target_positions: np.ndarray, duration: float = 0.01) -> bool:
        raise NotImplementedError

    def home(self, duration: float = 2.0) -> bool:
        return self.move_to_joint_positions(np.zeros(len(self.mapper.gazebo_joint_names)), duration=duration)

    def replay_exported_plan(self, replay_plan: Dict, label: str = 'replay') -> bool:
        raise NotImplementedError

    def estimate_real_duration(self, target_positions: np.ndarray) -> float:
        max_time = 0.0
        current_positions, _ = self.get_joint_state()
        for idx in range(min(len(target_positions), len(SERVO_SPEED_PER_60DEG))):
            delta_deg = np.degrees(abs(float(target_positions[idx]) - float(current_positions[idx])))
            servo_time = (delta_deg / 60.0) * SERVO_SPEED_PER_60DEG[idx]
            max_time = max(max_time, servo_time)
        duration = max(max_time + TRAJECTORY_MARGIN, MIN_TRAJECTORY_DURATION)
        return round(duration, 2)


class GazeboBackend(MotionBackendBase):
    joint_state_topic = '/joint_states'
    supports_reward_feedback = True
    supports_high_rate_streaming = True
    uses_gazebo_model_states = True

    def __init__(self, node):
        super().__init__(node)
        self.trajectory_client = ActionClient(
            node,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
        )
        self.fast_trajectory_pub = node.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10,
        )
        node.get_logger().info("⏳ Connecting to Gazebo trajectory action server...")
        if not self.trajectory_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("Trajectory action server timeout")
        node.get_logger().info("✅ Gazebo trajectory action server connected!")

    def move_to_joint_positions(self, target_positions: np.ndarray, duration: Optional[float] = None) -> bool:
        target_positions = np.clip(
            np.asarray(target_positions, dtype=np.float64),
            self.node.gazebo_limits_low,
            self.node.gazebo_limits_high,
        )
        if duration is None:
            duration = self.estimate_real_duration(target_positions)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = list(self.mapper.gazebo_joint_names)
        goal_msg.trajectory.header.stamp = self.node.get_clock().now().to_msg()

        point = JointTrajectoryPoint()
        point.positions = target_positions.tolist()
        point.velocities = [0.0] * len(self.mapper.gazebo_joint_names)
        point.time_from_start = _duration_msg(duration)
        goal_msg.trajectory.points = [point]

        try:
            self.node.get_logger().info(
                f"Sending Gazebo trajectory: {np.degrees(target_positions).astype(int)}° (dur={duration:.2f}s)"
            )
            send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=2.0)

            goal_handle = send_goal_future.result()
            if goal_handle is None:
                self.node.get_logger().error(
                    "Trajectory goal failed before acceptance (no goal handle returned)"
                )
                return False
            if not goal_handle.accepted:
                self.node.get_logger().error("Goal rejected by action server")
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=duration + 2.0)
            result = result_future.result()
            if result is None:
                self.node.get_logger().error("Trajectory action did not return a result")
                return False
            time.sleep(0.2)
            return True
        except Exception as exc:
            self.node.get_logger().error(f"Trajectory execution error: {exc}")
            return False

    def stream_joint_positions(self, target_positions: np.ndarray, duration: float = 0.01) -> bool:
        target_positions = np.clip(
            np.asarray(target_positions, dtype=np.float64),
            self.node.gazebo_limits_low,
            self.node.gazebo_limits_high,
        )
        goal_msg = JointTrajectory()
        goal_msg.joint_names = list(self.mapper.gazebo_joint_names)
        goal_msg.header.stamp = self.node.get_clock().now().to_msg()

        point = JointTrajectoryPoint()
        point.positions = target_positions.tolist()
        point.velocities = [0.0] * len(self.mapper.gazebo_joint_names)
        point.time_from_start = _duration_msg(duration)
        goal_msg.points = [point]

        self.fast_trajectory_pub.publish(goal_msg)
        return True

    def replay_exported_plan(self, replay_plan: Dict, label: str = 'replay') -> bool:
        self.node.get_logger().warn(
            f"Gazebo backend received replay plan '{label}' but has no hardware replay path"
        )
        return False


class SimToRealShadowBackend(GazeboBackend):
    supports_reward_feedback = True
    supports_high_rate_streaming = True

    def __init__(self, node, mirror_safe_moves: bool = True):
        self.mirror_safe_moves = bool(mirror_safe_moves)
        super().__init__(node)
        self.real_joint_trajectory_pub = node.create_publisher(
            JointTrajectory,
            '/pca9685_servo/trajectory',
            10,
        )
        self.home_client = node.create_client(Trigger, '/pca9685_servo/home')
        self.pi_joint_positions = None
        self.pi_joint_state_sub = node.create_subscription(
            JointState,
            '/pca9685_servo/joint_states',
            self._pi_joint_state_callback,
            10
        )
        node.get_logger().info(
            "🔄 Sim-to-real shadow backend ready: Gazebo scores the episode, "
            "Pi replay is explicit and high-rate streaming is never mirrored"
        )

    def _pi_joint_state_callback(self, msg: JointState):
        try:
            positions, _, _ = self.mapper.pi_joint_state_to_gazebo(msg)
            self.pi_joint_positions = positions
        except Exception:
            pass

    def home(self, duration: float = 2.0) -> bool:
        ok_gz = super().home(duration=duration)
        ok_pi = self._home_physical_robot_only(duration=duration)
        return ok_gz and ok_pi

    def _home_physical_robot_only(self, duration: float = 2.0) -> bool:
        """Home only the physical robot, leaving Gazebo untouched."""
        ok_pi = False
        if self.home_client.wait_for_service(timeout_sec=1.0):
            try:
                future = self.home_client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
                resp = future.result()
                if resp is not None and resp.success:
                    ok_pi = True
                    time.sleep(0.5)
            except Exception as e:
                self.node.get_logger().error(f"Failed to call Pi home service: {e}")
        if not ok_pi:
            self.node.get_logger().warn(
                "Pi home service failed or not ready; falling back to joint trajectory home move"
            )
            home_joints = np.zeros(len(self.mapper.gazebo_joint_names))
            self._publish_real_robot_command(home_joints, duration)
            time.sleep(duration + 0.2)
            ok_pi = True
        return ok_pi

    def _publish_real_robot_command(self, target_positions: np.ndarray, duration: float):
        traj = self.mapper.build_pi_trajectory_from_gazebo(target_positions, duration)
        traj.header.stamp = self.node.get_clock().now().to_msg()
        if not traj.joint_names or not traj.points:
            self.node.get_logger().warn("Digital twin sync skipped: no joints mapped for Pi trajectory")
            return
        self.real_joint_trajectory_pub.publish(traj)

    def move_to_joint_positions(self, target_positions: np.ndarray, duration: Optional[float] = None) -> bool:
        ok = super().move_to_joint_positions(target_positions, duration=duration)
        if ok and self.mirror_safe_moves:
            if duration is None:
                duration = self.estimate_real_duration(target_positions)
            self._publish_real_robot_command(target_positions, duration)
        return ok

    def replay_episode_trajectory(
        self,
        commanded_trace_rad: List[np.ndarray],
        sample_dt: float,
        joint_limits_low: np.ndarray,
        joint_limits_high: np.ndarray,
        replay_rate_hz: Optional[float] = None,
    ) -> bool:
        """After a PID episode, replay the downsampled trajectory on the Pi."""
        rate = replay_rate_hz if replay_rate_hz is not None else PI_SHADOW_REPLAY_HZ
        try:
            replay_plan = self.mapper.export_pi_replay_plan(
                joint_samples_rad=commanded_trace_rad,
                sample_dt=sample_dt,
                joint_limits_low=joint_limits_low,
                joint_limits_high=joint_limits_high,
                replay_rate_hz=rate,
            )
            self.node.get_logger().info(
                f"🔄 Shadow replay: {replay_plan['segment_count']} segments at {rate:.1f}Hz"
            )
            return self.replay_exported_plan(replay_plan, label='shadow_pid_episode')
        except Exception as e:
            self.node.get_logger().warn(f"Shadow PID replay failed: {e}")
            return False

    def replay_exported_plan(self, replay_plan: Dict, label: str = 'shadow_replay') -> bool:
        from datetime import datetime
        segments = replay_plan.get('segments', [])
        if not segments:
            self.node.get_logger().warn(f"No replay segments available for {label}")
            return False

        rate = replay_plan.get('replay_rate_hz', PI_SHADOW_REPLAY_HZ)
        self.node.get_logger().info(
            f"🔄 Replaying {len(segments)} Pi-safe segments for {label} at {rate:.1f}Hz"
        )

        # Setup logging
        log_dir = os.path.join(os.getcwd(), 'training_results', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f"{label}_log_{timestamp}.txt")

        with open(log_path, 'w') as f:
            f.write(f"=== {label.upper()} REPLAY LOG ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Replay Rate: {rate} Hz\n")
            f.write(f"Total Segments: {len(segments)}\n")
            f.write("--------------------------------------------------------------------------------\n")

        # Use original uncompensated start position if available, to prevent skipping first step
        original_start_list = replay_plan.get('original_start_joint_deg')
        if original_start_list is not None:
            start_positions_deg = {
                name: float(pos)
                for name, pos in zip(self.mapper.pi_joint_names, original_start_list)
            }
        else:
            start_positions_deg = {
                name: float(pos)
                for name, pos in zip(segments[0]['joint_names_pi'], segments[0]['positions_deg'])
            }
        prep_str = (
            "🏠 Preparing physical robot for shadow replay...\n"
            "   Home -> move to replay start -> settle\n"
        )
        print(prep_str.strip())
        with open(log_path, 'a') as f:
            f.write(prep_str)

        self._home_physical_robot_only(duration=2.0)
        time.sleep(1.0)

        start_traj = self.mapper.build_pi_trajectory_msg(start_positions_deg, 2.0)
        start_traj.header.stamp = self.node.get_clock().now().to_msg()
        if not start_traj.joint_names or not start_traj.points:
            self.node.get_logger().warn(f"Shadow replay start move skipped: no joints mapped for {label}")
            return False
        self.real_joint_trajectory_pub.publish(start_traj)
        time.sleep(2.0)
        for _ in range(15):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        segments_sent = 0
        segments_with_feedback = 0

        for idx, segment in enumerate(segments):
            # Print and log commanded angles
            positions_deg = {
                name: float(pos)
                for name, pos in zip(segment['joint_names_pi'], segment['positions_deg'])
            }
            cmd_str = ", ".join(f"{k}={positions_deg[k]:.1f}°" for k in self.mapper.pi_joint_names)

            # Send trajectory segment
            traj = self.mapper.build_pi_trajectory_msg(positions_deg, float(segment['duration_sec']))
            traj.header.stamp = self.node.get_clock().now().to_msg()
            self.real_joint_trajectory_pub.publish(traj)
            segments_sent += 1

            # Reset tracking variable
            self.pi_joint_positions = None

            # Sleep for segment duration
            time.sleep(float(segment['duration_sec']) + 0.02)

            # Spin to process incoming '/pca9685_servo/joint_states' messages
            for _ in range(10):
                rclpy.spin_once(self.node, timeout_sec=0.01)

            # Measure actual angles
            if self.pi_joint_positions is not None:
                segments_with_feedback += 1
                actual_deg_dict = self.mapper.gazebo_positions_to_pi_deg(self.pi_joint_positions)
                actual_str = ", ".join(f"{k}={actual_deg_dict[k]:.1f}°" for k in self.mapper.pi_joint_names)
                packet_status = "OK"
            else:
                actual_str = "TIMEOUT/NO DATA"
                packet_status = "NO_FEEDBACK"

            log_line = (
                f"[SEG {idx+1}/{len(segments)}] "
                f"Cmd: [{cmd_str}] | "
                f"Actual: [{actual_str}] | "
                f"Status: {packet_status} | "
                f"dur={segment['duration_sec']:.2f}s\n"
            )
            print(log_line.strip())
            with open(log_path, 'a') as f:
                f.write(log_line)

        # Print & write summary
        feedback_miss_pct = (
            ((segments_sent - segments_with_feedback) / segments_sent) * 100.0
            if segments_sent > 0 else 0.0
        )
        summary_str = (
            f"\n--- Replay Summary ---\n"
            f"Sent: {segments_sent} | Segments with feedback: {segments_with_feedback} | "
            f"Feedback miss rate: {feedback_miss_pct:.1f}%\n"
            f"Log saved to: {log_path}\n"
        )
        print(summary_str)
        with open(log_path, 'a') as f:
            f.write(summary_str)

        # Automatic Homing
        print("🏠 Returning physical robot to home position...")
        self.home(duration=2.0)
        return True


class RealReplayBackend(MotionBackendBase):
    joint_state_topic = '/pca9685_servo/joint_states'
    supports_reward_feedback = False
    supports_high_rate_streaming = False
    uses_gazebo_model_states = False

    def __init__(self, node):
        super().__init__(node)
        self.real_joint_trajectory_pub = node.create_publisher(
            JointTrajectory,
            '/pca9685_servo/trajectory',
            10,
        )
        self.home_client = node.create_client(Trigger, '/pca9685_servo/home')
        node.get_logger().info("⏳ Connecting to Pi home service...")
        self.home_client.wait_for_service(timeout_sec=5.0)
        node.get_logger().info("✅ Real replay backend ready")

    def extract_joint_state(self, msg: JointState) -> Tuple[np.ndarray, np.ndarray, bool]:
        return self.mapper.pi_joint_state_to_gazebo(msg)

    def move_to_joint_positions(self, target_positions: np.ndarray, duration: Optional[float] = None) -> bool:
        target_positions = np.clip(
            np.asarray(target_positions, dtype=np.float64),
            self.node.gazebo_limits_low,
            self.node.gazebo_limits_high,
        )
        if duration is None:
            duration = self.estimate_real_duration(target_positions)

        traj = self.mapper.build_pi_trajectory_from_gazebo(target_positions, duration)
        traj.header.stamp = self.node.get_clock().now().to_msg()
        if not traj.joint_names or not traj.points:
            self.node.get_logger().error("Real replay move skipped: no joints mapped")
            return False

        self.real_joint_trajectory_pub.publish(traj)
        self.node.get_logger().info(
            f"Sending Pi trajectory: {traj.joint_names} dur={duration:.2f}s"
        )
        time.sleep(duration + 0.2)
        return True

    def stream_joint_positions(self, target_positions: np.ndarray, duration: float = 0.01) -> bool:
        self.node.get_logger().warn(
            "Real replay backend does not support high-rate streaming; use replay_exported_plan instead"
        )
        return False

    def home(self, duration: float = 2.0) -> bool:
        home_joints = np.zeros(len(self.mapper.gazebo_joint_names))

        if not self.home_client.service_is_ready():
            return self.move_to_joint_positions(home_joints, duration=duration)
        future = self.home_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        resp = future.result()
        if resp is None or not resp.success:
            self.node.get_logger().warn("Pi home service failed; falling back to joint trajectory home move")
            return self.move_to_joint_positions(home_joints, duration=duration)
        time.sleep(0.5)
        return True

    def replay_exported_plan(self, replay_plan: Dict, label: str = 'real_replay') -> bool:
        segments = replay_plan.get('segments', [])
        if not segments:
            self.node.get_logger().warn(f"No replay segments available for {label}")
            return False

        self.node.get_logger().info(
            f"▶️ Replaying {len(segments)} hardware segments for {label} "
            f"at {replay_plan.get('replay_rate_hz', PI_REPLAY_RATE_HZ):.1f}Hz"
        )
        for segment in segments:
            positions_deg = {
                name: float(pos)
                for name, pos in zip(segment['joint_names_pi'], segment['positions_deg'])
            }
            traj = self.mapper.build_pi_trajectory_msg(positions_deg, float(segment['duration_sec']))
            traj.header.stamp = self.node.get_clock().now().to_msg()
            self.real_joint_trajectory_pub.publish(traj)
            time.sleep(float(segment['duration_sec']) + 0.02)
        return True


class PiDirectBackend(RealReplayBackend):
    """Pi-direct training backend: send commands to Pi, get feedback via FK.

    Identical to RealReplayBackend for communication, but enables
    reward_feedback so the RL training loop can compute rewards from
    FK-derived end-effector positions.  No Gazebo required.

    Usage:
        - Pi must be running wicom_roboarm on the same ROS 2 DDS domain.
        - End-effector position is computed by FK (exact URDF math),
          not TF2, since there is no robot_state_publisher running.
    """
    supports_reward_feedback = True
    supports_high_rate_streaming = True
    uses_gazebo_model_states = False

    def __init__(self, node):
        super().__init__(node)
        self.real_joint_command_pub = node.create_publisher(
            JointState,
            '/pca9685_servo/command',
            10,
        )
        node.get_logger().info(
            "🤖 Pi-direct backend ready: training directly on hardware, "
            "reward computed via FK (no Gazebo needed)"
        )

    def stream_joint_positions(self, target_positions: np.ndarray, duration: float = 0.01) -> bool:
        target_positions = np.clip(
            np.asarray(target_positions, dtype=np.float64),
            self.node.gazebo_limits_low,
            self.node.gazebo_limits_high,
        )
        # Convert target_positions (radians) to Pi degrees dictionary
        cmd_deg = self.mapper.gazebo_positions_to_pi_deg(target_positions)
        
        # Build JointState message
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        for name, val in cmd_deg.items():
            msg.name.append(name)
            msg.position.append(val)
            
        self.real_joint_command_pub.publish(msg)
        return True


def create_motion_backend(node, backend_name: Optional[str] = None, mirror_safe_moves: bool = True):
    resolved = resolve_control_backend(backend_name)
    if resolved == 'sim':
        return GazeboBackend(node)
    if resolved == 'sim_to_real_shadow':
        return SimToRealShadowBackend(node, mirror_safe_moves=mirror_safe_moves)
    if resolved == 'real_replay':
        return RealReplayBackend(node)
    if resolved == 'pi_direct':
        return PiDirectBackend(node)
    raise ValueError(f"Unsupported control backend '{resolved}'")
