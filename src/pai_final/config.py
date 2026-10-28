"""Configuration dataclasses and physical constants for the PAI Final Project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic and extrinsic parameters for 3D back-projection."""

    width: int = 1280
    height: int = 720
    fov_deg: float = 40.0
    cam_pos: Tuple[float, float, float] = (0.55, 0.0, 1.15)
    cam_lookat: Tuple[float, float, float] = (0.55, 0.0, 0.02)

    @property
    def focal_length_px(self) -> float:
        import math

        fov_rad = math.radians(self.fov_deg)
        return (self.height / 2.0) / math.tan(fov_rad / 2.0)

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


@dataclass(frozen=True)
class SceneObjectSpec:
    """Specification of manipulable objects or obstacles in the workspace."""

    object_id: str
    category: str
    label_ja: str
    color_rgb: Tuple[float, float, float]
    size_xyz: Tuple[float, float, float]
    initial_pos: Tuple[float, float, float]
    target_bin_pos: Tuple[float, float, float]
    priority: int  # 1 = highest priority, 99 = static obstacle (do not grasp)
    is_obstacle: bool = False


@dataclass(frozen=True)
class FrankaTaskConfig:
    """Configuration for Franka Emika Panda 7-DoF autonomous triage manipulation."""

    dt: float = 0.01
    substeps: int = 2
    workspace_radius_min: float = 0.25
    workspace_radius_max: float = 0.78
    min_safe_z: float = 0.022
    hover_clearance_z: float = 0.16
    obstacle_clearance_z: float = 0.24
    grasp_force_newton: float = 18.0
    kp_arm: Tuple[float, ...] = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0)
    kv_arm: Tuple[float, ...] = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0)
    kp_finger: Tuple[float, float] = (150.0, 150.0)
    kv_finger: Tuple[float, float] = (15.0, 15.0)
    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    objects: Tuple[SceneObjectSpec, ...] = (
        SceneObjectSpec(
            object_id="med_kit_red",
            category="emergency_medical",
            label_ja="緊急医療キット (最優先)",
            color_rgb=(0.92, 0.22, 0.22),
            size_xyz=(0.045, 0.045, 0.045),
            initial_pos=(0.52, -0.18, 0.0225),
            target_bin_pos=(0.38, 0.28, 0.035),
            priority=1,
            is_obstacle=False,
        ),
        SceneObjectSpec(
            object_id="sensor_unit_blue",
            category="precision_instrument",
            label_ja="精密通信モジュール (第2優先)",
            color_rgb=(0.18, 0.48, 0.95),
            size_xyz=(0.045, 0.045, 0.045),
            initial_pos=(0.62, -0.04, 0.0225),
            target_bin_pos=(0.48, 0.28, 0.035),
            priority=2,
            is_obstacle=False,
        ),
        SceneObjectSpec(
            object_id="supply_box_green",
            category="general_supply",
            label_ja="一般補給コンテナ (第3優先)",
            color_rgb=(0.16, 0.78, 0.38),
            size_xyz=(0.045, 0.045, 0.045),
            initial_pos=(0.46, 0.06, 0.0225),
            target_bin_pos=(0.58, 0.28, 0.035),
            priority=3,
            is_obstacle=False,
        ),
        SceneObjectSpec(
            object_id="debris_hazard_black",
            category="hazard_obstacle",
            label_ja="崩落瓦礫障害物 (回避対象)",
            color_rgb=(0.25, 0.25, 0.28),
            size_xyz=(0.07, 0.09, 0.09),
            initial_pos=(0.51, 0.16, 0.045),
            target_bin_pos=(0.51, 0.16, 0.045),
            priority=99,
            is_obstacle=True,
        ),
    )


@dataclass(frozen=True)
class Go2RLConfig:
    """Configuration for Unitree Go2 Parallel PPO Locomotion & Disturbance Recovery."""

    exp_name: str = "go2-adaptive-recovery"
    num_envs_colab: int = 2048
    num_envs_fallback: int = 1024
    max_iterations: int = 300
    dt: float = 0.02
    episode_length_s: float = 20.0
    num_obs: int = 48
    num_actions: int = 12
    action_scale: float = 0.25
    kp: float = 20.0
    kd: float = 0.5
    base_init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.42)
    # Custom Creative Extensions beyond baseline tutorial:
    enable_lateral_push_perturbation: bool = True
    push_interval_s: float = 4.0
    max_push_vel_xy: float = 0.55  # Lateral impulse disturbance (m/s)
    enable_dynamic_heading_command: bool = True
    # Reward weights
    reward_tracking_lin_vel: float = 1.2
    reward_tracking_ang_vel: float = 0.6
    reward_lin_vel_z: float = -2.0
    reward_base_height: float = -35.0
    reward_orientation_stability: float = -1.5  # Custom stability term for push recovery
    reward_action_rate: float = -0.008
    reward_similar_to_default: float = -0.1


@dataclass(frozen=True)
class VideoRenderConfig:
    """Specification for final submission MP4 video (strictly 1m00s to 3m00s)."""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    total_duration_s: float = 110.0  # 1m50s (within [60s, 180s] rule)
    playback_speed_badge: str = "2.0x Speed (2倍速再生区間)"
    account_id: str = "pai2026_demo"
