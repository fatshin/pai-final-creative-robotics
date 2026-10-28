"""Physical AI Applied Course 1 Final Project Package.

Integrates:
1. VLM / 3D Pinhole Spatial Reasoning & Minimum-Jerk Trajectory Planning
2. Franka Emika Panda 7-DoF Inverse Kinematics & Autonomous Triage Manipulation
3. Unitree Go2 Parallel PPO Reinforcement Learning with Push-Perturbation Recovery
4. Telemetry HUD Video Compositor for Google Colab & Headless GPU Execution
"""

from .config import (
    CameraIntrinsics,
    FrankaTaskConfig,
    Go2RLConfig,
    SceneObjectSpec,
    VideoRenderConfig,
)
from .reasoning_planner import (
    MinimumJerkTrajectory,
    PlannedActionStep,
    SpatialReasoningPlanner,
)

__all__ = [
    "CameraIntrinsics",
    "FrankaTaskConfig",
    "Go2RLConfig",
    "MinimumJerkTrajectory",
    "PlannedActionStep",
    "SceneObjectSpec",
    "SpatialReasoningPlanner",
    "VideoRenderConfig",
]
