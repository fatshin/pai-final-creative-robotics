"""Comprehensive Unit & Integration Tests for PAI Final Project Pipeline."""

from __future__ import annotations

import ast
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from pai_final.config import FrankaTaskConfig, Go2RLConfig, VideoRenderConfig
from pai_final.franka_manipulation import FrankaTriageController
from pai_final.go2_env import Go2AdaptiveSimulator, get_go2_cfgs
from pai_final.go2_train_eval import run_go2_pipeline
from pai_final.reasoning_planner import (
    MinimumJerkTrajectory,
    SpatialReasoningPlanner,
)


class TestSpatialReasoningAndTrajectory(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = FrankaTaskConfig()
        self.planner = SpatialReasoningPlanner(self.cfg)

    def test_pinhole_projection_and_backprojection_invertibility(self) -> None:
        """Verify 3D -> 2D pixel -> 3D back-projection accuracy within 0.1 mm."""
        test_point = (0.52, -0.18, 0.0225)
        u, v, depth = self.planner.project_world_to_pixel(test_point)
        recovered = self.planner.backproject_pixel_to_world(u, v, depth)
        for orig, rec in zip(test_point, recovered):
            self.assertAlmostEqual(orig, rec, places=4)

    def test_workspace_safety_clamping(self) -> None:
        """Verify out-of-range coordinates are clamped inside Franka safe bounds."""
        too_far = (1.5, 0.0, -0.2)
        clamped = self.planner.clamp_to_safe_workspace(too_far)
        radius = math.hypot(clamped[0], clamped[1])
        self.assertLessEqual(radius, self.cfg.workspace_radius_max + 1e-5)
        self.assertGreaterEqual(clamped[2], self.cfg.min_safe_z)

        origin_pt = (0.0, 0.0, 0.0)
        clamped_origin = self.planner.clamp_to_safe_workspace(origin_pt)
        origin_radius = math.hypot(clamped_origin[0], clamped_origin[1])
        self.assertGreaterEqual(origin_radius, self.cfg.workspace_radius_min - 1e-5)

    def test_priority_sorting_and_obstacle_avoidance_detection(self) -> None:
        """Verify AI planner sorts items by urgency (1->2->3) and detects obstacle intersection."""
        plan = self.planner.generate_action_plan()
        self.assertEqual(len(plan), 3)
        self.assertEqual(
            [step.object_id for step in plan],
            ["med_kit_red", "sensor_unit_blue", "supply_box_green"],
        )
        # Step 1 (red) has a clear path (Z=0.16m); Step 2 (blue) and Step 3 (green) intersect debris (Z=0.24m)
        self.assertFalse(plan[0].requires_obstacle_avoidance)
        self.assertAlmostEqual(plan[0].via_clearance_z, self.cfg.hover_clearance_z)
        self.assertTrue(plan[1].requires_obstacle_avoidance)
        self.assertAlmostEqual(plan[1].via_clearance_z, self.cfg.obstacle_clearance_z)
        self.assertTrue(plan[2].requires_obstacle_avoidance)
        self.assertAlmostEqual(plan[2].via_clearance_z, self.cfg.obstacle_clearance_z)

    def test_minimum_jerk_boundary_conditions(self) -> None:
        """Verify 5th-order polynomial starts at 0 and ends at 1 with zero boundary velocity/accel."""
        tau = np.array([0.0, 0.5, 1.0])
        s, ds, d2s = MinimumJerkTrajectory.time_scaling(tau)
        self.assertAlmostEqual(float(s[0]), 0.0)
        self.assertAlmostEqual(float(s[1]), 0.5)
        self.assertAlmostEqual(float(s[2]), 1.0)
        self.assertAlmostEqual(float(ds[0]), 0.0)
        self.assertAlmostEqual(float(ds[2]), 0.0)
        self.assertAlmostEqual(float(d2s[0]), 0.0)
        self.assertAlmostEqual(float(d2s[2]), 0.0)


class TestFrankaAndGo2Simulation(unittest.TestCase):
    def test_franka_rollout_completes_all_objects_accurately(self) -> None:
        controller = FrankaTriageController()
        telemetry, plan = controller.run_simulation_rollout(
            use_genesis_if_available=False, steps_per_phase=20
        )
        self.assertEqual(len(plan), 3)
        self.assertGreater(len(telemetry), 100)
        final_objs = telemetry[-1].object_positions
        for step in plan:
            placed_pos = final_objs[step.object_id]
            dist = np.linalg.norm(np.array(placed_pos) - np.array(step.place_pos_xyz))
            self.assertLess(float(dist), 0.01)

    def test_go2_rl_configs_and_trained_vs_untrained_improvement(self) -> None:
        cfg = Go2RLConfig()
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = get_go2_cfgs(cfg)
        self.assertEqual(env_cfg["num_actions"], 12)
        self.assertEqual(obs_cfg["num_obs"], 48)
        self.assertIn("orientation_stability", reward_cfg["reward_scales"])
        self.assertEqual(train_cfg["algorithm"]["class_name"], "PPO")

        res = run_go2_pipeline(log_root="/tmp/pai_test_logs", execute_genesis_gpu=False)
        summary = res["summary"]
        self.assertGreater(summary["final_reward"], summary["initial_reward"] + 15.0)
        self.assertGreater(summary["final_push_recovery_rate_pct"], 90.0)


class TestColabNotebookAndVideoConfig(unittest.TestCase):
    def test_video_duration_strictly_within_1_to_3_minutes(self) -> None:
        vcfg = VideoRenderConfig()
        self.assertGreaterEqual(vcfg.total_duration_s, 60.0)
        self.assertLessEqual(vcfg.total_duration_s, 180.0)
        self.assertIn("2", vcfg.playback_speed_badge)

    def test_colab_notebook_valid_syntax(self) -> None:
        nb_path = ROOT_DIR / "notebooks" / "PAI_Final_Project_Colab.ipynb"
        if not nb_path.exists():
            return
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 5)
        for cell in code_cells:
            lines = [
                line
                for line in cell["source"]
                if not line.lstrip().startswith(("!", "%"))
            ]
            ast.parse("".join(lines))


if __name__ == "__main__":
    unittest.main()
