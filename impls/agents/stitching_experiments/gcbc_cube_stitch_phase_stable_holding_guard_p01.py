"""
Phase-segment stitching with a stable holding futuregoal guard
    - transport bridges are accepted only when g_A and w_B are both inside stable-holding intervals.
    - Non holding waypoints keep the existing same pick-place segment rule.
    - Stable-holding waypoints only select g'_B from the same holding interval
"""

from agents.stitching_experiments.gcbc_cube_stitch_phase_segment_guard_p01 import (
    get_config as get_phase_segment_config,
)


def get_config():
    config = get_phase_segment_config()
    config.update(
        dict(
            stitch_future_guard_mode='same_holding_segment',
            stitch_holding_effector_object_max_dist=0.08,
            stitch_holding_contact_threshold=0.5,
            stitch_holding_lift_height=0.06,
            stitch_holding_gripper_closed_threshold=0.5,
            stitch_holding_min_segment_steps=3,
            stitch_debug_samples=2000,
        )
    )
    return config
