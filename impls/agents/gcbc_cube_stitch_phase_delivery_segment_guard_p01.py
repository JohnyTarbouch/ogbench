"""
Goal-relative delivery phases and single segment future guarding.
Extends the phase segment experiment. The transport
phase is split using the target of each offline pick-place segment.
"""

from agents.gcbc_cube_stitch_phase_segment_guard_p01 import get_config as get_segment_guard_config


def get_config():
    config = get_segment_guard_config()
    config.update(
        dict(
            stitch_phase_guard_mode='cube_delivery',
            stitch_phase_delivery_near_goal_dist=0.10,
            stitch_phase_delivery_place_goal_dist=0.04,
            stitch_phase_delivery_target_tail=5,
            stitch_phase_delivery_stabilize_steps=10,
            stitch_debug_samples=2000,
        )
    )
    return config
