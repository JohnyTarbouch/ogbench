"""
The stitch is accepted only when the original future goal and the
matched waypoint are in the same manipulation phase.
"""

from agents.gcbc_cube_stitch_guard_p01 import get_config as get_guard_p01_config


def get_config():
    config = get_guard_p01_config()
    config.update(
        dict(
            stitch_space='state',
            stitch_state_normalize=True,
            stitch_state_xy_max_dist=0.5,
            stitch_future_max=-1,
            stitch_phase_guard_mode='cube_simple',
            stitch_phase_effector_object_max_dist=0.08,
            stitch_phase_contact_threshold=0.5,
            stitch_phase_lift_height=0.06,
            stitch_phase_gripper_open_threshold=0.15,
            stitch_debug_samples=2000,
        )
    )
    return config
