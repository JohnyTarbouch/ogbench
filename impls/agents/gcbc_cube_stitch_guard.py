"""
GCBC with cube temporal stitching plus manipulation guards.

second manip relabeling experiment. It keeps the successful parts of the cube only setup:

    - policy input remains full 28D observation and full 28D goal
    - waypoint clusters are built from cube xyz dims obs[19:22]
    - relabeling is cross-trajectory

But before accepting a waypoint w for original goal g, it checks that
the robot-gripper phase is also compatible. This directly tests the failure
analysis of cube-only stitching: cube position overlap alone is too loose.
"""

from agents.gcbc_cube_stitch import get_config as get_cube_stitch_config


def get_config():
    config = get_cube_stitch_config()
    config.update(
        dict(
            stitch_guard_mode='cube_gripper',
            stitch_guard_position_scale=10.0,
            stitch_guard_xy_max_dist=0.05,  # cube g-w <= 5 cm
            stitch_guard_effector_dims=(12, 13, 14),
            stitch_guard_object_dims=(19, 20, 21),
            stitch_guard_effector_max_dist=0.10,  # end-effector g-w <= 10 cm
            stitch_guard_rel_effector_object_max_dist=0.10,  # relative gripper-cube <= 10 cm
            stitch_guard_gripper_open_dim=17,
            stitch_guard_gripper_open_scale=3.0,
            stitch_guard_gripper_open_max_diff=0.20,
            stitch_guard_contact_dim=18,
            stitch_guard_contact_max_diff=0.5,
            stitch_guard_max_retries=32,
            stitch_guard_exhaustive_fallback=True,
            stitch_debug_samples=2000,
        )
    )
    return config
