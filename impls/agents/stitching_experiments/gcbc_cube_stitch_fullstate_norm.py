"""
GCBC with normalized full-state temporal stitching for cube manipulation.
"""

from agents.gcbc_cube_stitch import get_config as get_cube_stitch_config
from agents.stitching_experiments.gcbc_stitch_advanced import add_advanced_stitching_config


def get_config():
    config = get_cube_stitch_config()
    config = add_advanced_stitching_config(config)
    config.update(
        dict(
            stitch_space='state',
            stitch_state_normalize=True,
            stitch_guard_mode='none',
            stitch_phase_guard_mode='none',
            stitch_future_guard_mode='none',
            stitch_state_xy_max_dist=-1.0,
        )
    )
    return config
