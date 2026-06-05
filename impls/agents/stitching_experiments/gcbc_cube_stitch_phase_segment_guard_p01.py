"""
GCBC with phase-aware stitching plus single-segment future-goal guarding.
"""

from agents.stitching_experiments.gcbc_cube_stitch_phase_guard_p01 import get_config as get_phase_guard_config


def get_config():
    config = get_phase_guard_config()
    config.update(
        dict(
            stitch_future_guard_mode='same_segment',
            stitch_segment_phase_label=3,
            stitch_debug_samples=2000,
        )
    )
    return config
