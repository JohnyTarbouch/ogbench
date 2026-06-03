"""
Full-state temporal stitching for cube manipulation.
"""

from agents.gcbc_cube_stitch import get_config as get_cube_stitch_config


def get_config():
    config = get_cube_stitch_config()
    config.update(
        dict(
            stitch_space='state',
            stitch_state_normalize=False,
        )
    )
    return config
