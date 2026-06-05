from agents.stitching_experiments.gcbc_cube_stitch_guard import get_config as get_guard_config


def get_config():
    config = get_guard_config()
    config.update(
        dict(
            stitch_p_aug=0.1,
            # to keep enough examples to test the run
            stitch_debug_samples=1000,
        )
    )
    return config
