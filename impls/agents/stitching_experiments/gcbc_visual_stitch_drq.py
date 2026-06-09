"""GCBC visual cube with pure temporal stitching in a learned DrQ feature space."""

from agents.stitching_experiments.gcbc_visual_stitch_impala import get_config as get_impala_stitch_config


def get_config():
    config = get_impala_stitch_config()
    config.update(
        dict(
            encoder='drq',
            p_aug=0.5,
            aug_type='drq_shift',
            drq_shift_pad=2,
        )
    )
    return config
