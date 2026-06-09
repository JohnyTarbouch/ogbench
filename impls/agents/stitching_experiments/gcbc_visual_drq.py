"""
Visual cube GCBC with the DrQ encoder and image augmentation.
"""

from agents.gcbc import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.update(
        dict(
            encoder='drq',
            batch_size=256,
            p_aug=0.5,
            aug_type='drq_shift',
            drq_shift_pad=2,
        )
    )
    return config
