"""
Visual cube GCIQL with the DrQ encoder and DrQ-style image augmentation.
"""

from agents.gciql import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.update(
        dict(
            encoder='drq',
            alpha=1.0,
            batch_size=256,
            p_aug=0.5,
            aug_type='drq_shift',
            drq_shift_pad=2,
        )
    )
    return config
