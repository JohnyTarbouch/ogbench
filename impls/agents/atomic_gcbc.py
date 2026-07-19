"""
config for gcbc train on atomic cube tasks.
Each episod contain multiple pick and place tasks,
atomic gcbc trained on tasks, meaning the goal sampled within the task not episode.
"""

import ml_collections

from agents.gcbc import get_config as get_base_config

def get_config():
    config = get_base_config()
    config.update(
        ml_collections.ConfigDict(
            dict(
                dataset_class='AtomicGCDataset',
                atomic_train_manifest_path='',
                atomic_val_manifest_path='',
                atomic_goal_stack_mode='repeat_endpoint',
                atomic_require_source_fingerprint=False,
            )
        )
    )
    return config
