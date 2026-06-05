"""
GCBC config with pure temporal stitching.
"""

from agents.gcbc import get_config as get_gcbc_config


def get_config():
    config = get_gcbc_config()
    config.update(
        dict(
            dataset_class='TemporalStitchGCDataset',
            # Temporal stitching augmentation.
            stitch_p_aug=0.0,
            stitch_space='state',
            stitch_xy_dims=(0, 1),
            stitch_future_min=0,
            stitch_cross_traj_only=False,
            stitch_nclusters=40,
            stitch_kmeans_n_init='auto',
            stitch_kmeans_random_state=None,
            stitch_debug_samples=0,
        )
    )
    return config
