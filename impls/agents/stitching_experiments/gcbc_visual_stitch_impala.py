"""
GCBC visual cube with pure temporal stitching in a learned IMPALA feature space.

This is the visual analogue of the original pure stitching baseline:
Candidate retrieval is KMeans over a feature matrix, while GCBC still trains on image
observations and image goals.
"""

from agents.gcbc import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.update(
        dict(
            encoder='impala_small',
            batch_size=256,
            p_aug=0.5,
            aug_type='crop',
            dataset_class='VisualFeatureTemporalStitchGCDataset',
            stitch_p_aug=0.5,
            stitch_feature_path='',
            stitch_feature_key='features',
            stitch_feature_normalize=True,
            stitch_feature_normalize_eps=1e-6,
            stitch_feature_dtype='float32',
            stitch_future_min=0,
            stitch_cross_traj_only=True,
            stitch_nclusters=200,
            stitch_kmeans_n_init='auto',
            stitch_kmeans_random_state=0,
            stitch_kmeans_fit_sample_size=200000,
            stitch_kmeans_batch_size=65536,
            stitch_debug_samples=2000,
            # Debug-only feature dimensions; not a guard.
            stitch_xy_dims=(0, 1),
        )
    )
    return config
