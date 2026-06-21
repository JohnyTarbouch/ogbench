"""
Factored GCBC transfer experiment with cross-encoder visual stitchin
"""

from agents.representation_experiments.gcbc_factored_transfer import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.update(
        dict(
            encoder='drq',
            batch_size=256,
            p_aug=1.0,
            aug_type='drq_shift',
            drq_shift_pad=2,
            dataset_class='VisualFeatureTemporalStitchGCDataset',
            stitch_p_aug=0.5,
            stitch_feature_path='',
            stitch_state_feature_path='',
            stitch_goal_feature_path='',
            stitch_feature_key='features',
            stitch_state_feature_key='state_features',
            stitch_goal_feature_key='goal_features',
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
            stitch_xy_dims=(0, 1),
        )
    )
    return config
