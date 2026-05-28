"""
GCBC config with temporal stitching dataset.

Use this config for temporal stitching/relabeling experiments. Plain OGBench
GCBC runs should continue to use ``agents/gcbc.py``.
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
            stitch_future_max=-1,
            stitch_cross_traj_only=False,
            stitch_nclusters=40,
            stitch_kmeans_n_init='auto',
            stitch_kmeans_random_state=None,
            stitch_debug_samples=0,
            # State-space rescue ablations. Ignored unless stitch_space='state'.
            stitch_state_normalize=False,
            stitch_state_normalize_eps=1e-6,
            stitch_state_xy_weight=1.0,
            stitch_state_xy_max_dist=-1.0,
            # manipulation specific waypoint guard
            stitch_guard_mode='none',
            stitch_guard_max_retries=16,
            stitch_guard_exhaustive_fallback=True,
            stitch_guard_position_scale=1.0,
            stitch_guard_xy_max_dist=-1.0,
            stitch_guard_effector_dims=(),
            stitch_guard_object_dims=(),
            stitch_guard_effector_max_dist=-1.0,
            stitch_guard_rel_effector_object_max_dist=-1.0,
            stitch_guard_gripper_open_dim=-1,
            stitch_guard_gripper_open_scale=1.0,
            stitch_guard_gripper_open_max_diff=-1.0,
            stitch_guard_contact_dim=-1,
            stitch_guard_contact_max_diff=-1.0,
        )
    )
    return config
