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
            stitch_retrieval_mode='kmeans',
            stitch_nclusters=40,
            stitch_kmeans_n_init='auto',
            stitch_kmeans_random_state=None,
            # kNN retrieval params. only if stitch_retrieval_mode='knn'.
            stitch_knn_k=512,
            stitch_knn_algorithm='auto',
            stitch_knn_n_jobs=1,
            stitch_debug_samples=0,
            # State-space rescue ablations. only if stitch_space='state'.
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
            # rejects stitches where the original future goal
            # and matched waypoint are in different manipulation phases.
            stitch_phase_guard_mode='none',
            stitch_phase_effector_object_max_dist=0.08,
            stitch_phase_contact_threshold=0.5,
            stitch_phase_lift_height=0.06,
            stitch_phase_gripper_open_threshold=0.15,
            # the local target of each pick-place segment from the trajectory
            # and splits transport into far / near / place-release states.
            stitch_phase_delivery_near_goal_dist=0.10,
            stitch_phase_delivery_place_goal_dist=0.04,
            stitch_phase_delivery_target_tail=5,
            stitch_phase_delivery_stabilize_steps=10,
            # keeps g'B inside the same pick-place segment as the matched
            # waypoint, which avoids stitching into later repeated pick-place
            stitch_future_guard_mode='none',
            stitch_segment_phase_label=3,
        )
    )
    return config
