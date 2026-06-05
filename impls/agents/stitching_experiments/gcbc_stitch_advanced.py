"""
temporal stitching defaults
"""

from agents.gcbc_stitch import get_config as get_pure_stitch_config


ADVANCED_STITCH_DEFAULTS = dict(
    dataset_class='AdvancedTemporalStitchGCDataset',
    stitch_future_max=-1,
    stitch_retrieval_mode='kmeans',
    # kNN retrieval params. Only used when stitch_retrieval_mode='knn'.
    stitch_knn_k=512,
    stitch_knn_algorithm='auto',
    stitch_knn_n_jobs=1,
    # State-space retrieval extensions.
    stitch_state_normalize=False,
    stitch_state_normalize_eps=1e-6,
    stitch_state_xy_weight=1.0,
    stitch_state_xy_max_dist=-1.0,
    # Manipulation compatibility guards.
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
    # Phase-aware matching extensions.
    stitch_phase_guard_mode='none',
    stitch_phase_effector_object_max_dist=0.08,
    stitch_phase_contact_threshold=0.5,
    stitch_phase_lift_height=0.06,
    stitch_phase_gripper_open_threshold=0.15,
    # Delivery-phase splitting and segment-local futures.
    stitch_phase_delivery_near_goal_dist=0.10,
    stitch_phase_delivery_place_goal_dist=0.04,
    stitch_phase_delivery_target_tail=5,
    stitch_phase_delivery_stabilize_steps=10,
    stitch_future_guard_mode='none',
    stitch_segment_phase_label=3,
    # Stable-holding future guard.
    stitch_holding_effector_object_max_dist=0.08,
    stitch_holding_contact_threshold=0.5,
    stitch_holding_lift_height=0.06,
    stitch_holding_gripper_closed_threshold=0.5,
    stitch_holding_min_segment_steps=3,
)


def add_advanced_stitching_config(config):
    config.update(ADVANCED_STITCH_DEFAULTS)
    return config


def get_config():
    return add_advanced_stitching_config(get_pure_stitch_config())
