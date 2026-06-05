from __future__ import annotations

import dataclasses

import jax
import numpy as np

from utils.datasets import GCDataset


@dataclasses.dataclass
class AdvancedTemporalStitchGCDataset(GCDataset):
    """
    Base GCBC sampling uses a future state from the same trajectory as the
    actor goal. Temporal stitching keeps the current state/action pair from
    trajectory A, but sometimes replaces the actor goal with a later state from
    trajectory B.

    The condition is that original future goal g_A must match a waypoint
    w_B in another trajectory under the configured retrieval space and guards.
    The implementation is intentionally split into retrieval and filtering:

    1. Retrieve waypoint candidates for g_A using KMeans or kNN.
    2. Reject behaviorally incompatible waypoints with optional cube/gripper, contact, phase, and delivery guards.
    3. Sample a new future goal g'_B after waypoint w_B.
    4. Optionally keep g'_B inside the same phase, pick-place segment, or stable-holding interval.


    config:
    - stitch_p_aug: Probability of trying to replace an actor goal
    - stitch_space: Retrieval feature space. state uses full observation, xy uses selected dims
    - stitch_xy_dims: Observation dims used for XY matching and debug coordinates
    - stitch_future_min: Minimum future offset after the waypoint for the augmented goal
    - stitch_future_max: Maximum future offset after the waypoint for the augmented goal. Negative disables it
    - stitch_cross_traj_only: if waypoints must come from a different trajectory
    - stitch_retrieval_mode: candidate retrieval backend (kmeans or knn)
    - stitch_nclusters: Number of k-means clusters
    - stitch_kmeans_n_init: Number of k-means init
    - stitch_kmeans_random_state: random state for reproducible k-mean
    # KNN params:
    - stitch_knn_k: Number of nearest waypoint candidates for knn retrieval
    - stitch_knn_algorithm: sklearn NearestNeighbors algorithm
    - stitch_knn_n_jobs: sklearn NearestNeighbors worker count

    - stitch_state_normalize: State-only, z-score full-state features before retrieval
    - stitch_state_normalize_eps: Minimum standard deviation used for state normalization
    - stitch_state_xy_weight: upweight XY dims in the retrieval feature space
    - stitch_state_xy_max_dist: positive values reject waypoint candidates farther than this in true XY
    - stitch_guard_mode: manipulation compatibility guard

    # Manip phase guard params:
    - stitch_phase_guard_mode: optional manipulation phase guard
    - stitch_phase_delivery_near_goal_dist: distance threshold for near-target transport
    - stitch_phase_delivery_place_goal_dist: distance threshold for placement/release
    - stitch_phase_delivery_target_tail: stable segment-tail states used to infer the local target
    - stitch_phase_delivery_stabilize_steps: post-release states kept in the placement phase
    - stitch_future_guard_mode: optional guard on the substituted future goal
    - stitch_holding_*: thresholds used to identify uninterrupted stable-holding intervals
    """

    def __post_init__(self):
        super().__post_init__()
        self._init_stitching()

    def _config_get(self, key, default):
        return self.config[key] if key in self.config else default

    def _init_stitching(self):
        # core stitching controls:
        # this keeps the original temporal-stitching behavior where g'_B may be
        # any later state after the matched waypoint
        self.stitch_p_aug = float(self._config_get('stitch_p_aug', 0.0))
        self.stitch_space = self._config_get('stitch_space', 'state')
        self.stitch_xy_dims = tuple(self._config_get('stitch_xy_dims', (0, 1)))
        self.stitch_future_min = int(self._config_get('stitch_future_min', 0))
        stitch_future_max = int(self._config_get('stitch_future_max', -1))
        self.stitch_future_max = None if stitch_future_max < 0 else stitch_future_max
        self.stitch_cross_traj_only = bool(self._config_get('stitch_cross_traj_only', False))

        # boundaries were hiding useful nearby waypoint candidates
        self.stitch_retrieval_mode = self._config_get('stitch_retrieval_mode', 'kmeans')
        self.stitch_nclusters = int(self._config_get('stitch_nclusters', 40))
        self.stitch_kmeans_n_init = self._config_get('stitch_kmeans_n_init', 'auto')
        self.stitch_kmeans_random_state = self._config_get('stitch_kmeans_random_state', None)
        self.stitch_knn_k = int(self._config_get('stitch_knn_k', 512))
        self.stitch_knn_algorithm = self._config_get('stitch_knn_algorithm', 'auto')
        self.stitch_knn_n_jobs = int(self._config_get('stitch_knn_n_jobs', 1))
        self.stitch_debug_samples = int(self._config_get('stitch_debug_samples', 0))
        self.stitch_state_normalize = bool(self._config_get('stitch_state_normalize', False))
        self.stitch_state_normalize_eps = float(self._config_get('stitch_state_normalize_eps', 1e-6))
        self.stitch_state_xy_weight = float(self._config_get('stitch_state_xy_weight', 1.0))
        self.stitch_state_xy_max_dist = float(self._config_get('stitch_state_xy_max_dist', -1.0))
        if self.stitch_state_xy_max_dist <= 0:
            self.stitch_state_xy_max_dist = None

        # Manip guard: after retrieval, g_A and w_B are compatible in robot/cube state
        self.stitch_guard_mode = self._config_get('stitch_guard_mode', 'none')
        self.stitch_guard_max_retries = int(self._config_get('stitch_guard_max_retries', 16))
        self.stitch_guard_exhaustive_fallback = bool(self._config_get('stitch_guard_exhaustive_fallback', True))
        self.stitch_guard_position_scale = float(self._config_get('stitch_guard_position_scale', 1.0))
        self.stitch_guard_xy_max_dist = float(self._config_get('stitch_guard_xy_max_dist', -1.0))
        self.stitch_guard_effector_dims = tuple(self._config_get('stitch_guard_effector_dims', ()))
        self.stitch_guard_object_dims = tuple(self._config_get('stitch_guard_object_dims', self.stitch_xy_dims))
        self.stitch_guard_effector_max_dist = float(self._config_get('stitch_guard_effector_max_dist', -1.0))
        self.stitch_guard_rel_effector_object_max_dist = float(
            self._config_get('stitch_guard_rel_effector_object_max_dist', -1.0)
        )
        self.stitch_guard_gripper_open_dim = int(self._config_get('stitch_guard_gripper_open_dim', -1))
        self.stitch_guard_gripper_open_scale = float(self._config_get('stitch_guard_gripper_open_scale', 1.0))
        self.stitch_guard_gripper_open_max_diff = float(self._config_get('stitch_guard_gripper_open_max_diff', -1.0))
        self.stitch_guard_contact_dim = int(self._config_get('stitch_guard_contact_dim', -1))
        self.stitch_guard_contact_max_diff = float(self._config_get('stitch_guard_contact_max_diff', -1.0))

        # second manipulation-specific filter, to prevents
        # stitching across different control regimes
        self.stitch_phase_guard_mode = self._config_get('stitch_phase_guard_mode', 'none')
        self.stitch_phase_effector_object_max_dist = float(
            self._config_get('stitch_phase_effector_object_max_dist', 0.08)
        )
        self.stitch_phase_contact_threshold = float(self._config_get('stitch_phase_contact_threshold', 0.5))
        self.stitch_phase_lift_height = float(self._config_get('stitch_phase_lift_height', 0.06))
        self.stitch_phase_gripper_open_threshold = float(
            self._config_get('stitch_phase_gripper_open_threshold', 0.15)
        )
        self.stitch_phase_delivery_near_goal_dist = float(
            self._config_get('stitch_phase_delivery_near_goal_dist', 0.10)
        )
        self.stitch_phase_delivery_place_goal_dist = float(
            self._config_get('stitch_phase_delivery_place_goal_dist', 0.04)
        )
        self.stitch_phase_delivery_target_tail = int(
            self._config_get('stitch_phase_delivery_target_tail', 5)
        )
        self.stitch_phase_delivery_stabilize_steps = int(
            self._config_get('stitch_phase_delivery_stabilize_steps', 10)
        )

        # keeps the new goal inside one pick-place attempt, reducing repeated-pick-place confusion.
        self.stitch_future_guard_mode = self._config_get('stitch_future_guard_mode', 'none')
        self.stitch_segment_phase_label = int(self._config_get('stitch_segment_phase_label', 3))
        self.stitch_holding_effector_object_max_dist = float(
            self._config_get('stitch_holding_effector_object_max_dist', 0.08)
        )
        self.stitch_holding_contact_threshold = float(
            self._config_get('stitch_holding_contact_threshold', 0.5)
        )
        self.stitch_holding_lift_height = float(self._config_get('stitch_holding_lift_height', 0.06))
        self.stitch_holding_gripper_closed_threshold = float(
            self._config_get('stitch_holding_gripper_closed_threshold', 0.5)
        )
        self.stitch_holding_min_segment_steps = int(
            self._config_get('stitch_holding_min_segment_steps', 3)
        )
        self.stitch_enabled = self.stitch_p_aug > 0
        self._stitch_metric_sums = {}
        self._stitch_metric_count = 0
        self._stitch_sample_calls = 0
        self._stitch_debug_records = []

        # lets guards enforce cross-trajectory stitching and same-segment future goals.
        self.traj_ids = np.empty(self.size, dtype=np.int32)
        for traj_id, (initial_idx, terminal_idx) in enumerate(zip(self.initial_locs, self.terminal_locs)):
            self.traj_ids[initial_idx : terminal_idx + 1] = traj_id

        self.stitch_summary = {
            'enabled': bool(self.stitch_enabled),
            'stitch_p_aug': self.stitch_p_aug,
            'stitch_space': self.stitch_space,
            'stitch_xy_dims': list(self.stitch_xy_dims),
            'stitch_future_min': self.stitch_future_min,
            'stitch_future_max': self.stitch_future_max,
            'stitch_cross_traj_only': self.stitch_cross_traj_only,
            'stitch_retrieval_mode': self.stitch_retrieval_mode,
            'stitch_nclusters': self.stitch_nclusters,
            'stitch_kmeans_n_init': self.stitch_kmeans_n_init,
            'stitch_kmeans_random_state': self.stitch_kmeans_random_state,
            'stitch_knn_k': self.stitch_knn_k,
            'stitch_knn_algorithm': self.stitch_knn_algorithm,
            'stitch_knn_n_jobs': self.stitch_knn_n_jobs,
            'stitch_debug_samples': self.stitch_debug_samples,
            'stitch_state_normalize': self.stitch_state_normalize,
            'stitch_state_normalize_eps': self.stitch_state_normalize_eps,
            'stitch_state_xy_weight': self.stitch_state_xy_weight,
            'stitch_state_xy_max_dist': self.stitch_state_xy_max_dist,
            'stitch_guard_mode': self.stitch_guard_mode,
            'stitch_guard_max_retries': self.stitch_guard_max_retries,
            'stitch_guard_exhaustive_fallback': self.stitch_guard_exhaustive_fallback,
            'stitch_guard_position_scale': self.stitch_guard_position_scale,
            'stitch_guard_xy_max_dist': self.stitch_guard_xy_max_dist,
            'stitch_guard_effector_dims': list(self.stitch_guard_effector_dims),
            'stitch_guard_object_dims': list(self.stitch_guard_object_dims),
            'stitch_guard_effector_max_dist': self.stitch_guard_effector_max_dist,
            'stitch_guard_rel_effector_object_max_dist': self.stitch_guard_rel_effector_object_max_dist,
            'stitch_guard_gripper_open_dim': self.stitch_guard_gripper_open_dim,
            'stitch_guard_gripper_open_scale': self.stitch_guard_gripper_open_scale,
            'stitch_guard_gripper_open_max_diff': self.stitch_guard_gripper_open_max_diff,
            'stitch_guard_contact_dim': self.stitch_guard_contact_dim,
            'stitch_guard_contact_max_diff': self.stitch_guard_contact_max_diff,
            'stitch_phase_guard_mode': self.stitch_phase_guard_mode,
            'stitch_phase_effector_object_max_dist': self.stitch_phase_effector_object_max_dist,
            'stitch_phase_contact_threshold': self.stitch_phase_contact_threshold,
            'stitch_phase_lift_height': self.stitch_phase_lift_height,
            'stitch_phase_gripper_open_threshold': self.stitch_phase_gripper_open_threshold,
            'stitch_phase_delivery_near_goal_dist': self.stitch_phase_delivery_near_goal_dist,
            'stitch_phase_delivery_place_goal_dist': self.stitch_phase_delivery_place_goal_dist,
            'stitch_phase_delivery_target_tail': self.stitch_phase_delivery_target_tail,
            'stitch_phase_delivery_stabilize_steps': self.stitch_phase_delivery_stabilize_steps,
            'stitch_future_guard_mode': self.stitch_future_guard_mode,
            'stitch_segment_phase_label': self.stitch_segment_phase_label,
            'stitch_holding_effector_object_max_dist': self.stitch_holding_effector_object_max_dist,
            'stitch_holding_contact_threshold': self.stitch_holding_contact_threshold,
            'stitch_holding_lift_height': self.stitch_holding_lift_height,
            'stitch_holding_gripper_closed_threshold': self.stitch_holding_gripper_closed_threshold,
            'stitch_holding_min_segment_steps': self.stitch_holding_min_segment_steps,
        }

        if not self.stitch_enabled:
            self.stitch_points = None
            self.stitch_xy_points = None
            self.stitch_phase_labels = None
            self.stitch_coarse_phase_labels = None
            self.stitch_phase_goal_distances = None
            self.stitch_segment_goal_points = None
            self.stitch_segment_ids = None
            self.stitch_holding_mask = None
            self.stitch_holding_segment_ids = None
            self.stitch_cluster_labels = None
            self.stitch_cluster_to_waypoint_idxs = {}
            self.stitch_knn = None
            self.stitch_knn_waypoint_idxs = None
            self.stitch_summary.update({'num_waypoints': 0, 'num_groups': 0})
            return

        if self.stitch_space not in ('state', 'xy'):
            raise ValueError("stitch_space must be one of {'state', 'xy'}.")
        if self.stitch_retrieval_mode not in ('kmeans', 'knn'):
            raise ValueError("stitch_retrieval_mode must be one of {'kmeans', 'knn'}.")
        if self.stitch_nclusters <= 0:
            raise ValueError('stitch_nclusters must be positive.')
        if self.stitch_knn_k <= 0:
            raise ValueError('stitch_knn_k must be positive.')
        if self.stitch_future_min < 0:
            raise ValueError('stitch_future_min must be non-negative.')
        if self.stitch_future_max is not None and self.stitch_future_max < self.stitch_future_min:
            raise ValueError('stitch_future_max must be >= stitch_future_min, or negative to disable it.')
        if self.stitch_state_normalize_eps <= 0:
            raise ValueError('stitch_state_normalize_eps must be positive.')
        if self.stitch_state_xy_weight <= 0:
            raise ValueError('stitch_state_xy_weight must be positive.')
        if self.stitch_guard_mode not in ('none', 'cube_gripper'):
            raise ValueError("stitch_guard_mode must be one of {'none', 'cube_gripper'}.")
        if self.stitch_phase_guard_mode not in ('none', 'cube_simple', 'cube_delivery'):
            raise ValueError("stitch_phase_guard_mode must be one of {'none', 'cube_simple', 'cube_delivery'}.")
        if self.stitch_future_guard_mode not in ('none', 'same_phase', 'same_segment', 'same_holding_segment'):
            raise ValueError(
                "stitch_future_guard_mode must be one of "
                "{'none', 'same_phase', 'same_segment', 'same_holding_segment'}."
            )
        if self.stitch_future_guard_mode != 'none' and self.stitch_phase_guard_mode == 'none':
            raise ValueError('stitch_future_guard_mode requires stitch_phase_guard_mode to be enabled.')
        if self.stitch_guard_max_retries <= 0:
            raise ValueError('stitch_guard_max_retries must be positive.')
        if self.stitch_guard_position_scale <= 0:
            raise ValueError('stitch_guard_position_scale must be positive.')
        if self.stitch_guard_gripper_open_scale <= 0:
            raise ValueError('stitch_guard_gripper_open_scale must be positive.')
        if self.stitch_phase_delivery_place_goal_dist <= 0:
            raise ValueError('stitch_phase_delivery_place_goal_dist must be positive.')
        if self.stitch_phase_delivery_near_goal_dist < self.stitch_phase_delivery_place_goal_dist:
            raise ValueError(
                'stitch_phase_delivery_near_goal_dist must be >= stitch_phase_delivery_place_goal_dist.'
            )
        if self.stitch_phase_delivery_target_tail <= 0:
            raise ValueError('stitch_phase_delivery_target_tail must be positive.')
        if self.stitch_phase_delivery_stabilize_steps < 0:
            raise ValueError('stitch_phase_delivery_stabilize_steps must be non-negative.')
        if self.stitch_holding_effector_object_max_dist <= 0:
            raise ValueError('stitch_holding_effector_object_max_dist must be positive.')
        if self.stitch_holding_lift_height <= 0:
            raise ValueError('stitch_holding_lift_height must be positive.')
        if self.stitch_holding_min_segment_steps <= 0:
            raise ValueError('stitch_holding_min_segment_steps must be positive.')
        observations = self.dataset['observations']
        if not isinstance(observations, np.ndarray) or observations.ndim != 2:
            raise ValueError('AdvancedTemporalStitchGCDataset expects state observations as a 2D numpy array.')
        if max(self.stitch_xy_dims) >= observations.shape[-1]:
            raise ValueError(
                f'stitch_xy_dims={self.stitch_xy_dims} incompatible with observation shape {observations.shape}.'
            )
        if self.stitch_guard_mode != 'none':
            guard_dims = list(self.stitch_guard_effector_dims) + list(self.stitch_guard_object_dims)
            if self.stitch_guard_gripper_open_dim >= 0:
                guard_dims.append(self.stitch_guard_gripper_open_dim)
            if self.stitch_guard_contact_dim >= 0:
                guard_dims.append(self.stitch_guard_contact_dim)
            if not guard_dims:
                raise ValueError('stitch_guard_mode requires at least one guard dimension.')
            if max(guard_dims) >= observations.shape[-1] or min(guard_dims) < 0:
                raise ValueError(
                    f'stitch guard dimensions incompatible with observation shape {observations.shape}.'
                )
            if self.stitch_guard_rel_effector_object_max_dist > 0 and (
                len(self.stitch_guard_effector_dims) != len(self.stitch_guard_object_dims)
            ):
                raise ValueError(
                    'stitch_guard_rel_effector_object_max_dist requires effector/object dims of equal length.'
                )
        if self.stitch_phase_guard_mode != 'none':
            phase_dims = list(self.stitch_guard_effector_dims) + list(self.stitch_guard_object_dims)
            if self.stitch_guard_gripper_open_dim >= 0:
                phase_dims.append(self.stitch_guard_gripper_open_dim)
            if self.stitch_guard_contact_dim >= 0:
                phase_dims.append(self.stitch_guard_contact_dim)
            if not phase_dims:
                raise ValueError('stitch_phase_guard_mode requires cube/gripper guard dimensions.')
            if max(phase_dims) >= observations.shape[-1] or min(phase_dims) < 0:
                raise ValueError(
                    f'stitch phase dimensions incompatible with observation shape {observations.shape}.'
                )
            if len(self.stitch_guard_effector_dims) != len(self.stitch_guard_object_dims):
                raise ValueError('stitch_phase_guard_mode requires effector/object dims of equal length.')

        self.stitch_xy_points = observations[:, list(self.stitch_xy_dims)].astype(np.float32)
        self.stitch_observations = observations.astype(np.float32)
        self.stitch_coarse_phase_labels = self._compute_coarse_phase_labels(self.stitch_observations)
        self.stitch_segment_ids = self._compute_phase_segment_ids(self.stitch_coarse_phase_labels)
        self.stitch_phase_labels = self._compute_phase_labels(
            self.stitch_observations,
            self.stitch_coarse_phase_labels,
            self.stitch_segment_ids,
        )
        self.stitch_holding_mask, self.stitch_holding_segment_ids = self._compute_holding_segments(
            self.stitch_observations
        )
        if self.stitch_future_guard_mode == 'same_holding_segment' and not np.any(self.stitch_holding_mask):
            raise ValueError(
                'same_holding_segment found no stable-holding states; check holding thresholds and observation scales.'
            )
        if self.stitch_coarse_phase_labels is not None:
            coarse_phase_ids, coarse_phase_counts = np.unique(self.stitch_coarse_phase_labels, return_counts=True)
            self.stitch_summary.update(
                {
                    'coarse_phase_label_ids': [int(x) for x in coarse_phase_ids],
                    'coarse_phase_label_counts': [int(x) for x in coarse_phase_counts],
                }
            )
        if self.stitch_phase_labels is not None:
            phase_ids, phase_counts = np.unique(self.stitch_phase_labels, return_counts=True)
            self.stitch_summary.update(
                {
                    'phase_label_ids': [int(x) for x in phase_ids],
                    'phase_label_counts': [int(x) for x in phase_counts],
                }
            )
        if self.stitch_segment_ids is not None:
            segment_ids, segment_counts = np.unique(self.stitch_segment_ids, return_counts=True)
            self.stitch_summary.update(
                {
                    'num_phase_segments': int(len(segment_ids)),
                    'phase_segment_size_min': int(np.min(segment_counts)) if len(segment_counts) else 0,
                    'phase_segment_size_mean': float(np.mean(segment_counts)) if len(segment_counts) else 0.0,
                    'phase_segment_size_max': int(np.max(segment_counts)) if len(segment_counts) else 0,
                }
            )
        if self.stitch_holding_segment_ids is not None:
            holding_ids = self.stitch_holding_segment_ids[self.stitch_holding_segment_ids >= 0]
            _, holding_counts = np.unique(holding_ids, return_counts=True)
            transport = self.stitch_phase_labels == self.stitch_segment_phase_label
            self.stitch_summary.update(
                {
                    'holding_fraction': float(np.mean(self.stitch_holding_mask)),
                    'transport_holding_coverage': (
                        float(np.mean(self.stitch_holding_mask[transport])) if np.any(transport) else 0.0
                    ),
                    'unstable_transport_fraction': float(np.mean(transport & ~self.stitch_holding_mask)),
                    'num_holding_segments': int(len(holding_counts)),
                    'holding_segment_size_min': int(np.min(holding_counts)) if len(holding_counts) else 0,
                    'holding_segment_size_mean': float(np.mean(holding_counts)) if len(holding_counts) else 0.0,
                    'holding_segment_size_max': int(np.max(holding_counts)) if len(holding_counts) else 0,
                }
            )
        if self.stitch_space == 'state':
            self.stitch_points = observations.astype(np.float32)
            if self.stitch_state_normalize:

                # each observation dimension contributes equally 
                state_mean = np.mean(self.stitch_points, axis=0, keepdims=True)
                state_std = np.std(self.stitch_points, axis=0, keepdims=True)
                state_std = np.maximum(state_std, self.stitch_state_normalize_eps)
                self.stitch_points = (self.stitch_points - state_mean) / state_std
                self.stitch_summary.update(
                    {
                        'state_normalize_mean_abs_mean': float(np.mean(np.abs(state_mean))),
                        'state_normalize_std_min': float(np.min(state_std)),
                        'state_normalize_std_mean': float(np.mean(state_std)),
                    }
                )
            if self.stitch_state_xy_weight != 1.0:
                self.stitch_points = np.array(self.stitch_points, copy=True)
                self.stitch_points[:, list(self.stitch_xy_dims)] *= self.stitch_state_xy_weight
        else:
            self.stitch_points = self.stitch_xy_points

        # removes terminal states that cannot produce a stitched future goal.
        valid_waypoint_idxs = self.dataset.valid_idxs if hasattr(self.dataset, 'valid_idxs') else np.arange(self.size)
        waypoint_terminal_idxs = self.terminal_locs[self.traj_ids[valid_waypoint_idxs]]
        valid_waypoint_idxs = valid_waypoint_idxs[
            valid_waypoint_idxs + self.stitch_future_min <= waypoint_terminal_idxs
        ]
        self.valid_waypoint_idxs = valid_waypoint_idxs.astype(np.int64)

        self.stitch_cluster_labels = None
        self.stitch_cluster_to_waypoint_idxs = {}
        self.stitch_knn = None
        self.stitch_knn_waypoint_idxs = None
        if self.stitch_retrieval_mode == 'kmeans':
            self._build_kmeans_groups()
        else:
            self._build_knn_index()

    def _build_kmeans_groups(self):
        # cluster every observation . At training time only consider valid waypoint 
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "AdvancedTemporalStitchGCDataset requires scikit-learn. Install it with "
            ) from exc

        kmeans = KMeans(
            n_clusters=self.stitch_nclusters,
            n_init=self.stitch_kmeans_n_init,
            random_state=self.stitch_kmeans_random_state,
        )
        self.stitch_cluster_labels = kmeans.fit_predict(self.stitch_points).astype(np.int64)

        for cluster_id in range(self.stitch_nclusters):
            waypoint_idxs = self.valid_waypoint_idxs[self.stitch_cluster_labels[self.valid_waypoint_idxs] == cluster_id]
            if len(waypoint_idxs) > 0:
                self.stitch_cluster_to_waypoint_idxs[int(cluster_id)] = waypoint_idxs.astype(np.int64)

        group_sizes = np.array([len(v) for v in self.stitch_cluster_to_waypoint_idxs.values()], dtype=np.int64)
        self.stitch_summary.update(
            {
                'num_waypoints': int(len(self.valid_waypoint_idxs)),
                'num_groups': int(len(self.stitch_cluster_to_waypoint_idxs)),
                'group_size_min': int(np.min(group_sizes)) if len(group_sizes) else 0,
                'group_size_mean': float(np.mean(group_sizes)) if len(group_sizes) else 0.0,
                'group_size_max': int(np.max(group_sizes)) if len(group_sizes) else 0,
                'kmeans_inertia': float(kmeans.inertia_),
            }
        )

    def _build_knn_index(self):
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:
            raise ImportError(
                "AdvancedTemporalStitchGCDataset requires scikit-learn for knn retrieval. Install it with "
            ) from exc

        # avoids hard cluster boundaries by querying nearest valid waypoints directly
        self.stitch_knn_waypoint_idxs = self.valid_waypoint_idxs.astype(np.int64)
        if len(self.stitch_knn_waypoint_idxs) == 0:
            raise ValueError('No valid waypoint indices available for knn retrieval.')

        self.stitch_knn = NearestNeighbors(
            n_neighbors=min(self.stitch_knn_k, len(self.stitch_knn_waypoint_idxs)),
            algorithm=self.stitch_knn_algorithm,
            metric='euclidean',
            n_jobs=self.stitch_knn_n_jobs,
        )
        self.stitch_knn.fit(self.stitch_points[self.stitch_knn_waypoint_idxs])
        self.stitch_summary.update(
            {
                'num_waypoints': int(len(self.stitch_knn_waypoint_idxs)),
                'num_groups': 0,
                'knn_k_effective': int(min(self.stitch_knn_k, len(self.stitch_knn_waypoint_idxs))),
            }
        )

    def sample(self, batch_size, idxs=None, evaluation=False):
        # sample the normal GC batch
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )
        if not evaluation:
            actor_goal_idxs = self.augment_actor_goal_idxs(idxs, actor_goal_idxs)

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    ['observations', 'next_observations', 'value_goals', 'actor_goals']
                )

        return batch

    def augment_actor_goal_idxs(self, idxs, actor_goal_idxs):
        # temporal stitching step. For a subset of actor goals:
        # original goal g_A -> matched waypoint w_B -> substituted goal g'_B
        self._stitch_sample_calls += 1
        batch_size = len(idxs)
        if not self.stitch_enabled:
            self._record_stitch_metrics(
                {
                    'attempted_fraction': 0.0,
                    'accepted_fraction': 0.0,
                    'accepted_given_attempted': 0.0,
                    'candidate_count_mean': 0.0,
                    'candidate_count_after_xy_guard_mean': 0.0,
                    'goal_waypoint_distance_mean': np.nan,
                    'goal_waypoint_stitch_distance_mean': np.nan,
                    'goal_waypoint_xy_distance_mean': np.nan,
                    'future_offset_mean': np.nan,
                }
            )
            return actor_goal_idxs

        augmented_goal_idxs = np.array(actor_goal_idxs, copy=True)
        attempt_mask = np.random.rand(batch_size) < self.stitch_p_aug
        attempt_positions = np.flatnonzero(attempt_mask)
        accepted = 0
        candidate_counts = []
        guarded_candidate_counts = []
        stitch_distances = []
        xy_distances = []
        future_offsets = []
        effector_distances = []
        rel_effector_object_distances = []
        gripper_open_distances = []
        contact_distances = []
        phase_matches = []
        future_candidate_counts = []
        future_guarded_candidate_counts = []
        future_guard_kept_fractions = []
        future_phase_matches = []
        future_segment_matches = []
        holding_matches = []
        future_waypoint_holding = []
        future_goal_holding = []
        future_holding_segment_matches = []
        future_holding_guard_kept_fractions = []

        if len(attempt_positions) > 0:
            waypoint_infos = self.sample_waypoints(
                idxs[attempt_positions], actor_goal_idxs[attempt_positions]
            )

            for batch_pos, waypoint_info in zip(attempt_positions, waypoint_infos):
                if waypoint_info is None:
                    continue

                sample_idx = int(idxs[batch_pos])
                original_goal_idx = int(actor_goal_idxs[batch_pos])
                (
                    waypoint_idx,
                    stitch_distance,
                    xy_distance,
                    candidate_count,
                    guarded_candidate_count,
                    guard_stats,
                ) = waypoint_info
                waypoint_traj = self.traj_ids[waypoint_idx]
                terminal_idx = int(self.terminal_locs[waypoint_traj])
                min_goal_idx = waypoint_idx + self.stitch_future_min
                max_goal_idx = terminal_idx
                if self.stitch_future_max is not None:
                    max_goal_idx = min(max_goal_idx, waypoint_idx + self.stitch_future_max)
                if min_goal_idx > max_goal_idx:
                    continue

                # once w_B is accepted, sample the actual replacement 
                # goal g'_B from the future of trajectory B
                new_goal_idx, future_stats = self._sample_new_goal_idx(
                    original_goal_idx,
                    waypoint_idx,
                    min_goal_idx,
                    max_goal_idx,
                )
                if new_goal_idx is None:
                    continue
                augmented_goal_idxs[batch_pos] = new_goal_idx
                accepted += 1
                candidate_counts.append(candidate_count)
                guarded_candidate_counts.append(guarded_candidate_count)
                stitch_distances.append(stitch_distance)
                xy_distances.append(xy_distance)
                future_offsets.append(new_goal_idx - waypoint_idx)
                self._append_guard_metric(effector_distances, guard_stats.get('effector_distance'))
                self._append_guard_metric(
                    rel_effector_object_distances,
                    guard_stats.get('rel_effector_object_distance'),
                )
                self._append_guard_metric(gripper_open_distances, guard_stats.get('gripper_open_distance'))
                self._append_guard_metric(contact_distances, guard_stats.get('contact_distance'))
                self._append_guard_metric(phase_matches, guard_stats.get('phase_match'))
                self._append_guard_metric(holding_matches, guard_stats.get('holding_match'))
                self._append_guard_metric(future_candidate_counts, future_stats.get('future_candidate_count'))
                self._append_guard_metric(
                    future_guarded_candidate_counts,
                    future_stats.get('future_guard_candidate_count'),
                )
                self._append_guard_metric(
                    future_guard_kept_fractions,
                    future_stats.get('future_guard_kept_fraction'),
                )
                self._append_guard_metric(future_phase_matches, future_stats.get('future_goal_phase_match'))
                self._append_guard_metric(future_segment_matches, future_stats.get('future_goal_same_segment'))
                self._append_guard_metric(future_waypoint_holding, future_stats.get('waypoint_holding'))
                self._append_guard_metric(future_goal_holding, future_stats.get('future_goal_holding'))
                self._append_guard_metric(
                    future_holding_segment_matches,
                    future_stats.get('future_goal_same_holding_segment'),
                )
                self._append_guard_metric(
                    future_holding_guard_kept_fractions,
                    future_stats.get('future_holding_guard_kept_fraction'),
                )
                self._maybe_record_stitch_debug(
                    batch_pos=batch_pos,
                    sample_idx=sample_idx,
                    original_goal_idx=original_goal_idx,
                    waypoint_idx=waypoint_idx,
                    new_goal_idx=new_goal_idx,
                    stitch_distance=stitch_distance,
                    xy_distance=xy_distance,
                    candidate_count=candidate_count,
                    guarded_candidate_count=guarded_candidate_count,
                    guard_stats=guard_stats,
                    future_stats=future_stats,
                )

        attempted = len(attempt_positions)
        self._record_stitch_metrics(
            {
                'attempted_fraction': attempted / batch_size,
                'accepted_fraction': accepted / batch_size,
                'accepted_given_attempted': accepted / attempted if attempted > 0 else 0.0,
                'candidate_count_mean': float(np.mean(candidate_counts)) if candidate_counts else 0.0,
                'candidate_count_after_xy_guard_mean': (
                    float(np.mean(guarded_candidate_counts)) if guarded_candidate_counts else 0.0
                ),
                'goal_waypoint_stitch_distance_mean': (
                    float(np.mean(stitch_distances)) if stitch_distances else np.nan
                ),
                # Backward-compatible alias. For new analysis prefer the explicit stitch/XY distance metrics.
                'goal_waypoint_distance_mean': float(np.mean(stitch_distances)) if stitch_distances else np.nan,
                'goal_waypoint_xy_distance_mean': float(np.mean(xy_distances)) if xy_distances else np.nan,
                'future_offset_mean': float(np.mean(future_offsets)) if future_offsets else np.nan,
                'goal_waypoint_effector_distance_mean': (
                    float(np.mean(effector_distances)) if effector_distances else np.nan
                ),
                'goal_waypoint_rel_effector_object_distance_mean': (
                    float(np.mean(rel_effector_object_distances)) if rel_effector_object_distances else np.nan
                ),
                'goal_waypoint_gripper_open_distance_mean': (
                    float(np.mean(gripper_open_distances)) if gripper_open_distances else np.nan
                ),
                'goal_waypoint_contact_distance_mean': (
                    float(np.mean(contact_distances)) if contact_distances else np.nan
                ),
                'goal_waypoint_phase_match_mean': float(np.mean(phase_matches)) if phase_matches else np.nan,
                'goal_waypoint_holding_match_mean': (
                    float(np.mean(holding_matches)) if holding_matches else np.nan
                ),
                'future_candidate_count_mean': (
                    float(np.mean(future_candidate_counts)) if future_candidate_counts else np.nan
                ),
                'future_guard_candidate_count_mean': (
                    float(np.mean(future_guarded_candidate_counts)) if future_guarded_candidate_counts else np.nan
                ),
                'future_guard_kept_fraction_mean': (
                    float(np.mean(future_guard_kept_fractions)) if future_guard_kept_fractions else np.nan
                ),
                'future_goal_phase_match_mean': (
                    float(np.mean(future_phase_matches)) if future_phase_matches else np.nan
                ),
                'future_goal_same_segment_mean': (
                    float(np.mean(future_segment_matches)) if future_segment_matches else np.nan
                ),
                'future_waypoint_holding_mean': (
                    float(np.mean(future_waypoint_holding)) if future_waypoint_holding else np.nan
                ),
                'future_goal_holding_mean': (
                    float(np.mean(future_goal_holding)) if future_goal_holding else np.nan
                ),
                'future_goal_same_holding_segment_mean': (
                    float(np.mean(future_holding_segment_matches))
                    if future_holding_segment_matches
                    else np.nan
                ),
                'future_holding_guard_kept_fraction_mean': (
                    float(np.mean(future_holding_guard_kept_fractions))
                    if future_holding_guard_kept_fractions
                    else np.nan
                ),
            }
        )
        return augmented_goal_idxs

    def _append_guard_metric(self, values, value):
        if value is None:
            return
        value = float(value)
        if np.isnan(value):
            return
        values.append(value)

    def sample_waypoints_kmeans(self, sample_idxs, original_goal_idxs):
        # sample valid waypoint states in the same cluster as g_A
        results = []
        for sample_idx, original_goal_idx in zip(sample_idxs, original_goal_idxs):
            cluster_id = int(self.stitch_cluster_labels[int(original_goal_idx)])
            candidates = self.stitch_cluster_to_waypoint_idxs.get(cluster_id)
            results.append(self._sample_waypoint_from_candidates(sample_idx, original_goal_idx, candidates))
        return results

    def sample_waypoints(self, sample_idxs, original_goal_idxs):
        if self.stitch_retrieval_mode == 'kmeans':
            return self.sample_waypoints_kmeans(sample_idxs, original_goal_idxs)
        if self.stitch_retrieval_mode == 'knn':
            return self.sample_waypoints_knn(sample_idxs, original_goal_idxs)
        raise ValueError(f'Unknown stitch_retrieval_mode={self.stitch_retrieval_mode}.')

    def sample_waypoints_knn(self, sample_idxs, original_goal_idxs):
        if len(original_goal_idxs) == 0:
            return []

        # query nearest valid waypoint states for each g_A 
        n_neighbors = min(self.stitch_knn_k, len(self.stitch_knn_waypoint_idxs))
        _, knn_positions = self.stitch_knn.kneighbors(
            self.stitch_points[np.asarray(original_goal_idxs, dtype=np.int64)],
            n_neighbors=n_neighbors,
            return_distance=True,
        )
        results = []
        for sample_idx, original_goal_idx, neighbor_positions in zip(
            sample_idxs,
            original_goal_idxs,
            knn_positions,
        ):
            candidates = self.stitch_knn_waypoint_idxs[neighbor_positions]
            results.append(self._sample_waypoint_from_candidates(sample_idx, original_goal_idx, candidates))
        return results

    def _sample_waypoint_from_candidates(self, sample_idx, original_goal_idx, candidates):
        if candidates is None or len(candidates) == 0:
            return None

        original_candidate_count = int(len(candidates))
        original_goal_idx = int(original_goal_idx)
        if self.stitch_cross_traj_only:
            sample_traj = self.traj_ids[int(sample_idx)]
            candidates = candidates[self.traj_ids[candidates] != sample_traj]
            if len(candidates) == 0:
                return None

        guarded_candidate_count = int(len(candidates))
        for _ in range(self.stitch_guard_max_retries):
            waypoint_idx = int(candidates[np.random.randint(len(candidates))])
            if self._candidate_passes_guards(original_goal_idx, waypoint_idx):
                return self._format_waypoint_result(
                    original_goal_idx,
                    waypoint_idx,
                    original_candidate_count,
                    guarded_candidate_count,
                )

        if self._uses_any_candidate_guard() and self.stitch_guard_exhaustive_fallback:
            guard_mask = self._candidate_guard_mask(original_goal_idx, candidates)
            candidates = candidates[guard_mask]
            guarded_candidate_count = int(len(candidates))
            if guarded_candidate_count == 0:
                return None

        waypoint_idx = int(candidates[np.random.randint(len(candidates))])
        if not self._candidate_passes_guards(original_goal_idx, waypoint_idx):
            return None
        return self._format_waypoint_result(
            original_goal_idx,
            waypoint_idx,
            original_candidate_count,
            guarded_candidate_count,
        )

    def _format_waypoint_result(self, original_goal_idx, waypoint_idx, original_candidate_count, guarded_candidate_count):
        stitch_distance = float(
            np.linalg.norm(self.stitch_points[waypoint_idx] - self.stitch_points[original_goal_idx])
        )
        xy_distance = float(
            np.linalg.norm(self.stitch_xy_points[waypoint_idx] - self.stitch_xy_points[original_goal_idx])
        )
        guard_stats = self._guard_stats(original_goal_idx, waypoint_idx)
        return (
            waypoint_idx,
            stitch_distance,
            xy_distance,
            original_candidate_count,
            guarded_candidate_count,
            guard_stats,
        )

    def _uses_any_candidate_guard(self):
        return (
            (self.stitch_space == 'state' and self.stitch_state_xy_max_dist is not None)
            or self.stitch_guard_mode != 'none'
            or self.stitch_guard_xy_max_dist > 0
            or self.stitch_phase_guard_mode != 'none'
            or self.stitch_future_guard_mode == 'same_holding_segment'
        )

    def _candidate_passes_guards(self, original_goal_idx, waypoint_idx):
        # compare original future goal g_A to waypoint w_B
        if self.stitch_space == 'state' and self.stitch_state_xy_max_dist is not None:
            xy_distance = float(
                np.linalg.norm(self.stitch_xy_points[waypoint_idx] - self.stitch_xy_points[original_goal_idx])
            )
            if xy_distance > self.stitch_state_xy_max_dist:
                return False
        if self.stitch_guard_xy_max_dist > 0:
            xy_distance = float(
                np.linalg.norm(self.stitch_xy_points[waypoint_idx] - self.stitch_xy_points[original_goal_idx])
                / self.stitch_guard_position_scale
            )
            if xy_distance > self.stitch_guard_xy_max_dist:
                return False
        phase_pass = self._candidate_passes_phase_guard(original_goal_idx, waypoint_idx)
        holding_pass = self._candidate_passes_holding_guard(original_goal_idx, waypoint_idx)
        if self.stitch_guard_mode == 'none':
            return phase_pass and holding_pass
        guard_stats = self._guard_stats(original_goal_idx, waypoint_idx)
        return self._guard_stats_pass(guard_stats) and phase_pass and holding_pass

    def _candidate_guard_mask(self, original_goal_idx, candidates):
        mask = np.ones(len(candidates), dtype=bool)
        if self.stitch_space == 'state' and self.stitch_state_xy_max_dist is not None:
            xy_distances = np.linalg.norm(self.stitch_xy_points[candidates] - self.stitch_xy_points[original_goal_idx], axis=1)
            mask &= xy_distances <= self.stitch_state_xy_max_dist
        if self.stitch_guard_xy_max_dist > 0:
            xy_distances = (
                np.linalg.norm(self.stitch_xy_points[candidates] - self.stitch_xy_points[original_goal_idx], axis=1)
                / self.stitch_guard_position_scale
            )
            mask &= xy_distances <= self.stitch_guard_xy_max_dist
        if self.stitch_guard_mode != 'none':
            guard_stats = self._guard_stats(original_goal_idx, candidates)
            mask &= self._guard_stats_pass(guard_stats)
        if self.stitch_phase_guard_mode != 'none':
            mask &= self._candidate_phase_guard_mask(original_goal_idx, candidates)
        if self.stitch_future_guard_mode == 'same_holding_segment':
            mask &= self._candidate_holding_guard_mask(original_goal_idx, candidates)
        return mask

    def _compute_coarse_phase_labels(self, observations):
        if self.stitch_phase_guard_mode == 'none':
            return None
        if self.stitch_phase_guard_mode not in ('cube_simple', 'cube_delivery'):
            raise ValueError(f'Unknown stitch_phase_guard_mode={self.stitch_phase_guard_mode}.')

        # manipulation phase:
        eff = observations[:, list(self.stitch_guard_effector_dims)] / self.stitch_guard_position_scale
        obj = observations[:, list(self.stitch_guard_object_dims)] / self.stitch_guard_position_scale
        eff_obj_dist = np.linalg.norm(eff - obj, axis=-1)
        near = eff_obj_dist <= self.stitch_phase_effector_object_max_dist

        contact = np.zeros(len(observations), dtype=bool)
        if self.stitch_guard_contact_dim >= 0:
            contact = observations[:, self.stitch_guard_contact_dim] >= self.stitch_phase_contact_threshold

        lifted = np.zeros(len(observations), dtype=bool)
        if obj.shape[-1] >= 3:
            lifted = obj[:, 2] >= self.stitch_phase_lift_height

        closed = np.zeros(len(observations), dtype=bool)
        if self.stitch_guard_gripper_open_dim >= 0:
            gripper_open = (
                observations[:, self.stitch_guard_gripper_open_dim] / self.stitch_guard_gripper_open_scale
            )
            closed = gripper_open <= self.stitch_phase_gripper_open_threshold

        # 0: far/no-contact, 1: near/open/no-contact, 2: near/closed/no-contact,
        # 3: contact or lifted transport. Exact labels are less important than
        # preventing cross-phase stitches
        phase = np.zeros(len(observations), dtype=np.int8)
        phase[near] = 1
        phase[near & closed] = 2
        phase[contact | lifted] = 3
        return phase

    def _compute_phase_labels(self, observations, coarse_phase_labels, segment_ids):
        self.stitch_phase_goal_distances = None
        self.stitch_segment_goal_points = None
        if coarse_phase_labels is None or self.stitch_phase_guard_mode == 'cube_simple':
            return coarse_phase_labels
        if self.stitch_phase_guard_mode != 'cube_delivery':
            raise ValueError(f'Unknown stitch_phase_guard_mode={self.stitch_phase_guard_mode}.')

        # delivery-aware phase uses the end of each pick-place segment as a
        # local target estimate. This lets us split the process
        obj = observations[:, list(self.stitch_guard_object_dims)] / self.stitch_guard_position_scale
        self.stitch_segment_goal_points = self._compute_segment_goal_points(obj, segment_ids)
        self.stitch_phase_goal_distances = np.linalg.norm(
            obj - self.stitch_segment_goal_points[segment_ids],
            axis=-1,
        )

        phase = np.array(coarse_phase_labels, copy=True)
        transport = coarse_phase_labels == self.stitch_segment_phase_label
        near_goal = self.stitch_phase_goal_distances <= self.stitch_phase_delivery_near_goal_dist
        at_goal = self.stitch_phase_goal_distances <= self.stitch_phase_delivery_place_goal_dist


        # 3: transporting far from the local segment target
        # 4: transporting near the target
        # 5: place / release / stabilize after transport reached the target
        phase[transport & near_goal] = 4
        phase[transport & at_goal] = 5
        stabilize = self._states_after_transport(
            coarse_phase_labels,
            self.stitch_phase_delivery_stabilize_steps,
        )
        phase[stabilize & at_goal] = 5

        self.stitch_summary.update(
            {
                'phase_delivery_goal_distance_mean': float(np.mean(self.stitch_phase_goal_distances)),
                'phase_delivery_goal_distance_median': float(np.median(self.stitch_phase_goal_distances)),
                'phase_delivery_near_goal_fraction': float(np.mean(near_goal)),
                'phase_delivery_at_goal_fraction': float(np.mean(at_goal)),
            }
        )
        return phase

    def _compute_holding_segments(self, observations):
        if self.stitch_future_guard_mode != 'same_holding_segment':
            return None, None

        # the gripper must be near and closed
        # and the cube must either be contacted or lifted
        eff = observations[:, list(self.stitch_guard_effector_dims)] / self.stitch_guard_position_scale
        obj = observations[:, list(self.stitch_guard_object_dims)] / self.stitch_guard_position_scale
        near = np.linalg.norm(eff - obj, axis=-1) <= self.stitch_holding_effector_object_max_dist

        contact = np.zeros(len(observations), dtype=bool)
        if self.stitch_guard_contact_dim >= 0:
            contact = observations[:, self.stitch_guard_contact_dim] >= self.stitch_holding_contact_threshold

        lifted = np.zeros(len(observations), dtype=bool)
        if obj.shape[-1] >= 3:
            lifted = obj[:, 2] >= self.stitch_holding_lift_height

        closed = np.ones(len(observations), dtype=bool)
        if self.stitch_guard_gripper_open_dim >= 0:
            gripper_closure = (
                observations[:, self.stitch_guard_gripper_open_dim] / self.stitch_guard_gripper_open_scale
            )
            
            closed = gripper_closure >= self.stitch_holding_gripper_closed_threshold

        raw_holding = near & closed & (contact | lifted)
        holding = np.zeros(len(observations), dtype=bool)
        holding_segment_ids = np.full(len(observations), -1, dtype=np.int32)
        next_segment_id = 0

        for initial_idx, terminal_idx in zip(self.initial_locs, self.terminal_locs):
            start = int(initial_idx)
            end = int(terminal_idx) + 1
            traj_holding = raw_holding[start:end]
            run_starts = np.flatnonzero(traj_holding & np.concatenate([[True], ~traj_holding[:-1]]))
            run_ends = np.flatnonzero(traj_holding & np.concatenate([~traj_holding[1:], [True]])) + 1
            for run_start, run_end in zip(run_starts, run_ends):
                if run_end - run_start < self.stitch_holding_min_segment_steps:
                    continue
                global_start = start + int(run_start)
                global_end = start + int(run_end)
                holding[global_start:global_end] = True
                holding_segment_ids[global_start:global_end] = next_segment_id
                next_segment_id += 1

        return holding, holding_segment_ids

    def _compute_segment_goal_points(self, obj, segment_ids):
        if segment_ids is None or len(segment_ids) == 0:
            return None

        # each segment approximates one pick-place attempt 
        segment_goal_points = np.empty((int(np.max(segment_ids)) + 1, obj.shape[-1]), dtype=np.float32)
        starts = np.concatenate([[0], np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1])
        ends = np.concatenate([starts[1:], [len(segment_ids)]])
        for start, end in zip(starts, ends):
            tail_start = max(int(start), int(end) - self.stitch_phase_delivery_target_tail)
            segment_goal_points[int(segment_ids[start])] = np.median(obj[tail_start:end], axis=0)
        return segment_goal_points

    def _states_after_transport(self, coarse_phase_labels, max_steps):
        after_transport = np.zeros(len(coarse_phase_labels), dtype=bool)
        for initial_idx, terminal_idx in zip(self.initial_locs, self.terminal_locs):
            steps_after_transport = max_steps + 1
            for idx in range(int(initial_idx), int(terminal_idx) + 1):
                if coarse_phase_labels[idx] == self.stitch_segment_phase_label:
                    steps_after_transport = 0
                else:
                    steps_after_transport += 1
                after_transport[idx] = 0 < steps_after_transport <= max_steps
        return after_transport

    def _compute_phase_segment_ids(self, phase_labels):
        if phase_labels is None:
            return None

        # split long play trajectories into repeated pick-place segments
        segment_ids = np.empty(self.size, dtype=np.int32)
        next_segment_id = 0
        for initial_idx, terminal_idx in zip(self.initial_locs, self.terminal_locs):
            segment_id = next_segment_id
            next_segment_id += 1
            seen_transport = False
            released_after_transport = False

            for idx in range(int(initial_idx), int(terminal_idx) + 1):
                phase = int(phase_labels[idx])
                if phase == self.stitch_segment_phase_label:
                    if seen_transport and released_after_transport:
                        segment_id = next_segment_id
                        next_segment_id += 1
                        released_after_transport = False
                    seen_transport = True
                elif seen_transport:
                    released_after_transport = True

                segment_ids[idx] = segment_id

        return segment_ids

    def _candidate_passes_phase_guard(self, original_goal_idx, waypoint_idx):
        if self.stitch_phase_guard_mode == 'none':
            return True
        return bool(self.stitch_phase_labels[int(original_goal_idx)] == self.stitch_phase_labels[int(waypoint_idx)])

    def _candidate_phase_guard_mask(self, original_goal_idx, candidates):
        if self.stitch_phase_guard_mode == 'none':
            return np.ones(len(candidates), dtype=bool)
        return self.stitch_phase_labels[candidates] == self.stitch_phase_labels[int(original_goal_idx)]

    def _candidate_passes_holding_guard(self, original_goal_idx, waypoint_idx):
        if self.stitch_future_guard_mode != 'same_holding_segment':
            return True
        original_goal_idx = int(original_goal_idx)
        waypoint_idx = int(waypoint_idx)
        original_holding = bool(self.stitch_holding_mask[original_goal_idx])
        waypoint_holding = bool(self.stitch_holding_mask[waypoint_idx])
        if self.stitch_phase_labels[original_goal_idx] == self.stitch_segment_phase_label:
            return original_holding and waypoint_holding
        return original_holding == waypoint_holding

    def _candidate_holding_guard_mask(self, original_goal_idx, candidates):
        if self.stitch_future_guard_mode != 'same_holding_segment':
            return np.ones(len(candidates), dtype=bool)
        original_goal_idx = int(original_goal_idx)
        original_holding = bool(self.stitch_holding_mask[original_goal_idx])
        if self.stitch_phase_labels[original_goal_idx] == self.stitch_segment_phase_label:
            return self.stitch_holding_mask[candidates]
        return self.stitch_holding_mask[candidates] == original_holding

    def _sample_new_goal_idx(self, original_goal_idx, waypoint_idx, min_goal_idx, max_goal_idx):
        candidates = np.arange(int(min_goal_idx), int(max_goal_idx) + 1, dtype=np.int64)
        stats = {
            'future_candidate_count': int(len(candidates)),
            'future_guard_candidate_count': int(len(candidates)),
            'future_guard_kept_fraction': 1.0,
            'future_goal_phase_match': np.nan,
            'future_goal_same_segment': np.nan,
            'waypoint_holding': np.nan,
            'future_goal_holding': np.nan,
            'future_goal_same_holding_segment': np.nan,
            'future_holding_guard_kept_fraction': np.nan,
        }

        if self.stitch_future_guard_mode == 'same_phase':
            # future g'_B must remain in the same phase as g_A
            original_phase = self.stitch_phase_labels[int(original_goal_idx)]
            candidates = candidates[self.stitch_phase_labels[candidates] == original_phase]
        elif self.stitch_future_guard_mode == 'same_segment':
            # allow phase changes inside one pick-place attempt
            waypoint_segment = self.stitch_segment_ids[int(waypoint_idx)]
            candidates = candidates[self.stitch_segment_ids[candidates] == waypoint_segment]
        elif self.stitch_future_guard_mode == 'same_holding_segment':
            waypoint_segment = self.stitch_segment_ids[int(waypoint_idx)]
            candidates = candidates[self.stitch_segment_ids[candidates] == waypoint_segment]
            waypoint_holding_segment = self.stitch_holding_segment_ids[int(waypoint_idx)]
            stats['waypoint_holding'] = float(waypoint_holding_segment >= 0)
            if waypoint_holding_segment >= 0:
                same_segment_candidate_count = len(candidates)
                candidates = candidates[
                    self.stitch_holding_segment_ids[candidates] == waypoint_holding_segment
                ]
                stats['future_holding_guard_kept_fraction'] = (
                    len(candidates) / same_segment_candidate_count
                    if same_segment_candidate_count > 0
                    else 0.0
                )

        stats['future_guard_candidate_count'] = int(len(candidates))
        stats['future_guard_kept_fraction'] = (
            stats['future_guard_candidate_count'] / stats['future_candidate_count']
            if stats['future_candidate_count'] > 0
            else 0.0
        )
        if len(candidates) == 0:
            return None, stats

        new_goal_idx = int(candidates[np.random.randint(len(candidates))])
        if self.stitch_phase_labels is not None:
            stats['future_goal_phase_match'] = float(
                self.stitch_phase_labels[new_goal_idx] == self.stitch_phase_labels[int(original_goal_idx)]
            )
        if self.stitch_segment_ids is not None:
            stats['future_goal_same_segment'] = float(
                self.stitch_segment_ids[new_goal_idx] == self.stitch_segment_ids[int(waypoint_idx)]
            )
        if self.stitch_holding_mask is not None:
            stats['future_goal_holding'] = float(self.stitch_holding_mask[new_goal_idx])
            waypoint_holding_segment = self.stitch_holding_segment_ids[int(waypoint_idx)]
            if waypoint_holding_segment >= 0:
                stats['future_goal_same_holding_segment'] = float(
                    self.stitch_holding_segment_ids[new_goal_idx] == waypoint_holding_segment
                )
        return new_goal_idx, stats

    def _guard_stats(self, original_goal_idx, waypoint_idx_or_idxs):
        if self.stitch_guard_mode == 'none':
            stats = {
                'effector_distance': np.nan,
                'rel_effector_object_distance': np.nan,
                'gripper_open_distance': np.nan,
                'contact_distance': np.nan,
            }
            stats.update(self._phase_stats(original_goal_idx, waypoint_idx_or_idxs))
            return stats

        obs_goal = self.stitch_observations[original_goal_idx]
        obs_waypoint = self.stitch_observations[waypoint_idx_or_idxs]
        stats = {
            'effector_distance': np.nan,
            'rel_effector_object_distance': np.nan,
            'gripper_open_distance': np.nan,
            'contact_distance': np.nan,
        }

        if len(self.stitch_guard_effector_dims) > 0:
            goal_eff = obs_goal[list(self.stitch_guard_effector_dims)]
            waypoint_eff = obs_waypoint[..., list(self.stitch_guard_effector_dims)]
            stats['effector_distance'] = (
                np.linalg.norm(waypoint_eff - goal_eff, axis=-1) / self.stitch_guard_position_scale
            )

        if (
            self.stitch_guard_rel_effector_object_max_dist > 0
            and len(self.stitch_guard_effector_dims) > 0
            and len(self.stitch_guard_object_dims) > 0
        ):
            goal_eff = obs_goal[list(self.stitch_guard_effector_dims)]
            waypoint_eff = obs_waypoint[..., list(self.stitch_guard_effector_dims)]
            goal_obj = obs_goal[list(self.stitch_guard_object_dims)]
            waypoint_obj = obs_waypoint[..., list(self.stitch_guard_object_dims)]
            stats['rel_effector_object_distance'] = (
                np.linalg.norm((waypoint_eff - waypoint_obj) - (goal_eff - goal_obj), axis=-1)
                / self.stitch_guard_position_scale
            )

        if self.stitch_guard_gripper_open_dim >= 0:
            stats['gripper_open_distance'] = (
                np.abs(obs_waypoint[..., self.stitch_guard_gripper_open_dim] - obs_goal[self.stitch_guard_gripper_open_dim])
                / self.stitch_guard_gripper_open_scale
            )

        if self.stitch_guard_contact_dim >= 0:
            stats['contact_distance'] = np.abs(
                obs_waypoint[..., self.stitch_guard_contact_dim] - obs_goal[self.stitch_guard_contact_dim]
            )

        stats.update(self._phase_stats(original_goal_idx, waypoint_idx_or_idxs))
        return stats

    def _phase_stats(self, original_goal_idx, waypoint_idx_or_idxs):
        if self.stitch_phase_guard_mode == 'none':
            return {
                'phase_match': np.nan,
                'original_goal_phase': np.nan,
                'waypoint_phase': np.nan,
                'holding_match': np.nan,
                'original_goal_holding': np.nan,
                'waypoint_holding': np.nan,
            }
        original_goal_phase = int(self.stitch_phase_labels[int(original_goal_idx)])
        waypoint_phase = self.stitch_phase_labels[waypoint_idx_or_idxs]
        if self.stitch_holding_mask is None:
            holding_match = np.nan
            original_goal_holding = np.nan
            waypoint_holding = np.nan
        else:
            original_goal_holding = bool(self.stitch_holding_mask[int(original_goal_idx)])
            waypoint_holding = self.stitch_holding_mask[waypoint_idx_or_idxs]
            holding_match = waypoint_holding == original_goal_holding
        return {
            'phase_match': waypoint_phase == original_goal_phase,
            'original_goal_phase': original_goal_phase,
            'waypoint_phase': waypoint_phase,
            'holding_match': holding_match,
            'original_goal_holding': original_goal_holding,
            'waypoint_holding': waypoint_holding,
        }

    def _guard_stats_pass(self, guard_stats):
        # Apply only enabled thresholds. Negative threshold values disable the
        # corresponding guard so configs can isolate one source of mismatch.
        mask = True
        if self.stitch_guard_effector_max_dist > 0:
            mask = mask & (guard_stats['effector_distance'] <= self.stitch_guard_effector_max_dist)
        if self.stitch_guard_rel_effector_object_max_dist > 0:
            mask = mask & (
                guard_stats['rel_effector_object_distance'] <= self.stitch_guard_rel_effector_object_max_dist
            )
        if self.stitch_guard_gripper_open_max_diff > 0:
            mask = mask & (guard_stats['gripper_open_distance'] <= self.stitch_guard_gripper_open_max_diff)
        if self.stitch_guard_contact_max_diff >= 0:
            mask = mask & (guard_stats['contact_distance'] <= self.stitch_guard_contact_max_diff)
        return mask

    def _maybe_record_stitch_debug(
        self,
        batch_pos,
        sample_idx,
        original_goal_idx,
        waypoint_idx,
        new_goal_idx,
        stitch_distance,
        xy_distance,
        candidate_count,
        guarded_candidate_count,
        guard_stats,
        future_stats,
    ):
        if self.stitch_debug_samples <= 0 or len(self._stitch_debug_records) >= self.stitch_debug_samples:
            return

        sample_xy = self.stitch_xy_points[sample_idx]
        original_goal_xy = self.stitch_xy_points[original_goal_idx]
        waypoint_xy = self.stitch_xy_points[waypoint_idx]
        new_goal_xy = self.stitch_xy_points[new_goal_idx]
        sample_traj = int(self.traj_ids[sample_idx])
        waypoint_traj = int(self.traj_ids[waypoint_idx])
        self._stitch_debug_records.append(
            {
                'sample_call': int(self._stitch_sample_calls),
                'batch_pos': int(batch_pos),
                'sample_idx': int(sample_idx),
                'sample_traj': sample_traj,
                'original_goal_idx': int(original_goal_idx),
                'original_goal_traj': int(self.traj_ids[original_goal_idx]),
                'waypoint_idx': int(waypoint_idx),
                'waypoint_traj': waypoint_traj,
                'new_goal_idx': int(new_goal_idx),
                'new_goal_traj': int(self.traj_ids[new_goal_idx]),
                'sample_xy_0': float(sample_xy[0]),
                'sample_xy_1': float(sample_xy[1]),
                'original_goal_xy_0': float(original_goal_xy[0]),
                'original_goal_xy_1': float(original_goal_xy[1]),
                'waypoint_xy_0': float(waypoint_xy[0]),
                'waypoint_xy_1': float(waypoint_xy[1]),
                'new_goal_xy_0': float(new_goal_xy[0]),
                'new_goal_xy_1': float(new_goal_xy[1]),
                'goal_waypoint_stitch_distance': float(stitch_distance),
                'goal_waypoint_xy_distance': float(xy_distance),
                'candidate_count': int(candidate_count),
                'candidate_count_after_xy_guard': int(guarded_candidate_count),
                'candidate_count_after_guard': int(guarded_candidate_count),
                'stitch_space': self.stitch_space,
                'stitch_retrieval_mode': self.stitch_retrieval_mode,
                'stitch_state_normalize': int(self.stitch_state_normalize),
                'stitch_state_xy_weight': float(self.stitch_state_xy_weight),
                'stitch_state_xy_max_dist': (
                    float(self.stitch_state_xy_max_dist)
                    if self.stitch_state_xy_max_dist is not None
                    else np.nan
                ),
                'stitch_guard_mode': self.stitch_guard_mode,
                'stitch_guard_xy_max_dist': float(self.stitch_guard_xy_max_dist),
                'goal_waypoint_effector_distance': float(guard_stats.get('effector_distance', np.nan)),
                'goal_waypoint_rel_effector_object_distance': float(
                    guard_stats.get('rel_effector_object_distance', np.nan)
                ),
                'goal_waypoint_gripper_open_distance': float(
                    guard_stats.get('gripper_open_distance', np.nan)
                ),
                'goal_waypoint_contact_distance': float(guard_stats.get('contact_distance', np.nan)),
                'goal_waypoint_phase_match': float(guard_stats.get('phase_match', np.nan)),
                'original_goal_phase': float(guard_stats.get('original_goal_phase', np.nan)),
                'waypoint_phase': float(guard_stats.get('waypoint_phase', np.nan)),
                'goal_waypoint_holding_match': float(guard_stats.get('holding_match', np.nan)),
                'original_goal_holding': float(guard_stats.get('original_goal_holding', np.nan)),
                'waypoint_holding': float(guard_stats.get('waypoint_holding', np.nan)),
                'new_goal_phase': (
                    float(self.stitch_phase_labels[new_goal_idx])
                    if self.stitch_phase_labels is not None
                    else np.nan
                ),
                'original_goal_phase_goal_distance': (
                    float(self.stitch_phase_goal_distances[original_goal_idx])
                    if self.stitch_phase_goal_distances is not None
                    else np.nan
                ),
                'waypoint_phase_goal_distance': (
                    float(self.stitch_phase_goal_distances[waypoint_idx])
                    if self.stitch_phase_goal_distances is not None
                    else np.nan
                ),
                'new_goal_phase_goal_distance': (
                    float(self.stitch_phase_goal_distances[new_goal_idx])
                    if self.stitch_phase_goal_distances is not None
                    else np.nan
                ),
                'original_goal_segment': (
                    int(self.stitch_segment_ids[original_goal_idx])
                    if self.stitch_segment_ids is not None
                    else -1
                ),
                'waypoint_segment': (
                    int(self.stitch_segment_ids[waypoint_idx])
                    if self.stitch_segment_ids is not None
                    else -1
                ),
                'new_goal_segment': (
                    int(self.stitch_segment_ids[new_goal_idx])
                    if self.stitch_segment_ids is not None
                    else -1
                ),
                'stitch_future_guard_mode': self.stitch_future_guard_mode,
                'future_candidate_count': int(future_stats.get('future_candidate_count', 0)),
                'future_guard_candidate_count': int(future_stats.get('future_guard_candidate_count', 0)),
                'future_guard_kept_fraction': float(future_stats.get('future_guard_kept_fraction', np.nan)),
                'future_goal_phase_match': float(future_stats.get('future_goal_phase_match', np.nan)),
                'future_goal_same_segment': float(future_stats.get('future_goal_same_segment', np.nan)),
                'future_waypoint_holding': float(future_stats.get('waypoint_holding', np.nan)),
                'future_goal_holding': float(future_stats.get('future_goal_holding', np.nan)),
                'future_goal_same_holding_segment': float(
                    future_stats.get('future_goal_same_holding_segment', np.nan)
                ),
                'future_holding_guard_kept_fraction': float(
                    future_stats.get('future_holding_guard_kept_fraction', np.nan)
                ),
                'future_offset': int(new_goal_idx - waypoint_idx),
                'stitch_future_max': -1 if self.stitch_future_max is None else int(self.stitch_future_max),
                'cross_trajectory': int(sample_traj != waypoint_traj),
            }
        )

    def _record_stitch_metrics(self, metrics):
        self._stitch_metric_count += 1
        for key, value in metrics.items():
            if key not in self._stitch_metric_sums:
                self._stitch_metric_sums[key] = 0.0
            if not np.isnan(value):
                self._stitch_metric_sums[key] += float(value)

    def get_and_reset_diagnostics(self):
        # return averaged temporal stitching diagnostics since the previous reset
        if self._stitch_metric_count == 0:
            return {}
        metrics = {
            f'stitch/{key}': value / self._stitch_metric_count for key, value in self._stitch_metric_sums.items()
        }
        metrics['stitch/num_metric_batches'] = self._stitch_metric_count
        self._stitch_metric_sums = {}
        self._stitch_metric_count = 0
        return metrics

    def get_and_reset_debug_records(self):
        # return sampled temporal stitching debug records and clear the buffer
        records = self._stitch_debug_records
        self._stitch_debug_records = []
        return records
