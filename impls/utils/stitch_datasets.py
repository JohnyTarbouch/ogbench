from __future__ import annotations

import dataclasses

import jax
import numpy as np

from utils.datasets import GCDataset


@dataclasses.dataclass
class TemporalStitchGCDataset(GCDataset):
    """
    Goal-conditioned dataset with temporal stitching augmentation.

    GCDataset sample goals a future state of the same trajectory as goal.
    With probability stitch_p_aug, replaces the sampled goal with a future goal from
    another trajectory that passes through the same state group as the original goal.

    Additional config keys:
    - stitch_p_aug: Probability of trying to replace goal
    - stitch_space: k-means cluster space. state matches the original code
    - stitch_xy_dims: Observation dim used for XY matching
    - stitch_future_min: Minimum future offset after the waypoint for the augmented goal
    - stitch_future_max: Maximum future offset after the waypoint for the augmented goal. Negative disables it
    - stitch_cross_traj_only: if waypoints must come from a different trajectory
    - stitch_nclusters: Number of k-means clusters
    - stitch_kmeans_n_init: Number of k-means init
    - stitch_kmeans_random_state: random state for reproducible k-mean
    - stitch_state_normalize: State-only , z-score full-state features before K-means
    - stitch_state_normalize_eps: Minimum standard deviation used for state normalization
    - stitch_state_xy_weight: upweight XY dim in the K-means feature space
    - stitch_state_xy_max_dist: positive values reject waypoint candidates farther than this in true XY
    - stitch_guard_mode: manipulation guard 
    """

    def __post_init__(self):
        super().__post_init__()
        self._init_stitching()

    def _config_get(self, key, default):
        return self.config[key] if key in self.config else default

    def _init_stitching(self):
        self.stitch_p_aug = float(self._config_get('stitch_p_aug', 0.0))
        self.stitch_space = self._config_get('stitch_space', 'state')
        self.stitch_xy_dims = tuple(self._config_get('stitch_xy_dims', (0, 1)))
        self.stitch_future_min = int(self._config_get('stitch_future_min', 0))
        stitch_future_max = int(self._config_get('stitch_future_max', -1))
        self.stitch_future_max = None if stitch_future_max < 0 else stitch_future_max
        self.stitch_cross_traj_only = bool(self._config_get('stitch_cross_traj_only', False))
        self.stitch_nclusters = int(self._config_get('stitch_nclusters', 40))
        self.stitch_kmeans_n_init = self._config_get('stitch_kmeans_n_init', 'auto')
        self.stitch_kmeans_random_state = self._config_get('stitch_kmeans_random_state', None)
        self.stitch_debug_samples = int(self._config_get('stitch_debug_samples', 0))
        self.stitch_state_normalize = bool(self._config_get('stitch_state_normalize', False))
        self.stitch_state_normalize_eps = float(self._config_get('stitch_state_normalize_eps', 1e-6))
        self.stitch_state_xy_weight = float(self._config_get('stitch_state_xy_weight', 1.0))
        self.stitch_state_xy_max_dist = float(self._config_get('stitch_state_xy_max_dist', -1.0))
        if self.stitch_state_xy_max_dist <= 0:
            self.stitch_state_xy_max_dist = None
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
        self.stitch_enabled = self.stitch_p_aug > 0
        self._stitch_metric_sums = {}
        self._stitch_metric_count = 0
        self._stitch_sample_calls = 0
        self._stitch_debug_records = []

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
            'stitch_nclusters': self.stitch_nclusters,
            'stitch_kmeans_n_init': self.stitch_kmeans_n_init,
            'stitch_kmeans_random_state': self.stitch_kmeans_random_state,
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
        }

        if not self.stitch_enabled:
            self.stitch_points = None
            self.stitch_xy_points = None
            self.stitch_cluster_labels = None
            self.stitch_cluster_to_waypoint_idxs = {}
            self.stitch_summary.update({'num_waypoints': 0, 'num_groups': 0})
            return

        if self.stitch_space not in ('state', 'xy'):
            raise ValueError("stitch_space must be one of {'state', 'xy'}.")
        if self.stitch_nclusters <= 0:
            raise ValueError('stitch_nclusters must be positive.')
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
        if self.stitch_guard_max_retries <= 0:
            raise ValueError('stitch_guard_max_retries must be positive.')
        if self.stitch_guard_position_scale <= 0:
            raise ValueError('stitch_guard_position_scale must be positive.')
        if self.stitch_guard_gripper_open_scale <= 0:
            raise ValueError('stitch_guard_gripper_open_scale must be positive.')
        observations = self.dataset['observations']
        if not isinstance(observations, np.ndarray) or observations.ndim != 2:
            raise ValueError('TemporalStitchGCDataset expects state observations as a 2D numpy array.')
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

        self.stitch_xy_points = observations[:, list(self.stitch_xy_dims)].astype(np.float32)
        self.stitch_observations = observations.astype(np.float32)
        if self.stitch_space == 'state':
            self.stitch_points = observations.astype(np.float32)
            if self.stitch_state_normalize:
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

        valid_waypoint_idxs = self.dataset.valid_idxs if hasattr(self.dataset, 'valid_idxs') else np.arange(self.size)
        waypoint_terminal_idxs = self.terminal_locs[self.traj_ids[valid_waypoint_idxs]]
        valid_waypoint_idxs = valid_waypoint_idxs[
            valid_waypoint_idxs + self.stitch_future_min <= waypoint_terminal_idxs
        ]
        self.valid_waypoint_idxs = valid_waypoint_idxs.astype(np.int64)

        self.stitch_cluster_labels = None
        self.stitch_cluster_to_waypoint_idxs = {}
        self._build_kmeans_groups()

    def _build_kmeans_groups(self):
        # cluster all states once
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "TemporalStitchGCDataset requires scikit-learn. Install it with "
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

    def sample(self, batch_size, idxs=None, evaluation=False):
        # sample a batch -> replace actor goals using temporal stitching
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
        # replace actor goals with temporally stitched goals
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

        if len(attempt_positions) > 0:
            waypoint_infos = self.sample_waypoints_kmeans(
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

                new_goal_idx = int(np.random.randint(min_goal_idx, max_goal_idx + 1))
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
            }
        )
        return augmented_goal_idxs

    def _append_guard_metric(self, values, value):
        if value is None:
            return
        if np.isnan(value):
            return
        values.append(float(value))

    def sample_waypoints_kmeans(self, sample_idxs, original_goal_idxs):
        # sample waypoint states from the same precomputed k-means cluster as the original goal
        results = []
        for sample_idx, original_goal_idx in zip(sample_idxs, original_goal_idxs):
            cluster_id = int(self.stitch_cluster_labels[int(original_goal_idx)])
            candidates = self.stitch_cluster_to_waypoint_idxs.get(cluster_id)
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
        return (self.stitch_space == 'state' and self.stitch_state_xy_max_dist is not None) or (
            self.stitch_guard_mode != 'none'
            or self.stitch_guard_xy_max_dist > 0
        )

    def _candidate_passes_guards(self, original_goal_idx, waypoint_idx):
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
        if self.stitch_guard_mode == 'none':
            return True
        guard_stats = self._guard_stats(original_goal_idx, waypoint_idx)
        return self._guard_stats_pass(guard_stats)

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
        return mask

    def _guard_stats(self, original_goal_idx, waypoint_idx_or_idxs):
        if self.stitch_guard_mode == 'none':
            return {
                'effector_distance': np.nan,
                'rel_effector_object_distance': np.nan,
                'gripper_open_distance': np.nan,
                'contact_distance': np.nan,
            }

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

        return stats

    def _guard_stats_pass(self, guard_stats):
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
