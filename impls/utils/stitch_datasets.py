"""Temporal stitching dataset extensions for OGBench GCBC experiments.

This module keeps the temporal stitching/relabeling implementation separate from
OGBench's original dataset utilities. Plain OGBench runs continue to use
``utils.datasets.GCDataset``; experiments opt into this class with
``agent.dataset_class=TemporalStitchGCDataset``.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np

from utils.datasets import GCDataset


@dataclasses.dataclass
class TemporalStitchGCDataset(GCDataset):
    """Goal-conditioned dataset with temporal stitching augmentation.

    The default GCDataset samples actor goals from the future of the same trajectory.
    This subclass keeps that behavior and, with probability stitch_p_aug,
    replaces the sampled actor goal with a future goal from another trajectory
    that passes through the same precomputed state group as the original actor goal.

    Additional config keys:
    - stitch_p_aug: Probability of trying to replace goal
    - stitch_radius: XY bucket size when stitch_mode is bucket.
    - stitch_space: K-means cluster space. state matches the original code, xy is an ablation.
    - stitch_xy_dims: Observation dimensions used for XY matching and debug logging.
    - stitch_future_min: Minimum future offset after the waypoint for the augmented goal.
    - stitch_cross_traj_only: Whether waypoints must come from a different trajectory.
    - stitch_mode: Waypoint grouping mode. kmeans follows the paper, 'bucket' is a fast ablation.
    - stitch_nclusters: Number of k-means clusters.
    - stitch_kmeans_n_init: Number of k-means initializations.
    - stitch_kmeans_random_state: Random state for reproducible k-means clusters.
    - stitch_state_normalize: State-only ablation; z-score full-state features before K-means.
    - stitch_state_normalize_eps: Minimum standard deviation used for state normalization.
    - stitch_state_xy_weight: State-only ablation; upweight XY dimensions in the K-means feature space.
    - stitch_state_xy_max_dist: State-only ablation; positive values reject waypoint candidates farther than this in true XY.
    """

    def __post_init__(self):
        super().__post_init__()
        self._init_stitching()

    def _config_get(self, key, default):
        return self.config[key] if key in self.config else default

    def _init_stitching(self):
        self.stitch_p_aug = float(self._config_get('stitch_p_aug', 0.0))
        self.stitch_radius = float(self._config_get('stitch_radius', 0.5))
        self.stitch_space = self._config_get('stitch_space', 'state')
        self.stitch_xy_dims = tuple(self._config_get('stitch_xy_dims', (0, 1)))
        self.stitch_future_min = int(self._config_get('stitch_future_min', 0))
        self.stitch_cross_traj_only = bool(self._config_get('stitch_cross_traj_only', False))
        self.stitch_mode = self._config_get('stitch_mode', 'kmeans')
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
            'stitch_radius': self.stitch_radius,
            'stitch_space': self.stitch_space,
            'stitch_xy_dims': list(self.stitch_xy_dims),
            'stitch_future_min': self.stitch_future_min,
            'stitch_cross_traj_only': self.stitch_cross_traj_only,
            'stitch_mode': self.stitch_mode,
            'stitch_nclusters': self.stitch_nclusters,
            'stitch_kmeans_n_init': self.stitch_kmeans_n_init,
            'stitch_kmeans_random_state': self.stitch_kmeans_random_state,
            'stitch_debug_samples': self.stitch_debug_samples,
            'stitch_state_normalize': self.stitch_state_normalize,
            'stitch_state_normalize_eps': self.stitch_state_normalize_eps,
            'stitch_state_xy_weight': self.stitch_state_xy_weight,
            'stitch_state_xy_max_dist': self.stitch_state_xy_max_dist,
        }

        if not self.stitch_enabled:
            self.stitch_points = None
            self.stitch_xy_points = None
            self.stitch_buckets = {}
            self.stitch_cluster_labels = None
            self.stitch_cluster_to_waypoint_idxs = {}
            self.stitch_summary.update({'num_waypoints': 0, 'num_groups': 0})
            return

        if self.stitch_space not in ('state', 'xy'):
            raise ValueError("stitch_space must be one of {'state', 'xy'}.")
        if self.stitch_mode not in ('kmeans', 'bucket'):
            raise ValueError("stitch_mode must be one of {'kmeans', 'bucket'}.")
        if self.stitch_mode == 'bucket' and self.stitch_space != 'xy':
            raise ValueError("stitch_mode='bucket' requires stitch_space='xy'.")
        if self.stitch_mode == 'bucket' and self.stitch_radius <= 0:
            raise ValueError("stitch_radius must be positive when stitch_mode='bucket'.")
        if self.stitch_mode == 'kmeans' and self.stitch_nclusters <= 0:
            raise ValueError("stitch_nclusters must be positive when stitch_mode='kmeans'.")
        if self.stitch_state_normalize_eps <= 0:
            raise ValueError('stitch_state_normalize_eps must be positive.')
        if self.stitch_state_xy_weight <= 0:
            raise ValueError('stitch_state_xy_weight must be positive.')
        observations = self.dataset['observations']
        if not isinstance(observations, np.ndarray) or observations.ndim != 2:
            raise ValueError('TemporalStitchGCDataset expects state observations as a 2D numpy array.')
        if max(self.stitch_xy_dims) >= observations.shape[-1]:
            raise ValueError(
                f'stitch_xy_dims={self.stitch_xy_dims} incompatible with observation shape {observations.shape}.'
            )

        self.stitch_xy_points = observations[:, list(self.stitch_xy_dims)].astype(np.float32)
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

        self.stitch_buckets = {}
        self.stitch_cluster_labels = None
        self.stitch_cluster_to_waypoint_idxs = {}
        if self.stitch_mode == 'kmeans':
            self._build_kmeans_groups()
        else:
            self._build_bucket_groups()

    def _build_kmeans_groups(self):
        # Cluster all states once, as in the paper.
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "stitch_mode='kmeans' requires scikit-learn. Install it with "
                "`python -m pip install scikit-learn` or reinstall OGBench with the train extra."
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

    def _build_bucket_groups(self):
        # Assign states to XY grid buckets once (not the paper implementation)
        self.stitch_buckets = {}
        bucket_coords = np.floor(self.stitch_points[self.valid_waypoint_idxs] / self.stitch_radius).astype(np.int64)
        for idx, bucket_coord in zip(self.valid_waypoint_idxs, bucket_coords):
            self.stitch_buckets.setdefault(tuple(bucket_coord), []).append(int(idx))
        self.stitch_buckets = {key: np.asarray(value, dtype=np.int64) for key, value in self.stitch_buckets.items()}

        bucket_sizes = np.array([len(v) for v in self.stitch_buckets.values()], dtype=np.int64)
        self.stitch_summary.update(
            {
                'num_waypoints': int(len(self.valid_waypoint_idxs)),
                'num_groups': int(len(self.stitch_buckets)),
                'group_size_min': int(np.min(bucket_sizes)) if len(bucket_sizes) else 0,
                'group_size_mean': float(np.mean(bucket_sizes)) if len(bucket_sizes) else 0.0,
                'group_size_max': int(np.max(bucket_sizes)) if len(bucket_sizes) else 0,
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

        if len(attempt_positions) > 0:
            if self.stitch_mode == 'kmeans':
                waypoint_infos = self.sample_waypoints_kmeans(
                    idxs[attempt_positions], actor_goal_idxs[attempt_positions]
                )
            else:
                waypoint_infos = self.sample_waypoints_bucket(
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
                ) = waypoint_info
                waypoint_traj = self.traj_ids[waypoint_idx]
                terminal_idx = int(self.terminal_locs[waypoint_traj])
                min_goal_idx = waypoint_idx + self.stitch_future_min
                if min_goal_idx > terminal_idx:
                    continue

                new_goal_idx = int(np.random.randint(min_goal_idx, terminal_idx + 1))
                augmented_goal_idxs[batch_pos] = new_goal_idx
                accepted += 1
                candidate_counts.append(candidate_count)
                guarded_candidate_counts.append(guarded_candidate_count)
                stitch_distances.append(stitch_distance)
                xy_distances.append(xy_distance)
                future_offsets.append(new_goal_idx - waypoint_idx)
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
            }
        )
        return augmented_goal_idxs

    def sample_waypoints_kmeans(self, sample_idxs, original_goal_idxs):
        # sample waypoint states from the same precomputed k-means cluster as the original goal
        results = []
        for sample_idx, original_goal_idx in zip(sample_idxs, original_goal_idxs):
            cluster_id = int(self.stitch_cluster_labels[int(original_goal_idx)])
            candidates = self.stitch_cluster_to_waypoint_idxs.get(cluster_id)
            results.append(self._sample_waypoint_from_candidates(sample_idx, original_goal_idx, candidates))
        return results

    def sample_waypoints_bucket(self, sample_idxs, original_goal_idxs):
        # sample waypoint states from the same discrete XY bucket as the original goal
        results = []
        bucket_coords = np.floor(self.stitch_points[original_goal_idxs] / self.stitch_radius).astype(np.int64)
        for sample_idx, original_goal_idx, bucket_coord in zip(sample_idxs, original_goal_idxs, bucket_coords):
            candidates = self.stitch_buckets.get(tuple(bucket_coord))
            results.append(self._sample_waypoint_from_candidates(sample_idx, original_goal_idx, candidates))
        return results

    def _sample_waypoint_from_candidates(self, sample_idx, original_goal_idx, candidates):
        if candidates is None or len(candidates) == 0:
            return None

        original_candidate_count = int(len(candidates))
        if self.stitch_cross_traj_only:
            sample_traj = self.traj_ids[int(sample_idx)]
            candidates = candidates[self.traj_ids[candidates] != sample_traj]
            if len(candidates) == 0:
                return None

        guarded_candidate_count = int(len(candidates))
        if self.stitch_space == 'state' and self.stitch_state_xy_max_dist is not None:
            original_goal_xy = self.stitch_xy_points[int(original_goal_idx)]
            for _ in range(16):
                waypoint_idx = int(candidates[np.random.randint(len(candidates))])
                xy_distance = float(np.linalg.norm(self.stitch_xy_points[waypoint_idx] - original_goal_xy))
                if xy_distance <= self.stitch_state_xy_max_dist:
                    stitch_distance = float(
                        np.linalg.norm(self.stitch_points[waypoint_idx] - self.stitch_points[int(original_goal_idx)])
                    )
                    return (
                        waypoint_idx,
                        stitch_distance,
                        xy_distance,
                        original_candidate_count,
                        guarded_candidate_count,
                    )

            xy_distances = np.linalg.norm(self.stitch_xy_points[candidates] - original_goal_xy, axis=1)
            candidates = candidates[xy_distances <= self.stitch_state_xy_max_dist]
            guarded_candidate_count = int(len(candidates))
            if guarded_candidate_count == 0:
                return None

        waypoint_idx = int(candidates[np.random.randint(len(candidates))])
        stitch_distance = float(
            np.linalg.norm(self.stitch_points[waypoint_idx] - self.stitch_points[int(original_goal_idx)])
        )
        xy_distance = float(
            np.linalg.norm(self.stitch_xy_points[waypoint_idx] - self.stitch_xy_points[int(original_goal_idx)])
        )
        return waypoint_idx, stitch_distance, xy_distance, original_candidate_count, guarded_candidate_count

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
                'stitch_space': self.stitch_space,
                'stitch_state_normalize': int(self.stitch_state_normalize),
                'stitch_state_xy_weight': float(self.stitch_state_xy_weight),
                'stitch_state_xy_max_dist': (
                    float(self.stitch_state_xy_max_dist)
                    if self.stitch_state_xy_max_dist is not None
                    else np.nan
                ),
                'future_offset': int(new_goal_idx - waypoint_idx),
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
