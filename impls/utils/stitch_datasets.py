from __future__ import annotations

import dataclasses

import jax
import numpy as np

from utils.datasets import GCDataset


@dataclasses.dataclass
class TemporalStitchGCDataset(GCDataset):
    """
    Temporal stitching dataset.

    1. GCBC samples a normal actor goal g_A from the future of the same trajectory as s_A
    2. KMeans retrieves a waypoint w_B whose observation is close to g_A in the stitch space.
    3. The actor goal is replaced with a later state g'_B from the waypoint trajectory.

    Training: policy(s_A, goal=g'_B) -> action_A
    """

    def __post_init__(self):
        super().__post_init__()
        self._init_stitching()

    def _config_get(self, key, default):
        return self.config[key] if key in self.config else default

    def _init_stitching(self):
        self.stitch_p_aug = float(self._config_get('stitch_p_aug', 0.0))
        self.stitch_space = self._config_get('stitch_space', 'state')
        self.stitch_xy_dims = tuple(int(x) for x in self._config_get('stitch_xy_dims', (0, 1)))
        self.stitch_future_min = int(self._config_get('stitch_future_min', 0))
        self.stitch_cross_traj_only = bool(self._config_get('stitch_cross_traj_only', False))
        self.stitch_nclusters = int(self._config_get('stitch_nclusters', 40))
        self.stitch_kmeans_n_init = self._config_get('stitch_kmeans_n_init', 'auto')
        self.stitch_kmeans_random_state = self._config_get('stitch_kmeans_random_state', None)
        self.stitch_debug_samples = int(self._config_get('stitch_debug_samples', 0))

        observations = self.dataset['observations']
        if not isinstance(observations, np.ndarray) or observations.ndim != 2:
            raise ValueError('TemporalStitchGCDataset expects state observations as a 2D numpy array.')
        if self.stitch_future_min < 0:
            raise ValueError('stitch_future_min must be non-negative.')
        if self.stitch_nclusters <= 0:
            raise ValueError('stitch_nclusters must be positive.')
        if not self.stitch_xy_dims:
            raise ValueError('stitch_xy_dims must contain at least one dimension.')
        if min(self.stitch_xy_dims) < 0 or max(self.stitch_xy_dims) >= observations.shape[-1]:
            raise ValueError(
                f'stitch_xy_dims={self.stitch_xy_dims} incompatible with observation shape {observations.shape}.'
            )

        self.stitch_observations = observations
        if self.stitch_space == 'state':
            self.stitch_points = observations
        elif self.stitch_space == 'xy':
            self.stitch_points = observations[:, self.stitch_xy_dims]
        else:
            raise ValueError(f"Unsupported stitch_space={self.stitch_space!r}. Use 'state' or 'xy'.")

        self.traj_ids = np.searchsorted(self.terminal_locs, np.arange(self.size))
        self.final_state_idxs = self.terminal_locs[self.traj_ids]

        if hasattr(self.dataset, 'valid_idxs'):
            waypoint_idxs = np.asarray(self.dataset.valid_idxs, dtype=np.int64)
        else:
            waypoint_idxs = np.arange(self.size, dtype=np.int64)
        waypoint_idxs = waypoint_idxs[waypoint_idxs + self.stitch_future_min <= self.final_state_idxs[waypoint_idxs]]
        self.valid_waypoint_idxs = waypoint_idxs.astype(np.int64)
        if len(self.valid_waypoint_idxs) == 0:
            raise ValueError('No valid waypoint indices available for temporal stitching.')

        self.stitch_cluster_to_waypoint_idxs: dict[int, np.ndarray] = {}
        self.stitch_cluster_labels = None
        self._build_kmeans_groups()

        self._stitch_metric_sums: dict[str, float] = {}
        self._stitch_metric_counts: dict[str, int] = {}
        self._stitch_debug_records: list[dict[str, object]] = []
        self._stitch_sample_call = 0

        self.stitch_summary = {
            'enabled': self.stitch_p_aug > 0.0,
            'implementation': 'pure_kmeans_temporal',
            'stitch_p_aug': self.stitch_p_aug,
            'stitch_space': self.stitch_space,
            'stitch_xy_dims': self.stitch_xy_dims,
            'stitch_future_min': self.stitch_future_min,
            'stitch_cross_traj_only': self.stitch_cross_traj_only,
            'stitch_nclusters': self.stitch_nclusters,
            'stitch_kmeans_n_init': self.stitch_kmeans_n_init,
            'stitch_kmeans_random_state': self.stitch_kmeans_random_state,
            'stitch_debug_samples': self.stitch_debug_samples,
        }
        group_sizes = np.array([len(v) for v in self.stitch_cluster_to_waypoint_idxs.values()], dtype=np.int64)
        self.stitch_summary.update(
            {
                'num_waypoints': int(len(self.valid_waypoint_idxs)),
                'num_groups': int(len(self.stitch_cluster_to_waypoint_idxs)),
                'group_size_min': int(np.min(group_sizes)) if len(group_sizes) else 0,
                'group_size_mean': float(np.mean(group_sizes)) if len(group_sizes) else 0.0,
                'group_size_max': int(np.max(group_sizes)) if len(group_sizes) else 0,
                'kmeans_inertia': float(self._stitch_kmeans_inertia),
            }
        )

    def _build_kmeans_groups(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError('TemporalStitchGCDataset requires scikit-learn for KMeans retrieval.') from exc

        kmeans = KMeans(
            n_clusters=self.stitch_nclusters,
            n_init=self.stitch_kmeans_n_init,
            random_state=self.stitch_kmeans_random_state,
        )
        self.stitch_cluster_labels = kmeans.fit_predict(self.stitch_points).astype(np.int64)
        self._stitch_kmeans_inertia = kmeans.inertia_

        for cluster_id in range(self.stitch_nclusters):
            waypoint_idxs = self.valid_waypoint_idxs[self.stitch_cluster_labels[self.valid_waypoint_idxs] == cluster_id]
            if len(waypoint_idxs) > 0:
                self.stitch_cluster_to_waypoint_idxs[int(cluster_id)] = waypoint_idxs.astype(np.int64)

    def sample(self, batch_size, idxs=None, evaluation=False):
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
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def augment_actor_goal_idxs(self, sample_idxs, actor_goal_idxs):
        batch_size = len(actor_goal_idxs)
        if self.stitch_p_aug <= 0.0:
            self._record_stitch_metrics(
                attempted_fraction=0.0,
                accepted_fraction=0.0,
                accepted_given_attempted=0.0,
            )
            return actor_goal_idxs

        attempt_mask = np.random.rand(batch_size) < self.stitch_p_aug
        attempted_positions = np.flatnonzero(attempt_mask)
        new_actor_goal_idxs = np.array(actor_goal_idxs, copy=True)
        if len(attempted_positions) == 0:
            self._record_stitch_metrics(
                attempted_fraction=0.0,
                accepted_fraction=0.0,
                accepted_given_attempted=0.0,
            )
            return new_actor_goal_idxs

        original_goal_idxs = actor_goal_idxs[attempted_positions]
        sample_subset_idxs = sample_idxs[attempted_positions]
        waypoint_idxs, candidate_counts = self.sample_waypoints_kmeans(original_goal_idxs, sample_subset_idxs)

        accepted_mask = waypoint_idxs >= 0
        accepted_positions = attempted_positions[accepted_mask]
        accepted_waypoints = waypoint_idxs[accepted_mask]
        accepted_original_goals = original_goal_idxs[accepted_mask]
        accepted_samples = sample_subset_idxs[accepted_mask]

        if len(accepted_positions) > 0:
            new_goal_idxs = self._sample_new_goal_idxs(accepted_waypoints)
            new_actor_goal_idxs[accepted_positions] = new_goal_idxs
            self._maybe_record_stitch_debug(
                batch_positions=accepted_positions,
                sample_idxs=accepted_samples,
                original_goal_idxs=accepted_original_goals,
                waypoint_idxs=accepted_waypoints,
                new_goal_idxs=new_goal_idxs,
                candidate_counts=candidate_counts[accepted_mask],
            )
            self._record_stitch_metrics(
                future_offset=(new_goal_idxs - accepted_waypoints).astype(np.float64),
                goal_waypoint_stitch_distance=np.linalg.norm(
                    self.stitch_points[accepted_original_goals] - self.stitch_points[accepted_waypoints],
                    axis=-1,
                ),
                goal_waypoint_distance=np.linalg.norm(
                    self.stitch_points[accepted_original_goals] - self.stitch_points[accepted_waypoints],
                    axis=-1,
                ),
                goal_waypoint_xy_distance=np.linalg.norm(
                    self.stitch_observations[accepted_original_goals][:, self.stitch_xy_dims]
                    - self.stitch_observations[accepted_waypoints][:, self.stitch_xy_dims],
                    axis=-1,
                ),
            )

        self._record_stitch_metrics(
            attempted_fraction=float(np.mean(attempt_mask)),
            accepted_fraction=float(len(accepted_positions) / batch_size),
            accepted_given_attempted=float(len(accepted_positions) / len(attempted_positions)),
            candidate_count=candidate_counts,
            candidate_count_after_xy_guard=candidate_counts,
            candidate_count_after_guard=candidate_counts,
        )
        return new_actor_goal_idxs

    def sample_waypoints_kmeans(self, goal_idxs, sample_idxs):
        waypoint_idxs = np.full(len(goal_idxs), -1, dtype=np.int64)
        candidate_counts = np.zeros(len(goal_idxs), dtype=np.float64)
        goal_cluster_labels = self.stitch_cluster_labels[goal_idxs]
        for i, (goal_idx, sample_idx, cluster_id) in enumerate(zip(goal_idxs, sample_idxs, goal_cluster_labels)):
            candidates = self.stitch_cluster_to_waypoint_idxs.get(int(cluster_id))
            if candidates is None or len(candidates) == 0:
                continue
            waypoint_idx, candidate_count = self._sample_waypoint_from_candidates(candidates, sample_idx)
            waypoint_idxs[i] = waypoint_idx
            candidate_counts[i] = candidate_count
        return waypoint_idxs, candidate_counts

    def _sample_waypoint_from_candidates(self, candidates, sample_idx):
        candidates = np.asarray(candidates, dtype=np.int64)
        if self.stitch_cross_traj_only:
            sample_traj = self.traj_ids[int(sample_idx)]
            candidates = candidates[self.traj_ids[candidates] != sample_traj]
        candidates = candidates[candidates + self.stitch_future_min <= self.final_state_idxs[candidates]]
        if len(candidates) == 0:
            return -1, 0.0
        return int(candidates[np.random.randint(len(candidates))]), float(len(candidates))

    def _sample_new_goal_idxs(self, waypoint_idxs):
        new_goal_idxs = np.empty(len(waypoint_idxs), dtype=np.int64)
        low_idxs = waypoint_idxs + self.stitch_future_min
        high_idxs = self.final_state_idxs[waypoint_idxs]
        for i, (low, high) in enumerate(zip(low_idxs, high_idxs)):
            new_goal_idxs[i] = np.random.randint(int(low), int(high) + 1)
        return new_goal_idxs

    def _maybe_record_stitch_debug(
        self,
        batch_positions,
        sample_idxs,
        original_goal_idxs,
        waypoint_idxs,
        new_goal_idxs,
        candidate_counts,
    ):
        if self.stitch_debug_samples <= 0:
            return
        remaining = self.stitch_debug_samples - len(self._stitch_debug_records)
        if remaining <= 0:
            return

        for row_i in range(min(remaining, len(batch_positions))):
            sample_idx = int(sample_idxs[row_i])
            original_goal_idx = int(original_goal_idxs[row_i])
            waypoint_idx = int(waypoint_idxs[row_i])
            new_goal_idx = int(new_goal_idxs[row_i])
            record = {
                'sample_call': int(self._stitch_sample_call),
                'batch_pos': int(batch_positions[row_i]),
                'sample_idx': sample_idx,
                'sample_traj': int(self.traj_ids[sample_idx]),
                'original_goal_idx': original_goal_idx,
                'original_goal_traj': int(self.traj_ids[original_goal_idx]),
                'waypoint_idx': waypoint_idx,
                'waypoint_traj': int(self.traj_ids[waypoint_idx]),
                'new_goal_idx': new_goal_idx,
                'new_goal_traj': int(self.traj_ids[new_goal_idx]),
                'candidate_count': float(candidate_counts[row_i]),
                'candidate_count_after_xy_guard': float(candidate_counts[row_i]),
                'candidate_count_after_guard': float(candidate_counts[row_i]),
                'future_offset': int(new_goal_idx - waypoint_idx),
                'cross_trajectory': bool(self.traj_ids[sample_idx] != self.traj_ids[waypoint_idx]),
                'stitch_space': self.stitch_space,
                'goal_waypoint_stitch_distance': float(
                    np.linalg.norm(self.stitch_points[original_goal_idx] - self.stitch_points[waypoint_idx])
                ),
                'goal_waypoint_distance': float(
                    np.linalg.norm(self.stitch_points[original_goal_idx] - self.stitch_points[waypoint_idx])
                ),
                'goal_waypoint_xy_distance': float(
                    np.linalg.norm(
                        self.stitch_observations[original_goal_idx, self.stitch_xy_dims]
                        - self.stitch_observations[waypoint_idx, self.stitch_xy_dims]
                    )
                ),
            }
            for dim_i, dim in enumerate(self.stitch_xy_dims[:3]):
                record[f'sample_xy_{dim_i}'] = float(self.stitch_observations[sample_idx, dim])
                record[f'original_goal_xy_{dim_i}'] = float(self.stitch_observations[original_goal_idx, dim])
                record[f'waypoint_xy_{dim_i}'] = float(self.stitch_observations[waypoint_idx, dim])
                record[f'new_goal_xy_{dim_i}'] = float(self.stitch_observations[new_goal_idx, dim])
            self._stitch_debug_records.append(record)
        self._stitch_sample_call += 1

    def _record_stitch_metrics(self, **metrics):
        for name, values in metrics.items():
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            self._stitch_metric_sums[name] = self._stitch_metric_sums.get(name, 0.0) + float(np.sum(values))
            self._stitch_metric_counts[name] = self._stitch_metric_counts.get(name, 0) + int(values.size)

    def get_and_reset_diagnostics(self):
        diagnostics = {}
        for name, total in self._stitch_metric_sums.items():
            count = self._stitch_metric_counts.get(name, 0)
            if count > 0:
                value = total / count
                if name in ('attempted_fraction', 'accepted_fraction', 'accepted_given_attempted'):
                    diagnostics[f'stitch/{name}'] = value
                elif name.endswith('_mean'):
                    diagnostics[f'stitch/{name}'] = value
                else:
                    diagnostics[f'stitch/{name}_mean'] = value
        diagnostics['stitch/num_metric_batches'] = max(self._stitch_metric_counts.values(), default=0)
        self._stitch_metric_sums.clear()
        self._stitch_metric_counts.clear()
        return diagnostics

    def get_and_reset_debug_records(self):
        records = self._stitch_debug_records
        self._stitch_debug_records = []
        return records
