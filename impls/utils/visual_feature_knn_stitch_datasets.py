"""
Visual feature temporal stitching with local nearest-neighbor retrieval.
"""

from __future__ import annotations

import numpy as np

from utils.stitch_datasets import VisualFeatureTemporalStitchGCDataset


class VisualFeatureLocalKnnTemporalStitchGCDataset(VisualFeatureTemporalStitchGCDataset):
    """
    Visual feature stitching with approximate local top-k retrieval.
    """

    def _init_stitching(self):
        super()._init_stitching()
        self.stitch_local_knn_k = int(self._config_get('stitch_local_knn_k', 32))
        self.stitch_local_knn_max_candidates = int(
            self._config_get('stitch_local_knn_max_candidates', 256)
        )
        self.stitch_local_knn_sample_topk = bool(
            self._config_get('stitch_local_knn_sample_topk', True)
        )

        if self.stitch_local_knn_k <= 0:
            raise ValueError('stitch_local_knn_k must be positive.')
        if self.stitch_local_knn_max_candidates < 0:
            raise ValueError('stitch_local_knn_max_candidates must be non-negative.')

        self.stitch_summary.update(
            {
                'implementation': 'visual_feature_local_knn_temporal',
                'stitch_local_knn_k': self.stitch_local_knn_k,
                'stitch_local_knn_max_candidates': self.stitch_local_knn_max_candidates,
                'stitch_local_knn_sample_topk': self.stitch_local_knn_sample_topk,
            }
        )

    def sample_waypoints_kmeans(self, goal_idxs, sample_idxs):
        waypoint_idxs = np.full(len(goal_idxs), -1, dtype=np.int64)
        candidate_counts = np.zeros(len(goal_idxs), dtype=np.float64)
        goal_cluster_labels = self.stitch_cluster_labels[goal_idxs]

        for cluster_id in np.unique(goal_cluster_labels):
            row_positions = np.flatnonzero(goal_cluster_labels == cluster_id)
            candidates = self.stitch_cluster_to_waypoint_idxs.get(int(cluster_id))
            selected, counts = self._sample_local_knn_waypoints_for_cluster(
                candidates,
                np.asarray(goal_idxs[row_positions], dtype=np.int64),
                np.asarray(sample_idxs[row_positions], dtype=np.int64),
            )
            waypoint_idxs[row_positions] = selected
            candidate_counts[row_positions] = counts

        return waypoint_idxs, candidate_counts

    def _sample_local_knn_waypoints_for_cluster(self, candidates, goal_idxs, sample_idxs):
        selected = np.full(len(goal_idxs), -1, dtype=np.int64)
        candidate_counts = np.zeros(len(goal_idxs), dtype=np.float64)
        if candidates is None or len(candidates) == 0:
            return selected, candidate_counts

        candidates = np.asarray(candidates, dtype=np.int64)
        candidates = candidates[candidates + self.stitch_future_min <= self.final_state_idxs[candidates]]
        if len(candidates) == 0:
            return selected, candidate_counts

        if self.stitch_cross_traj_only:
            candidate_trajs_all = self.traj_ids[candidates]
            for row_i, sample_idx in enumerate(sample_idxs):
                candidate_counts[row_i] = float(np.sum(candidate_trajs_all != self.traj_ids[int(sample_idx)]))
        else:
            candidate_counts[:] = float(len(candidates))

        if 0 < self.stitch_local_knn_max_candidates < len(candidates):
            positions = np.random.choice(
                len(candidates),
                size=self.stitch_local_knn_max_candidates,
                replace=False,
            )
            candidates = candidates[positions]

        candidate_points = self.stitch_points[candidates]
        goal_points = self.stitch_points[np.asarray(goal_idxs, dtype=np.int64)]
        candidate_norms = np.sum(candidate_points * candidate_points, axis=1)
        goal_norms = np.sum(goal_points * goal_points, axis=1)
        distances = goal_norms[:, None] + candidate_norms[None, :] - 2.0 * goal_points @ candidate_points.T
        distances = np.maximum(distances, 0.0)

        if self.stitch_cross_traj_only:
            candidate_trajs = self.traj_ids[candidates]
            for row_i, sample_idx in enumerate(sample_idxs):
                distances[row_i, candidate_trajs == self.traj_ids[int(sample_idx)]] = np.inf

        for row_i in range(len(goal_idxs)):
            row_distances = distances[row_i]
            finite = np.flatnonzero(np.isfinite(row_distances))
            if len(finite) == 0:
                continue

            k = min(self.stitch_local_knn_k, len(finite))
            finite_distances = row_distances[finite]
            if k == len(finite):
                top_positions = finite
            else:
                top_positions = finite[np.argpartition(finite_distances, k - 1)[:k]]

            if self.stitch_local_knn_sample_topk:
                chosen_pos = int(top_positions[np.random.randint(len(top_positions))])
            else:
                chosen_pos = int(top_positions[np.argmin(row_distances[top_positions])])
            selected[row_i] = int(candidates[chosen_pos])
        return selected, candidate_counts
