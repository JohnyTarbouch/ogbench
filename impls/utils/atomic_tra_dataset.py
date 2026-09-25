"""
Atomic15 BC samples plus isolated temporal and language TRA pairs.
"""

import dataclasses

import numpy as np

from utils.datasets import (
    AtomicGoalLanguageDataset,
    _attach_language_condition,
    random_shifts_batch,
)


@dataclasses.dataclass
class AtomicTRADataset(AtomicGoalLanguageDataset):
    def __post_init__(self):
        super().__post_init__()
        if self.config['policy_conditioning'] not in {'goal_image', 'language', 'goal_language'}:
            raise ValueError('Atomic TRA requires GCBC, LCBC or Image+Language conditioning.')
        if self.config['encoder'] != 'drq' or self.config['aug_type'] != 'drq_shift':
            raise ValueError('Atomic TRA currently requires DrQ with random shifts.')
        if not 0 <= float(self.config['discount']) < 1:
            raise ValueError('TRA discount must lie in [0, 1).')
        if not self.config['value_geom_sample']:
            raise ValueError('TRA requires geometric within-movement future sampling.')
        allow_train = bool(self.config.get('tra_allow_train_paraphrases', False))
        if self.language_train_control != 'none':
            raise ValueError('Atomic TRA grounding requires unmodified labels.')
        if self.language_train_variant != 'canonical' and not allow_train:
            raise ValueError(
                'Atomic TRA grounding requires canonical labels unless '
                'tra_allow_train_paraphrases is enabled.'
            )
        if self.language_train_variant == 'train' and self.language_cache['train_embeddings'].shape[1] == 0:
            raise ValueError('Atomic TRA training paraphrases require a nonempty train cache.')
        if int(self.config['num_language_tasks']) != 15:
            raise ValueError('This campaign is specifically the original Atomic15 cohort.')
        self._tra_task_ids = np.arange(1, 16, dtype=np.int32)
        with np.load(self.manifest_path, allow_pickle=False) as manifest:
            segment_tasks = np.asarray(manifest['task_id'], dtype=np.int32)
        self._tra_task_segments = [
            np.flatnonzero(segment_tasks == task_id) for task_id in self._tra_task_ids
        ]
        if any(len(indices) == 0 for indices in self._tra_task_segments):
            raise ValueError('Every coarse task must have a movement in this split.')
        split_id = {'train': 0, 'val': 1}.get(self.source_split, 2)
        seed = int(self.config.get('run_seed', 0))
        self._tra_future_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7A01, split_id]))
        self._tra_task_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7A02, split_id]))
        self._tra_aug_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7A03, split_id]))
        # Auxiliary wording never consumes BC's language RNG or the task-endpoint
        # RNG, so enabling alignment cannot change any ordinary BC sample.
        self._tra_language_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7A04, split_id]))
        self._tra_count = self._tra_offset_sum = self._tra_endpoint_count = 0
        self.manifest_summary['tra_contract'] = {
            'policy_conditioning': self.config['policy_conditioning'],
            'bc_sampling': 'unchanged_uniform_action_transition',
            'bc_goal': 'same_movement_repeated_stable_endpoint',
            'temporal_pair': 'min(current_plus_geometric_offset, same_movement_endpoint)',
            'discount': float(self.config['discount']),
            'task_alignment': 'one_uniform_movement_endpoint_per_unique_coarse_task',
            'task_alignment_count': 15,
            'task_alignment_label_source': (
                'same_split_manifest_and_pinned_train_cache'
                if self.language_train_variant == 'train'
                else 'same_split_manifest_and_pinned_canonical_cache'
            ),
            'bc_augmentation_rng': 'ordinary_policy_global_numpy_stream',
            'auxiliary_rng': (
                'independent_future_task_language_and_augmentation_streams'
                if self.language_train_variant == 'train'
                else 'independent_future_task_and_augmentation_streams'
            ),
            'is_byol': self.config.get('temporal_objective', 'tra') == 'byol',
            'temporal_objective': self.config.get('temporal_objective', 'tra'),
        }

    def sample(self, batch_size, idxs=None, evaluation=False):
        batch, task_ids, atomic_idxs = self._sample_atomic(
            batch_size, idxs=idxs, evaluation=evaluation, augment_keys=None
        )
        raw_idxs = self.transition_indices[atomic_idxs]
        segment_ids = self.transition_segment_ids[atomic_idxs]
        endpoint_idxs = self.segment_goal_indices[segment_ids]
        offsets = self._tra_future_rng.geometric(1.0 - float(self.config['discount']), len(raw_idxs))
        future_idxs = np.minimum(raw_idxs + offsets, endpoint_idxs)
        if np.any(future_idxs <= raw_idxs) or np.any(future_idxs > endpoint_idxs):
            raise RuntimeError('TRA temporal pair escaped its atomic movement.')
        batch['actor_goals'] = self._get_endpoint_goals(endpoint_idxs)
        batch['value_goals'] = self.get_observations(future_idxs)

        task_segments = np.asarray([
            indices[self._tra_task_rng.integers(len(indices))]
            for indices in self._tra_task_segments
        ])
        task_endpoints = self.segment_goal_indices[task_segments]
        batch['task_alignment_goals'] = self._get_endpoint_goals(task_endpoints)
        if self.language_train_variant == 'train':
            variants = self._tra_language_rng.integers(
                self.language_cache['train_embeddings'].shape[1], size=len(self._tra_task_ids),
                dtype=np.int32,
            )
            task_language = self.language_cache['train_embeddings'][self._tra_task_ids - 1, variants]
            batch['tra_task_language_variant_indices'] = variants
        else:
            task_language = self.language_cache['canonical_embeddings']
        batch['task_alignment_language_embeddings'] = np.asarray(task_language, dtype=np.float32)
        batch['task_alignment_task_ids'] = self._tra_task_ids.copy()
        # Retain auditable indices: all are training-only and ignored by actor.
        batch['tra_current_indices'] = raw_idxs
        batch['tra_future_indices'] = future_idxs
        batch['tra_endpoint_indices'] = endpoint_idxs
        batch['tra_task_endpoint_indices'] = task_endpoints
        batch['tra_task_segment_ids'] = task_segments
        batch['tra_task_ids'] = np.asarray(task_ids, dtype=np.int32)

        # Match each ordinary dataset's ordering: Image+Language draws wording
        # before augmentation, whereas LCBC draws it afterward. This matters
        # when wording uses the legacy global NumPy stream.
        if self.config['policy_conditioning'] == 'goal_language':
            _attach_language_condition(self, batch, task_ids, evaluation)
        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                ordinary_keys = ['observations', 'next_observations']
                auxiliary_keys = ['value_goals', 'task_alignment_goals']
                if self.config['policy_conditioning'] in {'goal_image', 'goal_language'}:
                    ordinary_keys.append('actor_goals')
                else:
                    auxiliary_keys.append('actor_goals')
                pad = self.config.get('drq_shift_pad', 2)
                random_shifts_batch(batch, ordinary_keys, pad=pad)
                random_shifts_batch(batch, auxiliary_keys, pad=pad, rng=self._tra_aug_rng)
        if self.config['policy_conditioning'] != 'goal_language':
            _attach_language_condition(self, batch, task_ids, evaluation)
        if not evaluation:
            self._tra_count += len(raw_idxs)
            self._tra_offset_sum += int(np.sum(future_idxs - raw_idxs))
            self._tra_endpoint_count += int(np.sum(future_idxs == endpoint_idxs))
        return batch

    def get_and_reset_diagnostics(self):
        metrics = super().get_and_reset_diagnostics()
        metrics['data/tra_samples'] = float(self._tra_count)
        if self._tra_count:
            metrics['data/tra_effective_offset_mean'] = self._tra_offset_sum / self._tra_count
            metrics['data/tra_future_endpoint_fraction'] = self._tra_endpoint_count / self._tra_count
        self._tra_count = self._tra_offset_sum = self._tra_endpoint_count = 0
        return metrics
