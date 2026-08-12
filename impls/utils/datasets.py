import dataclasses
import hashlib
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from utils.language import language_retrieval_top1, load_language_cache


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


def _file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

#################################################################
# DrQ random shift augmentation.
def shift_batch_np(arr, shifts, pad):
    """Replicate-pad images and crop each image using its own integer shift.

    arr:
        Image batch, shape (N, H, W, C)

    shifts:
        Integer shifts, shape (N, 2), each value in [0, 2 * pad]

    This matches DrQ-style random shifts:
        edge padding + random crop
    """
    n, h, w, _ = arr.shape
    padded = np.pad(arr, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')

    rows = shifts[:, 0:1] + np.arange(h)
    cols = shifts[:, 1:2] + np.arange(w)
    batch_idxs = np.arange(n)[:, None, None]

    return padded[batch_idxs, rows[:, :, None], cols[:, None, :], :]


def random_shifts_batch(batch, keys, pad=2, rng=None):
    """Apply DrQ-style random shift augmentation to image arrays in batch.

    Only 4D arrays are augmented:
        (N, H, W, C)

    This stays on the host with NumPy, matching the common DrQ random-shift
    augmentation style: edge-pad, then crop each image by a random integer shift.
    """
    for key in keys:
        arr = batch[key]
        if getattr(arr, 'ndim', 0) != 4:
            continue

        n = arr.shape[0]
        if rng is None:
            shifts = np.random.randint(0, 2 * pad + 1, size=(n, 2))
        elif hasattr(rng, 'integers'):
            shifts = rng.integers(0, 2 * pad + 1, size=(n, 2))
        else:
            shifts = rng.randint(0, 2 * pad + 1, size=(n, 2))
        batch[key] = shift_batch_np(np.asarray(arr), shifts, pad)

    return batch
#################################################################


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - aug_type: Image augmentation type (crop or drq_shift).
    - drq_shift_pad: Padding used by DrQ-style random-shift augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1
        if 'valids' in self.dataset:
            valids = np.asarray(self.dataset['valids'])
            if valids.shape != (self.size,) or np.any(~np.isin(valids, (0, 1))):
                raise ValueError('Compact valids must be a one-dimensional binary array.')
            action_free_locs = np.flatnonzero(valids == 0)
            if len(action_free_locs) == 0 or action_free_locs[-1] != self.size - 1:
                raise ValueError('Compact data must end in an action-free state.')
            self.initial_locs = np.concatenate([[0], action_free_locs[:-1] + 1])

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs, actor_goal_idxs = self.sample_goal_indices(idxs)

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])
        # print("Batch sampled with indices:", idxs)
        return batch

    def sample_goal_indices(self, idxs):
        """Sample value and actor goal indices using the standard OGBench order."""
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
        return value_goal_idxs, actor_goal_idxs

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.get_trajectory_goal_final_state_idxs(idxs)
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def get_trajectory_goal_final_state_idxs(self, idxs):
        """Include final state ( image) as goal."""
        return self.terminal_locs[
            np.searchsorted(self.terminal_locs, idxs)
        ]

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        aug_type = self.config.get('aug_type', 'crop')

        if aug_type == 'drq_shift':
            pad = self.config.get('drq_shift_pad', 2)
            random_shifts_batch(batch, keys, pad=pad)
            return

        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class EndpointInclusiveGCDataset(GCDataset):
    """
    Sample future goals through the trajectory endpoint.
    """

    def __post_init__(self):
        super().__post_init__()
        if 'valids' not in self.dataset:
            raise ValueError(
                'EndpointInclusiveGCDataset requires compact data with action-free endpoints.'
            )
        self.action_free_endpoint_locs = np.flatnonzero(self.dataset['valids'] == 0)
        if len(self.action_free_endpoint_locs) == 0:
            raise ValueError('No action-free trajectory endpoints were found.')

    def get_trajectory_goal_final_state_idxs(self, idxs):
        positions = np.searchsorted(
            self.action_free_endpoint_locs,
            idxs
        )
        if np.any(positions >= len(self.action_free_endpoint_locs)):
            raise ValueError('A sampled current state lies after the final trajectory endpoint.')
        return self.action_free_endpoint_locs[positions]


def _initialize_language_conditioning(owner):
    """Load and validate the frozen language-conditioning artifact."""
    cache = load_language_cache(
        owner.config['language_embedding_path'],
        int(owner.config['num_language_tasks']),
        int(owner.config['language_embedding_dim']),
    )
    cache_sha256 = _file_sha256(cache['path'])
    expected_model = str(owner.config.get('language_embedding_model', '') or '')
    if expected_model and cache['model_name'] != expected_model:
        raise ValueError(
            f"Language cache model {cache['model_name']!r} does not match expected {expected_model!r}."
        )
    expected_sha256 = str(owner.config.get('language_embedding_sha256', '') or '')
    if expected_sha256 and cache_sha256 != expected_sha256:
        raise ValueError('Language cache SHA-256 does not match the configured artifact.')
    retrieval = {
        split: language_retrieval_top1(cache, split) for split in ('train', 'heldout')
    }
    for split, score in retrieval.items():
        minimum = float(owner.config.get(f'language_min_{split}_retrieval_top1', 0.0))
        if not 0.0 <= minimum <= 1.0:
            raise ValueError(f'language_min_{split}_retrieval_top1 must lie in [0, 1].')
        if minimum > 0.0 and score is None:
            raise ValueError(f'Language cache has no {split} retrieval diagnostics.')
        if score is not None and score + 1e-12 < minimum:
            raise ValueError(
                f'Language cache {split} retrieval top-1 {score:.4f} is below '
                f'the configured minimum {minimum:.4f}.'
            )
    train_variant = str(owner.config.get('language_train_variant', 'canonical'))
    if train_variant not in {'canonical', 'train'}:
        raise ValueError("language_train_variant must be 'canonical' or 'train'.")
    train_control = str(owner.config.get('language_train_control', 'none'))
    if train_control not in {'none', 'zero'}:
        raise ValueError("language_train_control must be 'none' or 'zero'.")
    owner.language_cache = cache
    owner.language_train_variant = train_variant
    owner.language_train_control = train_control
    owner._language_task_counts = np.zeros(int(owner.config['num_language_tasks']), dtype=np.int64)
    owner._language_variant_counts = np.zeros(
        1 if train_variant == 'canonical' else cache['train_embeddings'].shape[1],
        dtype=np.int64,
    )
    owner.language_summary = {
        'path': cache['path'],
        'cache_sha256': cache_sha256,
        'model_name': cache['model_name'],
        'normalized': cache['normalized'],
        'embedding_dim': int(cache['canonical_embeddings'].shape[-1]),
        'train_variant': train_variant,
        'train_control': train_control,
        'train_variants_per_task': int(cache['train_embeddings'].shape[1]),
        'heldout_variants_per_task': int(cache['heldout_embeddings'].shape[1]),
        'train_retrieval_top1': retrieval['train'],
        'heldout_retrieval_top1': retrieval['heldout'],
    }
    for optional_key in ('pooling', 'task_spec_sha256'):
        if optional_key in cache:
            owner.language_summary[optional_key] = cache[optional_key]


def _attach_language_condition(owner, batch, task_ids, evaluation):
    task_ids = np.asarray(task_ids, dtype=np.int64)
    num_tasks = int(owner.config['num_language_tasks'])
    if np.any(task_ids < 1) or np.any(task_ids > num_tasks):
        raise ValueError(f'Language task IDs must be in [1, {num_tasks}].')
    task_rows = task_ids - 1

    if owner.language_train_control == 'zero':
        embeddings = np.zeros(
            (len(task_rows), int(owner.config['language_embedding_dim'])),
            dtype=np.float32,
        )
        variant_idxs = np.zeros(len(task_rows), dtype=np.int32)
    elif evaluation or owner.language_train_variant == 'canonical':
        embeddings = owner.language_cache['canonical_embeddings'][task_rows]
        variant_idxs = np.zeros(len(task_rows), dtype=np.int32)
    else:
        num_variants = owner.language_cache['train_embeddings'].shape[1]
        variant_idxs = np.random.randint(num_variants, size=len(task_rows)).astype(np.int32)
        embeddings = owner.language_cache['train_embeddings'][task_rows, variant_idxs]
    if not evaluation:
        np.add.at(owner._language_task_counts, task_rows, 1)
        np.add.at(owner._language_variant_counts, variant_idxs, 1)
    batch['language_embeddings'] = np.asarray(embeddings, dtype=np.float32)
    return batch


def _get_and_reset_language_diagnostics(owner):
    total = int(np.sum(owner._language_task_counts))
    metrics = {
        'data/language_samples': float(total),
        'data/language_train_control_zero': float(owner.language_train_control == 'zero'),
    }
    if total:
        for task_idx, count in enumerate(owner._language_task_counts, start=1):
            metrics[f'data/task_{task_idx}_fraction'] = float(count / total)
        variant_total = int(np.sum(owner._language_variant_counts))
        variant_prefix = 'canonical' if owner.language_train_variant == 'canonical' else 'train'
        for variant_idx, count in enumerate(owner._language_variant_counts):
            metrics[f'data/{variant_prefix}_variant_{variant_idx}_fraction'] = float(count / variant_total)
    owner._language_task_counts.fill(0)
    owner._language_variant_counts.fill(0)
    return metrics


@dataclasses.dataclass
class _AtomicSegmentDataset(GCDataset):
    """Sample transitions from a validated atomic Cube segment manifest.

    Manifest positions form this dataset's public index space. Each position
    maps to one transition in the original compact OGBench dataset and one
    atomic segment. Keeping the original dataset intact lets frame
    stacking use the true episode boundaries without copying image observations.
    """

    # Only a small manifest-selected subset is sampled. Building stacks for
    # every raw frame would need tens of gigabytes without changing a batch.
    preprocess_frame_stack: bool = False
    manifest_path: Optional[Union[str, Path]] = None
    source_dataset_name: Optional[str] = None
    source_split: Optional[str] = None
    source_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        super().__post_init__()
        self.raw_size = self.dataset.size
        if self.manifest_path is None:
            raise ValueError('Atomic segment datasets require manifest_path.')

        manifest_path = Path(self.manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Atomic segment manifest not found: {manifest_path}')
        manifest_sha256 = _file_sha256(manifest_path)
        split = str(self.source_split or '')
        expected_manifest_sha256 = str(
            self.config.get(f'atomic_{split}_manifest_sha256', '') or ''
        )
        if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                f'Atomic {split} manifest SHA-256 does not match the configured artifact.'
            )

        required = {'transition_indices', 'transition_segment_ids', 'task_id'}
        with np.load(manifest_path, allow_pickle=False) as manifest:
            missing = required.difference(manifest.files)
            if missing:
                raise ValueError(f'Atomic segment manifest is missing keys: {sorted(missing)}')
            require_source_fingerprint = bool(
                self.config.get('atomic_require_source_fingerprint', False)
            )
            provenance_keys = {
                'schema_version',
                'source_dataset_name',
                'source_split',
                'source_num_states',
                'source_num_valid_transitions',
                'source_num_episodes',
                'source_file_sha256',
                'source_actions_valids_sha256',
            }
            if require_source_fingerprint:
                missing_provenance = provenance_keys.difference(manifest.files)
                if missing_provenance:
                    raise ValueError(
                        'Atomic manifest source provenance is required but missing keys: '
                        f'{sorted(missing_provenance)}'
                    )
            provenance = {
                key: np.asarray(manifest[key])
                for key in provenance_keys
                if key in manifest.files
            }
            transition_indices = np.asarray(manifest['transition_indices'], dtype=np.int64)
            transition_segment_ids = np.asarray(manifest['transition_segment_ids'], dtype=np.int64)
            segment_task_ids = np.asarray(manifest['task_id'], dtype=np.int64)
            segment_episode_ids = (
                np.asarray(manifest['episode_id'], dtype=np.int64)
                if 'episode_id' in manifest.files
                else None
            )
            segment_goal_indices = (
                np.asarray(manifest['goal_index'], dtype=np.int64)
                if 'goal_index' in manifest.files
                else None
            )
            segment_ids = (
                np.asarray(manifest['segment_id'], dtype=np.int64)
                if 'segment_id' in manifest.files
                else None
            )
            segment_start_indices = (
                np.asarray(manifest['start_index'], dtype=np.int64)
                if 'start_index' in manifest.files
                else None
            )
            segment_num_transitions = (
                np.asarray(manifest['num_transitions'], dtype=np.int64)
                if 'num_transitions' in manifest.files
                else None
            )

        def provenance_scalar(key, cast):
            value = provenance[key]
            if value.shape != ():
                raise ValueError(f'Atomic manifest provenance key {key!r} must be scalar.')
            return cast(value.item())

        if require_source_fingerprint:
            if self.source_dataset_name is None or self.source_split is None:
                raise ValueError(
                    'Atomic source fingerprint validation requires the expected '
                    'source_dataset_name and source_split.'
                )
            if 'valids' not in self.dataset:
                raise ValueError('Atomic source fingerprint validation requires compact valids.')
            if provenance_scalar('schema_version', int) != 1:
                raise ValueError('Unsupported atomic manifest provenance schema version.')
            if provenance_scalar('source_dataset_name', str) != self.source_dataset_name:
                raise ValueError(
                    'Atomic manifest source dataset name does not match the requested dataset.'
                )
            if provenance_scalar('source_split', str) != self.source_split:
                raise ValueError('Atomic manifest source split does not match the requested split.')
            source_file_sha256 = provenance_scalar('source_file_sha256', str)
            if len(source_file_sha256) != 64 or any(
                character not in '0123456789abcdef' for character in source_file_sha256
            ):
                raise ValueError('Atomic manifest source file SHA-256 is malformed.')
            if self.source_path is None:
                raise ValueError(
                    'Atomic source fingerprint validation requires source_path for '
                    'archive checksum verification.'
                )
            source_path = Path(self.source_path).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f'Atomic source archive not found: {source_path}')
            if _file_sha256(source_path) != source_file_sha256:
                raise ValueError(
                    'Atomic manifest source archive checksum does not match the dataset file.'
                )
            if provenance_scalar('source_num_states', int) != self.raw_size:
                raise ValueError('Atomic manifest source state count does not match the dataset.')
            raw_valids = np.asarray(self.dataset['valids'])
            if provenance_scalar('source_num_valid_transitions', int) != int(
                np.sum(raw_valids > 0)
            ):
                raise ValueError(
                    'Atomic manifest source valid-transition count does not match the dataset.'
                )
            source_num_episodes = int(np.sum(raw_valids <= 0))
            if provenance_scalar('source_num_episodes', int) != source_num_episodes:
                raise ValueError('Atomic manifest source episode count does not match the dataset.')
            expected_fingerprint = _actions_valids_fingerprint(self.dataset)
            if provenance_scalar('source_actions_valids_sha256', str) != expected_fingerprint:
                raise ValueError(
                    'Atomic manifest source actions/valids fingerprint does not match the dataset.'
                )

        if transition_indices.ndim != 1 or transition_segment_ids.ndim != 1:
            raise ValueError('Atomic manifest transition arrays must be one-dimensional.')
        if len(transition_indices) == 0:
            raise ValueError('Atomic manifest contains no language-labelled transitions.')
        if len(transition_indices) != len(transition_segment_ids):
            raise ValueError('transition_indices and transition_segment_ids must have equal length.')
        if np.any(transition_segment_ids < 0) or np.any(transition_segment_ids >= len(segment_task_ids)):
            raise ValueError('Atomic manifest contains an out-of-range transition_segment_id.')
        if np.any(transition_indices < 0) or np.any(transition_indices >= self.raw_size - 1):
            raise ValueError(
                f'Atomic transition indices must be in [0, {self.raw_size - 2}] so next observations exist.'
            )
        if len(np.unique(transition_indices)) != len(transition_indices):
            raise ValueError('Atomic manifest contains duplicate transition indices.')

        num_segments = len(segment_task_ids)
        if segment_episode_ids is not None and segment_episode_ids.shape != (num_segments,):
            raise ValueError('Atomic manifest episode_id must have one entry per segment.')
        if segment_goal_indices is not None:
            if segment_goal_indices.shape != (num_segments,):
                raise ValueError('Atomic manifest goal_index must have one entry per segment.')
            if np.any(segment_goal_indices < 0) or np.any(segment_goal_indices >= self.raw_size):
                raise ValueError(f'Atomic goal indices must lie in [0, {self.raw_size - 1}].')
        if segment_ids is not None and not np.array_equal(segment_ids, np.arange(num_segments)):
            raise ValueError('Atomic manifest segment_id must equal consecutive row indices.')

        structural_arrays = (segment_start_indices, segment_num_transitions, segment_goal_indices)
        if any(array is not None for array in structural_arrays):
            if not all(array is not None for array in structural_arrays):
                raise ValueError(
                    'Atomic manifest must provide start_index, goal_index, and num_transitions together.'
                )
            for name, array in zip(
                ('start_index', 'num_transitions', 'goal_index'), structural_arrays
            ):
                if array.shape != (num_segments,):
                    raise ValueError(f'Atomic manifest {name} must have one entry per segment.')
            if np.any(segment_start_indices < 0) or np.any(
                segment_start_indices >= segment_goal_indices
            ):
                raise ValueError('Atomic segment start_index must be nonnegative and precede goal_index.')
            if not np.array_equal(
                segment_num_transitions, segment_goal_indices - segment_start_indices
            ):
                raise ValueError('Atomic num_transitions must equal goal_index - start_index.')
            expected_segment_ids = np.repeat(np.arange(num_segments), segment_num_transitions)
            expected_transition_indices = np.concatenate(
                [
                    np.arange(start, goal, dtype=np.int64)
                    for start, goal in zip(segment_start_indices, segment_goal_indices)
                ]
            )
            if not np.array_equal(transition_segment_ids, expected_segment_ids):
                raise ValueError('Atomic transition_segment_ids do not encode contiguous segments.')
            if not np.array_equal(transition_indices, expected_transition_indices):
                raise ValueError('Atomic transition_indices do not equal the declared [start, goal) ranges.')

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, transition_indices)]
        if np.any(transition_indices >= final_state_idxs):
            raise ValueError('Atomic manifest includes a terminal transition without an in-episode successor.')

        if 'valids' in self.dataset and np.any(
            np.asarray(self.dataset['valids'])[transition_indices] <= 0
        ):
            raise ValueError('Atomic manifest includes an action-free transition row.')

        if segment_goal_indices is not None:
            transition_goal_indices = segment_goal_indices[transition_segment_ids]
            if np.any(transition_indices >= transition_goal_indices):
                raise ValueError('Atomic goals must be strictly later than their transitions.')
            if 'valids' in self.dataset:
                episode_end_indices = np.flatnonzero(np.asarray(self.dataset['valids']) <= 0)
            else:
                episode_end_indices = self.terminal_locs
            transition_episode_rows = np.searchsorted(
                episode_end_indices, transition_indices, side='left'
            )
            goal_episode_rows = np.searchsorted(
                episode_end_indices, transition_goal_indices, side='left'
            )
            if np.any(transition_episode_rows != goal_episode_rows):
                raise ValueError('Atomic transition and goal indices must belong to the same episode.')

        task_ids = segment_task_ids[transition_segment_ids]

        self.manifest_path = manifest_path
        self.transition_indices = transition_indices
        self.transition_segment_ids = transition_segment_ids.astype(np.int32)
        self.transition_task_ids = task_ids.astype(np.int32)
        self.segment_goal_indices = segment_goal_indices
        self.size = len(transition_indices)

        task_values, task_counts = np.unique(self.transition_task_ids, return_counts=True)
        used_segment_ids = np.unique(self.transition_segment_ids)
        used_episode_ids = (
            np.unique(segment_episode_ids[used_segment_ids])
            if segment_episode_ids is not None
            else np.empty(0, dtype=np.int64)
        )
        self.manifest_summary = {
            'path': str(manifest_path),
            'manifest_sha256': manifest_sha256,
            'raw_dataset_size': int(self.raw_size),
            'num_transitions': int(self.size),
            'num_segments': int(len(used_segment_ids)),
            'num_episodes': int(len(used_episode_ids)),
            'transitions_per_task': {
                str(int(task_id)): int(count)
                for task_id, count in zip(task_values, task_counts)
            },
        }
        if provenance_keys.issubset(provenance):
            self.manifest_summary.update(
                {
                    'schema_version': provenance_scalar('schema_version', int),
                    'source_dataset_name': provenance_scalar('source_dataset_name', str),
                    'source_split': provenance_scalar('source_split', str),
                    'source_file_sha256': provenance_scalar('source_file_sha256', str),
                    'source_archive_path': str(source_path) if require_source_fingerprint else None,
                    'source_actions_valids_sha256': provenance_scalar(
                        'source_actions_valids_sha256', str
                    ),
                    'source_fingerprint_required': require_source_fingerprint,
                    'source_fingerprint_verified': require_source_fingerprint,
                    'source_archive_verified': require_source_fingerprint,
                }
            )

    def _sample_atomic(self, batch_size, idxs=None, evaluation=False, augment_keys=None):
        """Sample transitions and their internal language-task labels."""
        if idxs is None:
            idxs = np.random.randint(self.size, size=batch_size)
        idxs = np.asarray(idxs, dtype=np.int64)
        if np.any(idxs < 0) or np.any(idxs >= self.size):
            raise IndexError(f'Atomic dataset positions must be in [0, {self.size - 1}].')

        raw_idxs = self.transition_indices[idxs]
        batch = self.dataset.sample(len(idxs), raw_idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(raw_idxs)
            batch['next_observations'] = self.get_observations(raw_idxs + 1)

        if augment_keys and self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, list(augment_keys))

        return batch, self.transition_task_ids[idxs], idxs


@dataclasses.dataclass
class AtomicLanguageDataset(_AtomicSegmentDataset):
    """Atomic Cube transitions conditioned on frozen language embeddings."""

    def __post_init__(self):
        super().__post_init__()
        mode = str(self.config.get('language_dataset_mode', 'atomic_movement'))
        if mode != 'atomic_movement':
            raise ValueError("AtomicLanguageDataset requires language_dataset_mode='atomic_movement'.")
        num_language_tasks = int(self.config['num_language_tasks'])
        if np.any(self.transition_task_ids < 1) or np.any(
            self.transition_task_ids > num_language_tasks
        ):
            raise ValueError(f'Atomic task IDs must be in [1, {num_language_tasks}].')
        _initialize_language_conditioning(self)
        self._validate_language_contract()

    def _validate_language_contract(self):
        """Bind fine-language segment semantics to the exact frozen cache."""
        required_contract = bool(
            self.config.get('atomic_require_language_contract', False)
        )
        required = {
            'language_contract_schema_version',
            'task_ids',
            'task_ij',
            'task_coarse_ij',
            'task_local_ij',
            'task_instructions',
            'task_spec_sha256',
            'grid_world_x_edges',
            'grid_world_y_edges',
            'grid_row',
            'grid_column',
            'goal_outside_grid',
        }
        with np.load(self.manifest_path, allow_pickle=False) as manifest:
            available = set(manifest.files)
            if not required_contract and not required.issubset(available):
                self.manifest_summary['language_contract_verified'] = False
                return
            missing = required.difference(available)
            if missing:
                raise ValueError(
                    'Atomic language contract is required but the manifest is missing '
                    f'{sorted(missing)}.'
                )
            contract_version = np.asarray(
                manifest['language_contract_schema_version']
            )
            if contract_version.shape != () or int(contract_version.item()) != 1:
                raise ValueError('Unsupported atomic language-contract schema version.')
            manifest_task_ids = np.asarray(manifest['task_ids'], dtype=np.int32)
            manifest_task_ij = np.asarray(manifest['task_ij'], dtype=np.int32)
            manifest_coarse_ij = np.asarray(manifest['task_coarse_ij'], dtype=np.int32)
            manifest_local_ij = np.asarray(manifest['task_local_ij'], dtype=np.int32)
            manifest_instructions = np.asarray(manifest['task_instructions']).astype(str)
            manifest_spec_sha = str(np.asarray(manifest['task_spec_sha256']).item())
            manifest_x_edges = np.asarray(manifest['grid_world_x_edges'], dtype=np.float32)
            manifest_y_edges = np.asarray(manifest['grid_world_y_edges'], dtype=np.float32)
            segment_rows = np.asarray(manifest['grid_row'], dtype=np.int32)
            segment_columns = np.asarray(manifest['grid_column'], dtype=np.int32)
            outside = np.asarray(manifest['goal_outside_grid'], dtype=bool)

        expected_ids = np.arange(1, int(self.config['num_language_tasks']) + 1, dtype=np.int32)
        if not np.array_equal(manifest_task_ids, expected_ids):
            raise ValueError('Atomic language manifest task IDs are not complete and consecutive.')
        cache = self.language_cache
        for key in (
            'cache_schema_version',
            'task_ij',
            'task_coarse_ij',
            'task_local_ij',
            'task_spec_sha256',
            'grid_world_x_edges',
            'grid_world_y_edges',
        ):
            if key not in cache:
                raise ValueError(f'Language cache is missing required contract key {key!r}.')
        if int(cache['cache_schema_version']) != 1:
            raise ValueError('Unsupported language-cache schema version.')
        comparisons = {
            'task_ij': (manifest_task_ij, cache['task_ij']),
            'task_coarse_ij': (manifest_coarse_ij, cache['task_coarse_ij']),
            'task_local_ij': (manifest_local_ij, cache['task_local_ij']),
            'grid_world_x_edges': (manifest_x_edges, cache['grid_world_x_edges']),
            'grid_world_y_edges': (manifest_y_edges, cache['grid_world_y_edges']),
            'task_instructions': (manifest_instructions, cache['canonical_texts'].astype(str)),
        }
        for label, (manifest_value, cache_value) in comparisons.items():
            if not np.array_equal(manifest_value, cache_value):
                raise ValueError(f'Language cache {label} does not match the atomic manifest.')
        if manifest_spec_sha != str(cache['task_spec_sha256']):
            raise ValueError('Language cache task-spec fingerprint does not match the manifest.')
        configured_spec_sha = str(
            self.config.get('language_task_spec_sha256', '') or ''
        )
        if configured_spec_sha and manifest_spec_sha != configured_spec_sha:
            raise ValueError('Language task-spec SHA-256 does not match the configured artifact.')
        if np.any(outside):
            raise ValueError('Strict atomic language manifest contains an out-of-grid endpoint.')
        expected_cells = manifest_task_ij[
            np.asarray(self.transition_task_ids, dtype=np.int64) - 1
        ]
        transition_cells = np.stack(
            [
                segment_rows[self.transition_segment_ids],
                segment_columns[self.transition_segment_ids],
            ],
            axis=1,
        )
        if not np.array_equal(expected_cells, transition_cells):
            raise ValueError('Atomic transition task IDs disagree with their fine-grid cells.')
        self.manifest_summary.update(
            {
                'language_contract_verified': True,
                'language_contract_schema_version': 1,
                'task_spec_sha256': manifest_spec_sha,
                'num_language_tasks': int(len(manifest_task_ids)),
                'grid_shape': [
                    int(len(manifest_x_edges) - 1),
                    int(len(manifest_y_edges) - 1),
                ],
            }
        )

    def sample(self, batch_size, idxs=None, evaluation=False):
        batch, task_ids, _ = self._sample_atomic(
            batch_size,
            idxs=idxs,
            evaluation=evaluation,
            augment_keys=('observations', 'next_observations'),
        )
        return _attach_language_condition(self, batch, task_ids, evaluation)

    def get_and_reset_diagnostics(self):
        """Report the task and paraphrase mixture actually seen by training."""
        return _get_and_reset_language_diagnostics(self)


@dataclasses.dataclass
class AtomicGoalLanguageDataset(AtomicLanguageDataset):
    """Atomic Cube transitions conditioned on one shared endpoint image and text.

    The endpoint image and language task are derived from the same manifest
    segment.  This keeps the visual and language conditions exactly coupled
    while preserving the transition/action distribution used by the existing
    Atomic GCBC and Atomic LCBC datasets.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.segment_goal_indices is None:
            raise ValueError('AtomicGoalLanguageDataset requires goal_index in its atomic manifest.')
        self.atomic_goal_stack_mode = str(
            self.config.get('atomic_goal_stack_mode', 'repeat_endpoint')
        )
        if self.atomic_goal_stack_mode != 'repeat_endpoint':
            raise ValueError(
                "AtomicGoalLanguageDataset currently requires "
                "atomic_goal_stack_mode='repeat_endpoint'."
            )
        self.manifest_summary['atomic_goal_stack_mode'] = self.atomic_goal_stack_mode
        self.manifest_summary['condition_coupling'] = 'same_segment_stable_endpoint_image_and_language'
        self._validate_goal_language_coupling_contract()

    def _validate_goal_language_coupling_contract(self):
        """
        Validate the Grid-15 endpoint-to-language metadata.
        """
        required_contract = bool(
            self.config.get('atomic_require_goal_language_coupling', False)
        )
        required = {
            'task_ids',
            'task_grid_rows',
            'task_grid_columns',
            'task_instructions',
            'task_index',
            'task_id',
            'grid_row',
            'grid_column',
            'goal_xyz',
            'goal_outside_grid',
            'grid_world_x_edges',
            'grid_world_y_edges',
        }
        with np.load(self.manifest_path, allow_pickle=False) as manifest:
            available = set(manifest.files)
            if not required_contract and not required.issubset(available):
                self.manifest_summary['goal_language_coupling_verified'] = False
                return
            missing = required.difference(available)
            if missing:
                raise ValueError(
                    'Atomic goal-language coupling is required but the manifest '
                    f'is missing {sorted(missing)}.'
                )
            task_ids = np.asarray(manifest['task_ids'], dtype=np.int32)
            task_rows = np.asarray(manifest['task_grid_rows'], dtype=np.int32)
            task_columns = np.asarray(manifest['task_grid_columns'], dtype=np.int32)
            task_instructions = np.asarray(manifest['task_instructions']).astype(str)
            segment_task_indices = np.asarray(manifest['task_index'], dtype=np.int32)
            segment_task_ids = np.asarray(manifest['task_id'], dtype=np.int32)
            segment_rows = np.asarray(manifest['grid_row'], dtype=np.int32)
            segment_columns = np.asarray(manifest['grid_column'], dtype=np.int32)
            goal_xyz = np.asarray(manifest['goal_xyz'], dtype=np.float32)
            goal_outside = np.asarray(manifest['goal_outside_grid'], dtype=bool)
            x_edges = np.asarray(manifest['grid_world_x_edges'], dtype=np.float32)
            y_edges = np.asarray(manifest['grid_world_y_edges'], dtype=np.float32)

        num_tasks = int(self.config['num_language_tasks'])
        expected_task_ids = np.arange(1, num_tasks + 1, dtype=np.int32)
        if not np.array_equal(task_ids, expected_task_ids):
            raise ValueError('Atomic goal-language task IDs must be complete and consecutive.')
        if not (
            task_rows.shape
            == task_columns.shape
            == task_instructions.shape
            == task_ids.shape
        ):
            raise ValueError('Atomic goal-language task metadata has inconsistent shapes.')
        if np.any(np.diff(x_edges) <= 0) or np.any(np.diff(y_edges) <= 0):
            raise ValueError('Atomic goal-language grid edges must be strictly increasing.')
        expected_cells = {
            (row, column)
            for row in range(len(x_edges) - 1)
            for column in range(len(y_edges) - 1)
        }
        if set(zip(task_rows.tolist(), task_columns.tolist())) != expected_cells:
            raise ValueError('Atomic goal-language tasks must cover every Grid-15 cell once.')

        num_segments = len(segment_task_ids)
        segment_arrays = (
            segment_task_indices,
            segment_rows,
            segment_columns,
            goal_outside,
        )
        if any(array.shape != (num_segments,) for array in segment_arrays):
            raise ValueError('Atomic goal-language segment metadata has inconsistent shapes.')
        if goal_xyz.shape != (num_segments, 3):
            raise ValueError('Atomic goal-language goal_xyz must have one XYZ row per segment.')
        if np.any(segment_task_indices < 0) or np.any(segment_task_indices >= num_tasks):
            raise ValueError('Atomic goal-language manifest has an invalid task_index.')
        if not np.array_equal(task_ids[segment_task_indices], segment_task_ids):
            raise ValueError('Atomic goal-language task_index and task_id disagree.')
        if not np.array_equal(task_rows[segment_task_indices], segment_rows) or not np.array_equal(
            task_columns[segment_task_indices], segment_columns
        ):
            raise ValueError('Atomic goal-language task IDs disagree with their grid cells.')

        computed_rows = np.searchsorted(
            x_edges[1:-1], goal_xyz[:, 0], side='right'
        ).astype(np.int32)
        computed_columns = np.searchsorted(
            y_edges[1:-1], goal_xyz[:, 1], side='right'
        ).astype(np.int32)
        computed_rows = np.clip(computed_rows, 0, len(x_edges) - 2)
        computed_columns = np.clip(computed_columns, 0, len(y_edges) - 2)
        computed_outside = (
            (goal_xyz[:, 0] < x_edges[0])
            | (goal_xyz[:, 0] > x_edges[-1])
            | (goal_xyz[:, 1] < y_edges[0])
            | (goal_xyz[:, 1] > y_edges[-1])
        )
        if not np.array_equal(computed_rows, segment_rows) or not np.array_equal(
            computed_columns, segment_columns
        ):
            raise ValueError('Atomic goal coordinates disagree with their stored Grid-15 cells.')
        if not np.array_equal(computed_outside, goal_outside):
            raise ValueError('Atomic goal outside-grid flags disagree with goal coordinates.')

        cache_task_ids = np.asarray(self.language_cache['task_ids'], dtype=np.int32)
        cache_texts = np.asarray(self.language_cache['canonical_texts']).astype(str)
        if not np.array_equal(cache_task_ids, task_ids):
            raise ValueError('Atomic goal-language cache task IDs disagree with the manifest.')
        if not np.array_equal(cache_texts, task_instructions):
            raise ValueError('Atomic goal-language cache instructions disagree with the manifest.')

        self.manifest_summary.update(
            {
                'goal_language_coupling_verified': True,
                'goal_language_grid_shape': [len(x_edges) - 1, len(y_edges) - 1],
                'goal_language_outside_grid_segments': int(np.sum(goal_outside)),
                'goal_language_outside_grid_transitions': int(
                    np.sum(goal_outside[self.transition_segment_ids])
                ),
            }
        )

    def _get_endpoint_goals(self, goal_idxs):
        goals = jax.tree_util.tree_map(
            lambda arr: arr[goal_idxs], self.dataset['observations']
        )
        frame_stack = self.config['frame_stack']
        if frame_stack is not None:
            goals = jax.tree_util.tree_map(
                lambda arr: np.concatenate([arr] * int(frame_stack), axis=-1), goals
            )
        return goals

    def sample(self, batch_size, idxs=None, evaluation=False):
        # sample one, then derive both modalities from that transition exact atomic segment
        batch, task_ids, atomic_idxs = self._sample_atomic(
            batch_size,
            idxs=idxs,
            evaluation=evaluation,
            augment_keys=None,
        )
        segment_ids = self.transition_segment_ids[atomic_idxs]
        goal_idxs = self.segment_goal_indices[segment_ids]
        batch['actor_goals'] = self._get_endpoint_goals(goal_idxs)
        _attach_language_condition(self, batch, task_ids, evaluation)

        # apply one augmentation pass to all visual inputs
        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    ['observations', 'next_observations', 'actor_goals'],
                )
        return batch


@dataclasses.dataclass
class AtomicGCDataset(_AtomicSegmentDataset):
    """Atomic Cube conditioned on their endpoint image (not in an episode but in a pick-place task)"""

    def __post_init__(self):
        super().__post_init__()
        if self.segment_goal_indices is None:
            raise ValueError('AtomicGCDataset requires goal_index in its atomic manifest.')
        self.atomic_goal_stack_mode = str(
            self.config.get('atomic_goal_stack_mode', 'repeat_endpoint')
        )
        if self.atomic_goal_stack_mode != 'repeat_endpoint':
            raise ValueError(
                "AtomicGCDataset currently requires atomic_goal_stack_mode='repeat_endpoint'."
            )
        self.manifest_summary['atomic_goal_stack_mode'] = self.atomic_goal_stack_mode

    def _get_endpoint_goals(self, goal_idxs):
        # Return endpoint images.

        goals = jax.tree_util.tree_map(
            lambda arr: arr[goal_idxs], self.dataset['observations']
        )
        frame_stack = self.config['frame_stack']
        if frame_stack is not None:
            goals = jax.tree_util.tree_map(
                lambda arr: np.concatenate([arr] * int(frame_stack), axis=-1), goals
            )
        return goals

    def sample(self, batch_size, idxs=None, evaluation=False):
        batch, _, atomic_idxs = self._sample_atomic(
            batch_size,
            idxs=idxs,
            evaluation=evaluation,
            augment_keys=None,
        )
        segment_ids = self.transition_segment_ids[atomic_idxs]
        goal_idxs = self.segment_goal_indices[segment_ids]
        batch['actor_goals'] = self._get_endpoint_goals(goal_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    ['observations', 'next_observations', 'actor_goals'],
                )
        return batch


@dataclasses.dataclass
class AtomicBYOLDataset(AtomicGCDataset):
    """
    Atomic GCBC samples with a geometric within-movement BYOL target.
    The BC actor receives exactly the same fixed stable endpoint.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.config['frame_stack'] is not None:
            raise ValueError(
                'AtomicBYOLDataset is currently state-only and requires frame_stack=None.'
            )
        discount = float(self.config['discount'])
        if not 0.0 <= discount < 1.0:
            raise ValueError('AtomicBYOLDataset discount must lie in [0, 1).')
        if not bool(self.config['value_geom_sample']):
            raise ValueError('AtomicBYOLDataset requires value_geom_sample=True.')

        split_id = {'train': 0, 'val': 1}.get(self.source_split, 2)
        run_seed = int(self.config.get('run_seed', 0))
        self._byol_value_rng = np.random.default_rng(
            np.random.SeedSequence([run_seed, 0xB10A, split_id])
        )
        self._byol_sample_count = 0
        self._byol_requested_offset_sum = 0
        self._byol_effective_offset_sum = 0
        self._byol_endpoint_count = 0
        self.manifest_summary.update(
            {
                'byol_value_goal_sampling': 'clipped_geometric_within_atomic_segment',
                'byol_discount': discount,
                'byol_rng_seed': run_seed,
                'byol_state_only': True,
            }
        )

    def sample(self, batch_size, idxs=None, evaluation=False):
        batch, _, atomic_idxs = self._sample_atomic(
            batch_size,
            idxs=idxs,
            evaluation=evaluation,
            augment_keys=None,
        )
        raw_idxs = self.transition_indices[atomic_idxs]
        segment_ids = self.transition_segment_ids[atomic_idxs]
        endpoint_idxs = self.segment_goal_indices[segment_ids]

        requested_offsets = self._byol_value_rng.geometric(
            p=1.0 - float(self.config['discount']),
            size=len(raw_idxs),
        )
        value_goal_idxs = np.minimum(raw_idxs + requested_offsets, endpoint_idxs)
        if np.any(value_goal_idxs <= raw_idxs) or np.any(value_goal_idxs > endpoint_idxs):
            raise RuntimeError('Atomic BYOL target escaped its declared segment bounds.')

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self._get_endpoint_goals(endpoint_idxs)
        successes = (raw_idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if not evaluation:
            effective_offsets = value_goal_idxs - raw_idxs
            self._byol_sample_count += len(raw_idxs)
            self._byol_requested_offset_sum += int(np.sum(requested_offsets))
            self._byol_effective_offset_sum += int(np.sum(effective_offsets))
            self._byol_endpoint_count += int(np.sum(value_goal_idxs == endpoint_idxs))

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    ['observations', 'next_observations', 'value_goals', 'actor_goals'],
                )
        return batch

    def get_and_reset_diagnostics(self):
        count = self._byol_sample_count
        metrics = {'data/atomic_byol_samples': float(count)}
        if count:
            metrics.update(
                {
                    'data/atomic_byol_requested_offset_mean': (
                        self._byol_requested_offset_sum / count
                    ),
                    'data/atomic_byol_effective_offset_mean': (
                        self._byol_effective_offset_sum / count
                    ),
                    'data/atomic_byol_endpoint_fraction': self._byol_endpoint_count / count,
                }
            )
        self._byol_sample_count = 0
        self._byol_requested_offset_sum = 0
        self._byol_effective_offset_sum = 0
        self._byol_endpoint_count = 0
        return metrics


def _actions_valids_fingerprint(dataset):
    digest = hashlib.sha256()
    for key in ('actions', 'valids'):
        if key not in dataset:
            raise ValueError(f"Endpoint datasets require compact data key {key!r}.")
        array = np.asarray(dataset[key])
        digest.update(key.encode('utf-8'))
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


@dataclasses.dataclass
class _EndpointManifestDataset(GCDataset):
    """
    Selected transitions paired with one achieved episode endpoint
    """

    preprocess_frame_stack: bool = False
    manifest_path: Optional[Union[str, Path]] = None
    source_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        super().__post_init__()
        if 'next_observations' in self.dataset or 'valids' not in self.dataset:
            raise ValueError('Endpoint datasets require compact observations with a valids array.')
        if self.manifest_path is None:
            raise ValueError('Endpoint datasets require manifest_path.')

        self.raw_size = self.dataset.size
        raw_valids = np.asarray(self.dataset['valids'])
        if raw_valids.shape != (self.raw_size,) or np.any(~np.isin(raw_valids, (0, 1))):
            raise ValueError('Compact valids must be a one-dimensional binary array.')
        self.raw_goal_indices = np.flatnonzero(raw_valids == 0).astype(np.int64)
        if len(self.raw_goal_indices) == 0 or self.raw_goal_indices[-1] != self.raw_size - 1:
            raise ValueError('Compact data must end in an action-free endpoint row.')
        self.raw_episode_starts = np.concatenate(
            [np.asarray([0], dtype=np.int64), self.raw_goal_indices[:-1] + 1]
        )
        if np.any(self.raw_episode_starts >= self.raw_goal_indices):
            raise ValueError('Every compact episode must contain at least one valid transition.')
        expected_valids = np.ones(self.raw_size, dtype=raw_valids.dtype)
        expected_valids[self.raw_goal_indices] = 0
        if not np.array_equal(raw_valids, expected_valids):
            raise ValueError('Compact valids must contain exactly one action-free row per episode.')

        manifest_path = Path(self.manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Endpoint manifest not found: {manifest_path}')
        required = {
            'schema_version',
            'source_dataset_name',
            'source_split',
            'source_num_states',
            'source_num_valid_transitions',
            'source_num_episodes',
            'source_file_sha256',
            'source_actions_valids_sha256',
            'task_spec_sha256',
            'endpoint_id',
            'episode_id',
            'episode_start_index',
            'goal_index',
            'start_ij',
            'goal_ij',
            'task_id',
            'bfs_distance',
            'transition_indices',
            'transition_endpoint_ids',
            'task_ids',
            'task_ij',
            'maze_map',
            'stability_window',
            'required_bfs_distance',
            'endpoint_rule',
        }
        with np.load(manifest_path, allow_pickle=False) as manifest:
            missing = required.difference(manifest.files)
            if missing:
                raise ValueError(f'Endpoint manifest is missing keys: {sorted(missing)}')
            values = {key: np.asarray(manifest[key]) for key in required}

        def scalar(key, cast):
            value = values[key]
            if value.shape != ():
                raise ValueError(f'Endpoint manifest {key} must be a scalar.')
            return cast(value.item())

        schema_version = scalar('schema_version', int)
        if schema_version != 1:
            raise ValueError(f'Unsupported endpoint manifest schema: {schema_version!r}.')
        source_num_states = scalar('source_num_states', int)
        source_num_valid = scalar('source_num_valid_transitions', int)
        source_num_episodes = scalar('source_num_episodes', int)
        if source_num_states != self.raw_size:
            raise ValueError(
                f'Endpoint manifest has {source_num_states} states; dataset has {self.raw_size}.'
            )
        if source_num_valid != int(np.sum(raw_valids)):
            raise ValueError('Endpoint manifest valid-transition count does not match the dataset.')
        if source_num_episodes != len(self.raw_goal_indices):
            raise ValueError('Endpoint manifest episode count does not match the compact dataset.')
        source_fingerprint = scalar('source_actions_valids_sha256', str)
        if source_fingerprint != _actions_valids_fingerprint(self.dataset):
            raise ValueError('Endpoint manifest actions/valids fingerprint does not match the dataset.')

        task_ids = np.asarray(values['task_ids'], dtype=np.int32)
        task_ij = np.asarray(values['task_ij'], dtype=np.int32)
        maze_map = np.asarray(values['maze_map'], dtype=np.int8)
        expected_task_ids = np.arange(1, len(task_ids) + 1, dtype=np.int32)
        if not np.array_equal(task_ids, expected_task_ids):
            raise ValueError('Endpoint task IDs must be consecutive and one-indexed.')
        if task_ij.shape != (len(task_ids), 2) or len(np.unique(task_ij, axis=0)) != len(task_ids):
            raise ValueError('Endpoint task_ij must contain one unique cell per task.')
        open_ij = np.argwhere(maze_map == 0).astype(np.int32)
        if not np.array_equal(task_ij, open_ij):
            raise ValueError('Endpoint tasks must list every open maze cell in row-major order.')

        endpoint_ids = np.asarray(values['endpoint_id'], dtype=np.int64)
        episode_ids = np.asarray(values['episode_id'], dtype=np.int64)
        episode_starts = np.asarray(values['episode_start_index'], dtype=np.int64)
        goal_indices = np.asarray(values['goal_index'], dtype=np.int64)
        start_ij = np.asarray(values['start_ij'], dtype=np.int32)
        goal_ij = np.asarray(values['goal_ij'], dtype=np.int32)
        endpoint_task_ids = np.asarray(values['task_id'], dtype=np.int32)
        bfs_distances = np.asarray(values['bfs_distance'], dtype=np.int32)
        num_endpoints = len(endpoint_ids)
        one_dimensional = (episode_ids, episode_starts, goal_indices, endpoint_task_ids, bfs_distances)
        if any(array.shape != (num_endpoints,) for array in one_dimensional):
            raise ValueError('Endpoint-level manifest arrays have inconsistent lengths.')
        if start_ij.shape != (num_endpoints, 2) or goal_ij.shape != (num_endpoints, 2):
            raise ValueError('Endpoint start_ij and goal_ij must have shape (num_endpoints, 2).')
        if not np.array_equal(endpoint_ids, np.arange(num_endpoints, dtype=np.int64)):
            raise ValueError('Endpoint IDs must be consecutive and zero-indexed.')
        if len(np.unique(episode_ids)) != num_endpoints:
            raise ValueError('Endpoint manifest contains duplicate episode IDs.')
        if np.any(episode_ids < 0) or np.any(episode_ids >= len(self.raw_goal_indices)):
            raise ValueError('Endpoint manifest contains an out-of-range episode ID.')
        if not np.array_equal(episode_starts, self.raw_episode_starts[episode_ids]):
            raise ValueError('Endpoint manifest episode starts do not match compact valids.')
        if not np.array_equal(goal_indices, self.raw_goal_indices[episode_ids]):
            raise ValueError('Endpoint manifest goal indices do not match action-free endpoint rows.')
        if np.any(endpoint_task_ids < 1) or np.any(endpoint_task_ids > len(task_ids)):
            raise ValueError('Endpoint manifest contains an out-of-range task ID.')
        if not np.array_equal(task_ij[endpoint_task_ids - 1], goal_ij):
            raise ValueError('Endpoint task IDs do not agree with their goal_ij cells.')
        required_distance = scalar('required_bfs_distance', int)
        if np.any(bfs_distances != required_distance):
            raise ValueError('Endpoint manifest contains a trajectory with the wrong BFS distance.')

        transition_indices = np.asarray(values['transition_indices'], dtype=np.int64)
        transition_endpoint_ids = np.asarray(values['transition_endpoint_ids'], dtype=np.int64)
        expected_transition_indices = np.concatenate(
            [np.arange(start, goal, dtype=np.int64) for start, goal in zip(episode_starts, goal_indices)]
        )
        expected_transition_endpoint_ids = np.repeat(
            endpoint_ids, goal_indices - episode_starts
        )
        if not np.array_equal(transition_indices, expected_transition_indices):
            raise ValueError('Endpoint transition indices do not exactly cover each retained episode.')
        if not np.array_equal(transition_endpoint_ids, expected_transition_endpoint_ids):
            raise ValueError('Endpoint transition-to-endpoint mapping is invalid.')
        if len(np.unique(transition_indices)) != len(transition_indices):
            raise ValueError('Endpoint manifest contains duplicate transition indices.')
        if np.any(raw_valids[transition_indices] != 1):
            raise ValueError('Endpoint manifest includes an action-free row as a transition.')

        self.manifest_path = manifest_path
        self.transition_indices = transition_indices
        self.transition_endpoint_ids = transition_endpoint_ids.astype(np.int32)
        self.endpoint_episode_ids = episode_ids.astype(np.int32)
        self.endpoint_goal_indices = goal_indices
        self.endpoint_task_ids = endpoint_task_ids
        self.task_ids = task_ids
        self.task_ij = task_ij
        self.task_spec_sha256 = scalar('task_spec_sha256', str)
        self.source_file_sha256 = scalar('source_file_sha256', str)
        self.source_path = (
            Path(self.source_path).expanduser().resolve() if self.source_path is not None else None
        )
        if self.source_path is not None:
            if not self.source_path.is_file():
                raise FileNotFoundError(f'Endpoint source dataset not found: {self.source_path}')
            if _file_sha256(self.source_path) != self.source_file_sha256:
                raise ValueError('Endpoint manifest source-file SHA-256 does not match the dataset archive.')
        self.source_dataset_name = scalar('source_dataset_name', str)
        self.source_split = scalar('source_split', str)
        self.size = len(transition_indices)
        self.endpoint_sampling = str(self.config.get('endpoint_sampling', 'uniform_transitions'))
        if self.endpoint_sampling not in {'uniform_tasks', 'uniform_transitions'}:
            raise ValueError("endpoint_sampling must be 'uniform_tasks' or 'uniform_transitions'.")
        self._positions_by_task = [
            np.flatnonzero(self.endpoint_task_ids[self.transition_endpoint_ids] == task_id)
            for task_id in self.task_ids
        ]
        if any(len(positions) == 0 for positions in self._positions_by_task):
            raise ValueError('Every endpoint task must have at least one retained transition.')
        self._endpoint_task_counts = np.zeros(len(self.task_ids), dtype=np.int64)
        task_values, task_counts = np.unique(endpoint_task_ids, return_counts=True)
        self.endpoint_manifest_summary = {
            'path': str(manifest_path),
            'manifest_sha256': _file_sha256(manifest_path),
            'schema_version': schema_version,
            'source_dataset_name': self.source_dataset_name,
            'source_split': self.source_split,
            'source_file_sha256': self.source_file_sha256,
            'source_path': str(self.source_path) if self.source_path is not None else None,
            'source_file_verified': self.source_path is not None,
            'source_actions_valids_sha256': source_fingerprint,
            'task_spec_sha256': self.task_spec_sha256,
            'raw_dataset_size': int(self.raw_size),
            'raw_num_episodes': int(len(self.raw_goal_indices)),
            'num_retained_episodes': int(num_endpoints),
            'num_transitions': int(self.size),
            'stability_window': scalar('stability_window', int),
            'required_bfs_distance': required_distance,
            'endpoint_rule': scalar('endpoint_rule', str),
            'sampling': self.endpoint_sampling,
            'episodes_per_task': {
                str(int(task_id)): int(count) for task_id, count in zip(task_values, task_counts)
            },
        }

    def _sample_positions(self, batch_size):
        if self.endpoint_sampling == 'uniform_transitions':
            return np.random.randint(self.size, size=batch_size)
        task_rows = np.random.randint(len(self.task_ids), size=batch_size)
        return np.asarray(
            [
                self._positions_by_task[task_row][np.random.randint(len(self._positions_by_task[task_row]))]
                for task_row in task_rows
            ],
            dtype=np.int64,
        )

    def get_observations(self, idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        if self.config['frame_stack'] is None:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        episode_rows = np.searchsorted(self.raw_goal_indices, idxs)
        if np.any(episode_rows >= len(self.raw_goal_indices)):
            raise IndexError('Observation index lies beyond the final compact episode.')
        episode_starts = self.raw_episode_starts[episode_rows]
        stacks = []
        for offset in reversed(range(self.config['frame_stack'])):
            stack_idxs = np.maximum(idxs - offset, episode_starts)
            stacks.append(jax.tree_util.tree_map(lambda arr: arr[stack_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=-1), *stacks)

    def _sample_endpoint(self, batch_size, idxs=None, evaluation=False):
        positions = self._sample_positions(batch_size) if idxs is None else np.asarray(idxs, dtype=np.int64)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= self.size):
            raise IndexError(f'Endpoint dataset positions must be in [0, {self.size - 1}].')
        raw_idxs = self.transition_indices[positions]
        endpoint_ids = self.transition_endpoint_ids[positions]
        task_ids = self.endpoint_task_ids[endpoint_ids]
        batch = self.dataset.sample(len(positions), raw_idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(raw_idxs)
            batch['next_observations'] = self.get_observations(raw_idxs + 1)
        if not evaluation:
            np.add.at(self._endpoint_task_counts, task_ids - 1, 1)
        return batch, task_ids, self.endpoint_goal_indices[endpoint_ids]

    def _get_and_reset_endpoint_diagnostics(self):
        total = int(np.sum(self._endpoint_task_counts))
        metrics = {'data/endpoint_samples': float(total)}
        if total:
            for task_id, count in enumerate(self._endpoint_task_counts, start=1):
                metrics[f'data/endpoint_task_{task_id}_fraction'] = float(count / total)
        self._endpoint_task_counts.fill(0)
        return metrics


@dataclasses.dataclass
class EndpointGoalDataset(_EndpointManifestDataset):
    """
    Transitions conditioned on their fixed achieved endpoint image.
    """

    def __post_init__(self):
        super().__post_init__()
        mode = str(self.config.get('endpoint_dataset_mode', 'stable_achieved_endpoint'))
        if mode != 'stable_achieved_endpoint':
            raise ValueError(
                "EndpointGoalDataset requires "
                "endpoint_dataset_mode='stable_achieved_endpoint'."
            )
        self.endpoint_goal_stack_mode = str(
            self.config.get('endpoint_goal_stack_mode', 'repeat_endpoint')
        )
        if self.endpoint_goal_stack_mode != 'repeat_endpoint':
            raise ValueError(
                "EndpointGoalDataset requires "
                "endpoint_goal_stack_mode='repeat_endpoint'."
            )
        self.endpoint_manifest_summary['endpoint_goal_stack_mode'] = (
            self.endpoint_goal_stack_mode
        )

    def _get_endpoint_goals(self, goal_idxs):
        goals = jax.tree_util.tree_map(
            lambda arr: arr[goal_idxs], self.dataset['observations']
        )
        frame_stack = self.config['frame_stack']
        if frame_stack is not None:
            goals = jax.tree_util.tree_map(
                lambda arr: np.concatenate([arr] * int(frame_stack), axis=-1),
                goals,
            )
        return goals

    def sample(self, batch_size, idxs=None, evaluation=False):
        batch, _, goal_idxs = self._sample_endpoint(
            batch_size,
            idxs=idxs,
            evaluation=evaluation,
        )
        batch['actor_goals'] = self._get_endpoint_goals(goal_idxs)

        if (
            not evaluation
            and self.config['p_aug'] is not None
            and np.random.rand() < self.config['p_aug']
        ):
            self.augment(
                batch,
                ['observations', 'next_observations', 'actor_goals'],
            )
        return batch

    def get_and_reset_diagnostics(self):
        return self._get_and_reset_endpoint_diagnostics()


@dataclasses.dataclass
class EndpointLanguageDataset(_EndpointManifestDataset):
    """Transitions conditioned on language"""

    def __post_init__(self):
        super().__post_init__()
        mode = str(self.config.get('endpoint_dataset_mode', 'stable_achieved_endpoint'))
        if mode != 'stable_achieved_endpoint':
            raise ValueError("EndpointLanguageDataset requires endpoint_dataset_mode='stable_achieved_endpoint'.")
        _initialize_language_conditioning(self)
        cache_task_ij = self.language_cache.get('task_ij')
        if cache_task_ij is None or not np.array_equal(cache_task_ij, self.task_ij):
            raise ValueError('Language cache task_ij does not match the endpoint manifest.')
        cache_spec_sha = self.language_cache.get('task_spec_sha256')
        if cache_spec_sha != self.task_spec_sha256:
            raise ValueError('Language cache task-spec fingerprint does not match the endpoint manifest.')

    def sample(self, batch_size, idxs=None, evaluation=False):
        batch, task_ids, _ = self._sample_endpoint(batch_size, idxs=idxs, evaluation=evaluation)
        _attach_language_condition(self, batch, task_ids, evaluation)
        if not evaluation and self.config['p_aug'] is not None and np.random.rand() < self.config['p_aug']:
            self.augment(batch, ['observations', 'next_observations'])
        return batch

    def get_and_reset_diagnostics(self):
        metrics = self._get_and_reset_endpoint_diagnostics()
        metrics.update(_get_and_reset_language_diagnostics(self))
        return metrics


@dataclasses.dataclass
class FutureGoalLanguageDataset(GCDataset):
    """Standard OGBench future-goal sampling with goal images replaced by language.

    This class samples current transitions and future indices exactly through
    ``GCDataset.sample_goal_indices``. A precomputed per-state lookup maps each
    sampled future index to a destination-language task. Unlike
    ``AtomicLanguageDataset``, it does not use movement manifests.
    """

    preprocess_frame_stack: bool = False
    labels_path: Optional[Union[str, Path]] = None
    source_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        super().__post_init__()
        mode = str(self.config.get('language_dataset_mode', 'standard_future_goal'))
        if mode != 'standard_future_goal':
            raise ValueError(
                "FutureGoalLanguageDataset requires language_dataset_mode='standard_future_goal'."
            )
        if self.labels_path is None:
            raise ValueError('FutureGoalLanguageDataset requires labels_path.')

        labels_path = Path(self.labels_path).expanduser().resolve()
        if not labels_path.is_file():
            raise FileNotFoundError(f'Future-goal language labels not found: {labels_path}')
        with np.load(labels_path, allow_pickle=False) as labels:
            required = {'state_task_ids', 'task_ids', 'source_num_states'}
            missing = required.difference(labels.files)
            if missing:
                raise ValueError(f'Future-goal language labels are missing keys: {sorted(missing)}')
            state_task_ids_raw = np.asarray(labels['state_task_ids'])
            task_ids_raw = np.asarray(labels['task_ids'])
            state_task_ids = np.asarray(state_task_ids_raw, dtype=np.int32)
            task_ids = np.asarray(task_ids_raw, dtype=np.int32)
            source_num_states_raw = np.asarray(labels['source_num_states'])
            source_num_states = int(source_num_states_raw)
            outside_grid_states_raw = (
                np.asarray(labels['num_outside_grid_states'])
                if 'num_outside_grid_states' in labels
                else None
            )
            outside_grid_states = int(outside_grid_states_raw) if outside_grid_states_raw is not None else 0
            provenance_keys = {
                'schema_version',
                'source_dataset_name',
                'source_split',
                'source_file_sha256',
                'source_actions_valids_sha256',
                'source_num_valid_transitions',
                'source_num_episodes',
                'task_spec_sha256',
                'label_rule',
                'state_ij',
                'task_ij',
                'maze_map',
            }
            provenance = {
                key: np.asarray(labels[key]) for key in provenance_keys if key in labels.files
            }

        if 'schema_version' in provenance:
            antmaze_required = provenance_keys
            missing = antmaze_required.difference(provenance)
            if missing:
                raise ValueError(
                    f'Versioned future-goal labels are missing provenance keys: {sorted(missing)}'
                )
            if provenance['schema_version'].dtype != np.dtype(np.int32):
                raise ValueError('Future-goal schema_version must have dtype int32.')
            if provenance['schema_version'].shape != () or int(provenance['schema_version']) != 1:
                raise ValueError('Unsupported future-goal language-label schema version.')
            expected_dtypes = {
                'source_num_valid_transitions': np.dtype(np.int64),
                'source_num_episodes': np.dtype(np.int64),
                'state_ij': np.dtype(np.int32),
                'task_ij': np.dtype(np.int32),
                'maze_map': np.dtype(np.int8),
            }
            for key, expected_dtype in expected_dtypes.items():
                if provenance[key].dtype != expected_dtype:
                    raise ValueError(
                        f'Future-goal provenance key {key!r} must have dtype {expected_dtype}.'
                    )
            if state_task_ids_raw.dtype != np.dtype(np.int32) or task_ids_raw.dtype != np.dtype(np.int32):
                raise ValueError('Versioned future-goal task ID arrays must have dtype int32.')
            if source_num_states_raw.dtype != np.dtype(np.int64) or source_num_states_raw.shape != ():
                raise ValueError('Versioned future-goal source_num_states must be an int64 scalar.')
            if outside_grid_states_raw is None or outside_grid_states_raw.dtype != np.dtype(np.int64):
                raise ValueError('Versioned future-goal num_outside_grid_states must have dtype int64.')

        num_tasks = int(self.config['num_language_tasks'])
        expected_task_ids = np.arange(1, num_tasks + 1, dtype=np.int32)
        if not np.array_equal(task_ids, expected_task_ids):
            raise ValueError(f'Future-goal label task IDs must equal {expected_task_ids.tolist()}.')
        if state_task_ids.shape != (self.dataset.size,):
            raise ValueError(
                f'Future-goal state_task_ids has shape {state_task_ids.shape}; '
                f'expected {(self.dataset.size,)}.'
            )
        if source_num_states != self.dataset.size:
            raise ValueError(
                f'Future-goal labels were built for {source_num_states} states, '
                f'but the dataset has {self.dataset.size}.'
            )
        if np.any(state_task_ids < 1) or np.any(state_task_ids > num_tasks):
            raise ValueError(f'Future-goal state task IDs must be in [1, {num_tasks}].')

        if 'schema_version' in provenance:
            def provenance_scalar(key, cast):
                value = provenance[key]
                if value.shape != ():
                    raise ValueError(f'Future-goal provenance key {key!r} must be scalar.')
                return cast(value.item())

            if 'valids' not in self.dataset:
                raise ValueError('Versioned future-goal labels require compact data with valids.')
            num_valid = int(np.sum(self.dataset['valids'] > 0))
            num_episodes = int(np.sum(self.dataset['valids'] == 0))
            if provenance_scalar('source_num_valid_transitions', int) != num_valid:
                raise ValueError('Future-goal label valid-transition count does not match the dataset.')
            if provenance_scalar('source_num_episodes', int) != num_episodes:
                raise ValueError('Future-goal label episode count does not match the dataset.')
            if provenance_scalar('source_actions_valids_sha256', str) != _actions_valids_fingerprint(self.dataset):
                raise ValueError('Future-goal label actions/valids fingerprint does not match the dataset.')

            source_sha256 = provenance_scalar('source_file_sha256', str)
            if len(source_sha256) != 64 or any(character not in '0123456789abcdef' for character in source_sha256):
                raise ValueError('Future-goal source_file_sha256 is not a lowercase SHA-256 digest.')
            if self.source_path is None:
                raise ValueError('Versioned future-goal labels require source_path for archive verification.')
            source_path = Path(self.source_path).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f'Future-goal source archive not found: {source_path}')
            if _file_sha256(source_path) != source_sha256:
                raise ValueError('Future-goal label source archive checksum does not match the dataset file.')
            expected_split = 'val' if source_path.stem.endswith('-val') else 'train'
            expected_dataset_name = (
                source_path.stem[:-4] if expected_split == 'val' else source_path.stem
            )
            if provenance_scalar('source_split', str) != expected_split:
                raise ValueError('Future-goal label source split does not match the dataset file.')
            if provenance_scalar('source_dataset_name', str) != expected_dataset_name:
                raise ValueError('Future-goal label dataset name does not match the dataset file.')
            if provenance_scalar('label_rule', str) != 'strict_qpos_xy_open_cell':
                raise ValueError('Unsupported AntMaze future-goal label rule.')

            task_ij = np.asarray(provenance['task_ij'], dtype=np.int32)
            state_ij = np.asarray(provenance['state_ij'], dtype=np.int32)
            maze_map = np.asarray(provenance['maze_map'], dtype=np.int8)
            if task_ij.shape != (num_tasks, 2) or len(np.unique(task_ij, axis=0)) != num_tasks:
                raise ValueError('Future-goal task_ij must contain one unique cell per task.')
            if not np.array_equal(task_ij, np.argwhere(maze_map == 0).astype(np.int32)):
                raise ValueError('Future-goal task_ij must list all open maze cells in row-major order.')
            if maze_map.shape != (8, 8) or np.any(~np.isin(maze_map, (0, 1))):
                raise ValueError('Future-goal AntMaze maze_map must be a binary 8x8 array.')
            if state_ij.shape != (self.dataset.size, 2):
                raise ValueError(
                    f'Future-goal state_ij has shape {state_ij.shape}; expected {(self.dataset.size, 2)}.'
                )
            if not np.array_equal(task_ij[state_task_ids - 1], state_ij):
                raise ValueError('Future-goal state task IDs do not agree with state_ij.')
            if outside_grid_states != 0:
                raise ValueError('Versioned AntMaze future-goal labels may not contain outside-grid states.')
            self.task_ij = task_ij
            self.task_spec_sha256 = provenance_scalar('task_spec_sha256', str)
            self.source_actions_valids_sha256 = provenance_scalar(
                'source_actions_valids_sha256', str
            )

        self.labels_path = labels_path
        self.state_task_ids = state_task_ids
        task_values, task_counts = np.unique(state_task_ids, return_counts=True)
        self.future_label_summary = {
            'path': str(labels_path),
            'labels_sha256': _file_sha256(labels_path),
            'raw_dataset_size': int(self.dataset.size),
            'num_outside_grid_states': outside_grid_states,
            'states_per_task': {
                str(int(task_id)): int(count)
                for task_id, count in zip(task_values, task_counts)
            },
        }
        if 'schema_version' in provenance:
            self.future_label_summary.update(
                {
                    'schema_version': provenance_scalar('schema_version', int),
                    'source_dataset_name': provenance_scalar('source_dataset_name', str),
                    'source_split': provenance_scalar('source_split', str),
                    'source_file_sha256': provenance_scalar('source_file_sha256', str),
                    'source_actions_valids_sha256': self.source_actions_valids_sha256,
                    'source_num_valid_transitions': provenance_scalar(
                        'source_num_valid_transitions', int
                    ),
                    'source_num_episodes': provenance_scalar('source_num_episodes', int),
                    'task_spec_sha256': self.task_spec_sha256,
                    'label_rule': provenance_scalar('label_rule', str),
                }
            )
        self._future_goal_count = 0
        self._future_goal_offset_sum = 0
        self._future_goal_same_state_count = 0
        self._future_goal_same_cell_count = 0
        _initialize_language_conditioning(self)
        if 'schema_version' in provenance:
            if not np.array_equal(self.language_cache.get('task_ij'), self.task_ij):
                raise ValueError('Language cache task_ij does not match future-goal labels.')
            if self.language_cache.get('task_spec_sha256') != self.task_spec_sha256:
                raise ValueError('Language cache task-spec fingerprint does not match future-goal labels.')

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        idxs = np.asarray(idxs, dtype=np.int64)
        batch = self.dataset.sample(len(idxs), idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        _, actor_goal_idxs = self.sample_goal_indices(idxs)
        self._attach_future_language(batch, idxs, actor_goal_idxs, evaluation)
        if not evaluation:
            if self.config['p_aug'] is not None and np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def _attach_future_language(self, batch, idxs, actor_goal_idxs, evaluation):
        # attach language describing the same state selected as actor goal
        future_task_ids = self.state_task_ids[actor_goal_idxs]
        _attach_language_condition(self, batch, future_task_ids, evaluation)
        if not evaluation:
            offsets = actor_goal_idxs - idxs
            self._future_goal_count += len(idxs)
            self._future_goal_offset_sum += int(np.sum(offsets))
            self._future_goal_same_state_count += int(np.sum(offsets == 0))
            self._future_goal_same_cell_count += int(
                np.sum(future_task_ids == self.state_task_ids[idxs])
            )

    def get_and_reset_diagnostics(self):
        metrics = _get_and_reset_language_diagnostics(self)
        if self._future_goal_count:
            count = self._future_goal_count
            metrics.update(
                {
                    'data/future_goal_offset_mean': self._future_goal_offset_sum / count,
                    'data/future_goal_same_state_fraction': self._future_goal_same_state_count / count,
                    'data/future_goal_same_cell_fraction': self._future_goal_same_cell_count / count,
                }
            )
        self._future_goal_count = 0
        self._future_goal_offset_sum = 0
        self._future_goal_same_state_count = 0
        self._future_goal_same_cell_count = 0
        return metrics


@dataclasses.dataclass
class FutureGoalImageLanguageDataset(FutureGoalLanguageDataset):
    # OGBench goals paired with language for the same future state

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        idxs = np.asarray(idxs, dtype=np.int64)
        batch = self.dataset.sample(len(idxs), idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs, actor_goal_idxs = self.sample_goal_indices(idxs)
        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)
        self._attach_future_language(batch, idxs, actor_goal_idxs, evaluation)

        if not evaluation and self.config['p_aug'] is not None:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])
        return batch


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch
