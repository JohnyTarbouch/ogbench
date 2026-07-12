import dataclasses
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from utils.language import load_language_cache


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


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
    rng = np.random.default_rng() if rng is None else rng

    for key in keys:
        arr = batch[key]
        if getattr(arr, 'ndim', 0) != 4:
            continue

        n = arr.shape[0]
        shifts = rng.integers(0, 2 * pad + 1, size=(n, 2))
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
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
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


def _initialize_language_conditioning(owner):
    # Load the language embedding 
    # set up the language conditioning
    cache = load_language_cache(
        owner.config['language_embedding_path'],
        int(owner.config['num_language_tasks']),
        int(owner.config['language_embedding_dim']),
    )
    train_variant = str(owner.config.get('language_train_variant', 'canonical'))
    if train_variant not in {'canonical', 'train'}:
        raise ValueError("language_train_variant must be 'canonical' or 'train'.")
    owner.language_cache = cache
    owner.language_train_variant = train_variant
    owner._language_task_counts = np.zeros(int(owner.config['num_language_tasks']), dtype=np.int64)
    owner._language_variant_counts = np.zeros(
        1 if train_variant == 'canonical' else cache['train_embeddings'].shape[1],
        dtype=np.int64,
    )
    owner.language_summary = {
        'path': cache['path'],
        'model_name': cache['model_name'],
        'normalized': cache['normalized'],
        'embedding_dim': int(cache['canonical_embeddings'].shape[-1]),
        'train_variant': train_variant,
        'train_variants_per_task': int(cache['train_embeddings'].shape[1]),
        'heldout_variants_per_task': int(cache['heldout_embeddings'].shape[1]),
    }


def _attach_language_condition(owner, batch, task_ids, evaluation):
    task_ids = np.asarray(task_ids, dtype=np.int64)
    num_tasks = int(owner.config['num_language_tasks'])
    if np.any(task_ids < 1) or np.any(task_ids > num_tasks):
        raise ValueError(f'Language task IDs must be in [1, {num_tasks}].')
    task_rows = task_ids - 1
    if evaluation or owner.language_train_variant == 'canonical':
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
    metrics = {'data/language_samples': float(total)}
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
    language-task label. Keeping the original dataset intact lets frame
    stacking use the true episode boundaries without copying image observations.
    """

    # Only a small manifest-selected subset is sampled. Building stacks for
    # every raw frame would need tens of gigabytes without changing a batch.
    preprocess_frame_stack: bool = False
    manifest_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        super().__post_init__()
        self.raw_size = self.dataset.size
        if self.manifest_path is None:
            raise ValueError('AtomicLanguageDataset requires manifest_path.')

        manifest_path = Path(self.manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Atomic segment manifest not found: {manifest_path}')

        required = {'transition_indices', 'transition_segment_ids', 'task_id'}
        with np.load(manifest_path, allow_pickle=False) as manifest:
            missing = required.difference(manifest.files)
            if missing:
                raise ValueError(f'Atomic segment manifest is missing keys: {sorted(missing)}')
            transition_indices = np.asarray(manifest['transition_indices'], dtype=np.int64)
            transition_segment_ids = np.asarray(manifest['transition_segment_ids'], dtype=np.int64)
            segment_task_ids = np.asarray(manifest['task_id'], dtype=np.int64)
            segment_episode_ids = (
                np.asarray(manifest['episode_id'], dtype=np.int64)
                if 'episode_id' in manifest.files
                else None
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

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, transition_indices)]
        if np.any(transition_indices >= final_state_idxs):
            raise ValueError('Atomic manifest includes a terminal transition without an in-episode successor.')

        task_ids = segment_task_ids[transition_segment_ids]
        num_language_tasks = int(self.config['num_language_tasks'])
        if np.any(task_ids < 1) or np.any(task_ids > num_language_tasks):
            raise ValueError(f'Atomic task IDs must be in [1, {num_language_tasks}].')

        self.manifest_path = manifest_path
        self.transition_indices = transition_indices
        self.transition_segment_ids = transition_segment_ids.astype(np.int32)
        self.transition_task_ids = task_ids.astype(np.int32)
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
            'raw_dataset_size': int(self.raw_size),
            'num_transitions': int(self.size),
            'num_segments': int(len(used_segment_ids)),
            'num_episodes': int(len(used_episode_ids)),
            'transitions_per_task': {
                str(int(task_id)): int(count)
                for task_id, count in zip(task_values, task_counts)
            },
        }

    def _sample_atomic(self, batch_size, idxs=None, evaluation=False):
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

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations'])

        return batch, self.transition_task_ids[idxs]


@dataclasses.dataclass
class AtomicLanguageDataset(_AtomicSegmentDataset):
    """Atomic Cube transitions conditioned on frozen language embeddings."""

    def __post_init__(self):
        super().__post_init__()
        mode = str(self.config.get('language_dataset_mode', 'atomic_movement'))
        if mode != 'atomic_movement':
            raise ValueError("AtomicLanguageDataset requires language_dataset_mode='atomic_movement'.")
        _initialize_language_conditioning(self)

    def sample(self, batch_size, idxs=None, evaluation=False):
        batch, task_ids = self._sample_atomic(batch_size, idxs=idxs, evaluation=evaluation)
        return _attach_language_condition(self, batch, task_ids, evaluation)

    def get_and_reset_diagnostics(self):
        """Report the task and paraphrase mixture actually seen by training."""
        return _get_and_reset_language_diagnostics(self)


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
            state_task_ids = np.asarray(labels['state_task_ids'], dtype=np.int32)
            task_ids = np.asarray(labels['task_ids'], dtype=np.int32)
            source_num_states = int(labels['source_num_states'])
            outside_grid_states = int(labels['num_outside_grid_states']) if 'num_outside_grid_states' in labels else 0

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

        self.labels_path = labels_path
        self.state_task_ids = state_task_ids
        task_values, task_counts = np.unique(state_task_ids, return_counts=True)
        self.future_label_summary = {
            'path': str(labels_path),
            'raw_dataset_size': int(self.dataset.size),
            'num_outside_grid_states': outside_grid_states,
            'states_per_task': {
                str(int(task_id)): int(count)
                for task_id, count in zip(task_values, task_counts)
            },
        }
        self._future_goal_count = 0
        self._future_goal_offset_sum = 0
        self._future_goal_same_state_count = 0
        self._future_goal_same_cell_count = 0
        _initialize_language_conditioning(self)

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        idxs = np.asarray(idxs, dtype=np.int64)
        batch = self.dataset.sample(len(idxs), idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        _, actor_goal_idxs = self.sample_goal_indices(idxs)
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
            if self.config['p_aug'] is not None and np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

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
