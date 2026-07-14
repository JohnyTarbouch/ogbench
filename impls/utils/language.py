"""Validated frozen language-embedding caches for conditioned policies."""

from functools import lru_cache
from pathlib import Path

import numpy as np


SUPPORTED_LANGUAGE_EVAL_VARIANTS = ('canonical', 'heldout', 'zero', 'shuffled')


@lru_cache(maxsize=8)
def load_language_cache(path, num_tasks, embedding_dim):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f'Language embedding cache not found: {resolved}')

    required = {
        'task_ids',
        'canonical_texts',
        'canonical_embeddings',
        'train_texts',
        'train_embeddings',
        'heldout_texts',
        'heldout_embeddings',
    }
    with np.load(resolved, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f'Language cache is missing keys: {sorted(missing)}')
        cache = {key: np.asarray(data[key]) for key in required}
        cache['model_name'] = str(data['model_name']) if 'model_name' in data.files else 'unknown'
        cache['normalized'] = bool(data['normalized']) if 'normalized' in data.files else False
        if 'task_ij' in data.files:
            cache['task_ij'] = np.asarray(data['task_ij'], dtype=np.int32)
        if 'task_spec_sha256' in data.files:
            cache['task_spec_sha256'] = str(data['task_spec_sha256'])
        if 'pooling' in data.files:
            cache['pooling'] = str(data['pooling'])
        for split in ('train', 'heldout'):
            key = f'{split}_nearest_task_ids'
            if key in data.files:
                cache[key] = np.asarray(data[key], dtype=np.int32)

    expected_task_ids = np.arange(1, num_tasks + 1, dtype=np.int32)
    if not np.array_equal(cache['task_ids'], expected_task_ids):
        raise ValueError(f'Language cache task IDs must equal {expected_task_ids.tolist()}.')

    expected_shapes = {
        'canonical_texts': (num_tasks,),
        'canonical_embeddings': (num_tasks, embedding_dim),
    }
    for key, expected in expected_shapes.items():
        if cache[key].shape != expected:
            raise ValueError(f'{key} has shape {cache[key].shape}; expected {expected}.')
    for prefix in ('train', 'heldout'):
        texts = cache[f'{prefix}_texts']
        embeddings = cache[f'{prefix}_embeddings']
        if texts.ndim != 2 or texts.shape[0] != num_tasks:
            raise ValueError(f'{prefix}_texts must have shape (num_tasks, variants).')
        if embeddings.shape != (*texts.shape, embedding_dim):
            raise ValueError(
                f'{prefix}_embeddings has shape {embeddings.shape}; '
                f'expected {(*texts.shape, embedding_dim)}.'
            )

    if 'task_ij' in cache:
        if cache['task_ij'].shape != (num_tasks, 2):
            raise ValueError(f"task_ij has shape {cache['task_ij'].shape}; expected {(num_tasks, 2)}.")
        if len(np.unique(cache['task_ij'], axis=0)) != num_tasks:
            raise ValueError('task_ij must contain one unique cell per language task.')

    for key in ('canonical_embeddings', 'train_embeddings', 'heldout_embeddings'):
        embeddings = np.asarray(cache[key], dtype=np.float32)
        if not np.all(np.isfinite(embeddings)):
            raise ValueError(f'{key} contains non-finite values.')
        if cache['normalized']:
            norms = np.linalg.norm(embeddings, axis=-1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ValueError(f'{key} is marked normalized but has invalid norms.')
        embeddings.setflags(write=False)
        cache[key] = embeddings

    cache['path'] = str(resolved)
    return cache


def language_retrieval_top1(cache, split):
    """Return cached task-retrieval accuracy, or ``None`` if unavailable."""
    if split not in {'train', 'heldout'}:
        raise ValueError("split must be 'train' or 'heldout'.")
    key = f'{split}_nearest_task_ids'
    if key not in cache:
        return None
    nearest = np.asarray(cache[key], dtype=np.int32)
    expected_shape = cache[f'{split}_texts'].shape
    if nearest.shape != expected_shape:
        raise ValueError(f'{key} has shape {nearest.shape}; expected {expected_shape}.')
    expected = np.broadcast_to(cache['task_ids'][:, None], expected_shape)
    return float(np.mean(nearest == expected))


def language_task_id_for_goal(cache, goal_ij, fallback_task_id=None):
    """
    Map environment goal cell to its language task ID
    """
    if 'task_ij' not in cache:
        if fallback_task_id is None:
            raise ValueError('Language cache has no task_ij mapping and no fallback task ID was supplied.')
        return int(fallback_task_id)
    goal_ij = np.asarray(goal_ij, dtype=np.int32)
    if goal_ij.shape != (2,):
        raise ValueError(f'Environment goal_ij must have shape (2,), got {goal_ij.shape}.')
    matches = np.flatnonzero(np.all(cache['task_ij'] == goal_ij, axis=1))
    if len(matches) != 1:
        raise ValueError(f'Environment goal cell {goal_ij.tolist()} is absent or duplicated in the language cache.')
    return int(matches[0] + 1)


def _validate_evaluation_request(cache, task_id, variant, episode_index):
    if task_id < 1 or task_id > len(cache['task_ids']):
        raise ValueError(f'Language task ID {task_id} is out of range.')
    if variant not in SUPPORTED_LANGUAGE_EVAL_VARIANTS:
        raise ValueError(f'Unsupported language evaluation variant: {variant!r}')
    if episode_index < 0:
        raise ValueError('Language evaluation episode_index must be nonnegative.')


def evaluation_language_condition_task_id(cache, task_id, variant, episode_index):
    """
    Return the task ID represented by an evaluation language condition.
    """
    _validate_evaluation_request(cache, task_id, variant, episode_index)
    if variant in {'canonical', 'heldout'}:
        return int(task_id)
    if variant == 'zero':
        return 0
    num_tasks = len(cache['task_ids'])
    if num_tasks < 2:
        raise ValueError('Shuffled-language evaluation requires at least two tasks.')
    offset = episode_index % (num_tasks - 1) + 1
    return int((task_id - 1 + offset) % num_tasks + 1)


def evaluation_language_condition(cache, task_id, variant, episode_index):
    condition_task_id = evaluation_language_condition_task_id(
        cache, task_id, variant, episode_index
    )
    if variant == 'zero':
        embedding = np.zeros_like(cache['canonical_embeddings'][task_id - 1], dtype=np.float32)
        return embedding, '<zero-language-embedding>', condition_task_id

    condition_index = condition_task_id - 1
    if variant in {'canonical', 'shuffled'}:
        return (
            cache['canonical_embeddings'][condition_index],
            str(cache['canonical_texts'][condition_index]),
            condition_task_id,
        )

    embeddings = cache['heldout_embeddings'][condition_index]
    texts = cache['heldout_texts'][condition_index]
    if len(embeddings) == 0 or len(texts) == 0:
        raise ValueError('Held-out language evaluation requires at least one variant.')
    variant_index = episode_index % len(embeddings)
    return embeddings[variant_index], str(texts[variant_index]), condition_task_id


def evaluation_language_embedding(cache, task_id, variant, episode_index):
    return evaluation_language_condition(cache, task_id, variant, episode_index)[0]


def evaluation_language_text(cache, task_id, variant, episode_index):
    """Return the exact instruction paired with an evaluation episode."""
    return evaluation_language_condition(cache, task_id, variant, episode_index)[1]
