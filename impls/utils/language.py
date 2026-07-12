"""Validated frozen language-embedding caches for conditioned policies."""

from functools import lru_cache
from pathlib import Path

import numpy as np


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


def evaluation_language_embedding(cache, task_id, variant, episode_index):
    if task_id < 1 or task_id > len(cache['task_ids']):
        raise ValueError(f'Language task ID {task_id} is out of range.')
    task_index = task_id - 1
    if variant == 'canonical':
        return cache['canonical_embeddings'][task_index]
    if variant == 'heldout':
        variants = cache['heldout_embeddings'][task_index]
        return variants[episode_index % len(variants)]
    raise ValueError(f'Unsupported language evaluation variant: {variant!r}')


def evaluation_language_text(cache, task_id, variant, episode_index):
    """Return the exact instruction paired with an evaluation episode."""
    if task_id < 1 or task_id > len(cache['task_ids']):
        raise ValueError(f'Language task ID {task_id} is out of range.')
    task_index = task_id - 1
    if variant == 'canonical':
        return str(cache['canonical_texts'][task_index])
    if variant == 'heldout':
        texts = cache['heldout_texts'][task_index]
        return str(texts[episode_index % len(texts)])
    raise ValueError(f'Unsupported language evaluation variant: {variant!r}')
