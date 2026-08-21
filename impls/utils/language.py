"""Validated frozen language-embedding caches for conditioned policies."""

from functools import lru_cache
from pathlib import Path

import numpy as np


SUPPORTED_LANGUAGE_EVAL_VARIANTS = ('canonical', 'heldout', 'zero', 'shuffled')
LANGUAGE_SHUFFLE_SALT = 0x4C534846  # ASCII "LSHF".


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
        scalar_optional = (
            'cache_schema_version',
            'task_spec_sha256',
            'language_contract_schema_version',
            'task_id_semantics',
            'pooling',
            'model_revision',
            'retrieval_reference',
        )
        for key in scalar_optional:
            if key in data.files:
                value = np.asarray(data[key])
                if value.shape != ():
                    raise ValueError(f'Language cache scalar {key!r} must have shape ().')
                cache[key] = value.item()
        for key in ('task_ij', 'task_coarse_ij', 'task_local_ij'):
            if key in data.files:
                cache[key] = np.asarray(data[key], dtype=np.int32)
        for key in (
            'task_atomic_task_ids',
            'task_num_operations',
            'task_family_ids',
            'atomic_task_ids',
            'atomic_task_cube_ids',
            'atomic_task_destination_types',
            'atomic_task_grid_rows',
            'atomic_task_grid_columns',
        ):
            if key in data.files:
                cache[key] = np.asarray(data[key], dtype=np.int32)
        for key in ('task_family_names',):
            if key in data.files:
                cache[key] = np.asarray(data[key]).astype(str)
        if 'official_swap_staging_xyz' in data.files:
            cache['official_swap_staging_xyz'] = np.asarray(
                data['official_swap_staging_xyz'], dtype=np.float32
            )
        for key in ('grid_world_x_edges', 'grid_world_y_edges'):
            if key in data.files:
                cache[key] = np.asarray(data[key], dtype=np.float32)
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
    for key in ('task_coarse_ij', 'task_local_ij'):
        if key in cache and cache[key].shape != (num_tasks, 2):
            raise ValueError(f"{key} has shape {cache[key].shape}; expected {(num_tasks, 2)}.")

    composite_keys = {
        'task_atomic_task_ids',
        'task_num_operations',
        'task_family_ids',
        'task_family_names',
    }
    if composite_keys.intersection(cache):
        missing = composite_keys.difference(cache)
        if missing:
            raise ValueError(
                'Composite language cache metadata is incomplete: '
                f'{sorted(missing)}.'
            )
        if cache['task_atomic_task_ids'].shape != (num_tasks, 3):
            raise ValueError('task_atomic_task_ids must have shape (num_tasks, 3).')
        for key in ('task_num_operations', 'task_family_ids', 'task_family_names'):
            if cache[key].shape != (num_tasks,):
                raise ValueError(f'{key} must have shape (num_tasks,).')
        num_operations = cache['task_num_operations']
        if np.any((num_operations < 1) | (num_operations > 3)):
            raise ValueError('Composite language tasks must contain one to three operations.')
        operation_columns = np.arange(3, dtype=np.int32)[None, :]
        used = operation_columns < num_operations[:, None]
        if np.any(cache['task_atomic_task_ids'][used] <= 0) or np.any(
            cache['task_atomic_task_ids'][~used] != -1
        ):
            raise ValueError(
                'Composite task atomic IDs must be positive in used slots and -1 in padding.'
            )
        signatures = {
            (
                int(cache['task_family_ids'][index]),
                tuple(cache['task_atomic_task_ids'][index].tolist()),
            )
            for index in range(num_tasks)
        }
        if len(signatures) != num_tasks:
            raise ValueError('Composite language task signatures must be unique.')

    atomic_keys = {
        'atomic_task_ids',
        'atomic_task_cube_ids',
        'atomic_task_destination_types',
        'atomic_task_grid_rows',
        'atomic_task_grid_columns',
    }
    if atomic_keys.intersection(cache):
        missing = atomic_keys.difference(cache)
        if missing:
            raise ValueError(
                f'Composite atomic-task lookup metadata is incomplete: {sorted(missing)}.'
            )
        primitive_count = len(cache['atomic_task_ids'])
        for key in atomic_keys:
            if cache[key].shape != (primitive_count,):
                raise ValueError(f'{key} must have shape ({primitive_count},).')
        if not np.array_equal(
            cache['atomic_task_ids'],
            np.arange(1, primitive_count + 1, dtype=np.int32),
        ):
            raise ValueError('atomic_task_ids must be consecutive and one-indexed.')
    if 'official_swap_staging_xyz' in cache:
        staging = cache['official_swap_staging_xyz']
        if staging.shape != (3,) or not np.all(np.isfinite(staging)):
            raise ValueError('official_swap_staging_xyz must be one finite XYZ coordinate.')

    grid_keys = ('grid_world_x_edges', 'grid_world_y_edges')
    if any(key in cache for key in grid_keys):
        if not all(key in cache for key in grid_keys):
            raise ValueError('Language grid X and Y edges must be provided together.')
        for key in grid_keys:
            edges = cache[key]
            if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError(f'{key} must be a strictly increasing one-dimensional array.')
        expected_shape = (
            len(cache['grid_world_x_edges']) - 1,
            len(cache['grid_world_y_edges']) - 1,
        )
        if 'task_ij' in cache:
            if np.any(cache['task_ij'] < 0) or np.any(
                cache['task_ij'] >= np.asarray(expected_shape, dtype=np.int32)
            ):
                raise ValueError(
                    f'task_ij contains cells outside language grid shape {expected_shape}.'
                )
        elif atomic_keys.issubset(cache):
            table = cache['atomic_task_destination_types'] == 0
            primitive_cells = np.stack(
                [
                    cache['atomic_task_grid_rows'],
                    cache['atomic_task_grid_columns'],
                ],
                axis=1,
            )
            if np.any(primitive_cells[table] < 0) or np.any(
                primitive_cells[table]
                >= np.asarray(expected_shape, dtype=np.int32)
            ):
                raise ValueError(
                    'Composite primitive table cells lie outside language grid '
                    f'shape {expected_shape}.'
                )
            if np.any(primitive_cells[~table] != -1):
                raise ValueError(
                    'Non-table composite primitives must use -1 grid-cell metadata.'
                )
        else:
            raise ValueError(
                'Language grid edges require either per-task task_ij or '
                'composite primitive lookup metadata.'
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


def language_task_id_for_goal_xyz(cache, goal_xyz, fallback_task_id=None):
    """Map a Cube goal coordinate to the unique cached language-grid task."""
    grid_keys = ('grid_world_x_edges', 'grid_world_y_edges')
    if 'task_ij' not in cache or not all(key in cache for key in grid_keys):
        if fallback_task_id is None:
            raise ValueError(
                'Language cache has no coordinate grid and no fallback task ID was supplied.'
            )
        return int(fallback_task_id)


    grid_dtype = np.result_type(
        cache['grid_world_x_edges'].dtype,
        cache['grid_world_y_edges'].dtype,
    )
    goal_xyz = np.asarray(goal_xyz, dtype=grid_dtype)
    if goal_xyz.shape != (3,) or not np.all(np.isfinite(goal_xyz)):
        raise ValueError(f'Cube goal_xyz must have shape (3,), got {goal_xyz.shape}.')
    x_edges = np.asarray(cache['grid_world_x_edges'], dtype=grid_dtype)
    y_edges = np.asarray(cache['grid_world_y_edges'], dtype=grid_dtype)
    x, y = map(float, goal_xyz[:2])
    if not (x_edges[0] <= x <= x_edges[-1] and y_edges[0] <= y <= y_edges[-1]):
        raise ValueError(f'Cube goal {goal_xyz.tolist()} lies outside the language grid.')
    row = int(np.searchsorted(x_edges[1:-1], x, side='right'))
    column = int(np.searchsorted(y_edges[1:-1], y, side='right'))
    matches = np.flatnonzero(np.all(cache['task_ij'] == (row, column), axis=1))
    if len(matches) != 1:
        raise ValueError(
            f'Language grid cell {(row, column)} is absent or duplicated in the cache.'
        )
    return int(matches[0] + 1)


def language_task_id_for_atomic_sequence(cache, family_id, atomic_task_ids):
    """map an ordered Double-Cube primitive sequence to its language task id"""
    required = {'task_atomic_task_ids', 'task_num_operations', 'task_family_ids'}
    missing = required.difference(cache)
    if missing:
        raise ValueError(
            'Language cache has no ordered-composite contract: '
            f'{sorted(missing)}.'
        )
    atomic_task_ids = np.asarray(tuple(atomic_task_ids), dtype=np.int32)
    if atomic_task_ids.ndim != 1 or not 1 <= len(atomic_task_ids) <= 3:
        raise ValueError('An ordered atomic sequence must contain one to three task IDs.')
    padded = np.full(3, -1, dtype=np.int32)
    padded[: len(atomic_task_ids)] = atomic_task_ids
    matches = np.flatnonzero(
        (cache['task_family_ids'] == int(family_id))
        & (cache['task_num_operations'] == len(atomic_task_ids))
        & np.all(cache['task_atomic_task_ids'] == padded[None, :], axis=1)
    )
    if len(matches) != 1:
        raise ValueError(
            'Ordered Double-Cube language signature is absent or duplicated: '
            f'family={int(family_id)}, atomic_task_ids={atomic_task_ids.tolist()}.'
        )
    return int(cache['task_ids'][matches[0]])


def _validate_evaluation_request(cache, task_id, variant, episode_index):
    if task_id < 1 or task_id > len(cache['task_ids']):
        raise ValueError(f'Language task ID {task_id} is out of range.')
    if variant not in SUPPORTED_LANGUAGE_EVAL_VARIANTS:
        raise ValueError(f'Unsupported language evaluation variant: {variant!r}')
    if episode_index < 0:
        raise ValueError('Language evaluation episode_index must be nonnegative.')


@lru_cache(maxsize=1024)
def _shuffled_wrong_task_ids(task_id, num_tasks):
    """return one reproducible, unbiased ordering of every wrong task ID."""
    if not (1 <= task_id <= num_tasks):
        raise ValueError(f'Language task ID {task_id} is out of range.')
    candidates = np.arange(1, num_tasks + 1, dtype=np.int32)
    candidates = candidates[candidates != task_id]
    seed = np.random.SeedSequence(
        [int(task_id), int(num_tasks), int(LANGUAGE_SHUFFLE_SALT)]
    )
    shuffled = np.random.default_rng(seed).permutation(candidates)
    shuffled.setflags(write=False)
    return shuffled


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
    wrong_task_ids = _shuffled_wrong_task_ids(int(task_id), int(num_tasks))
    return int(wrong_task_ids[episode_index % len(wrong_task_ids)])


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
