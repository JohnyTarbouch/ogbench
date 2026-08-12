from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from utils.language import (
    evaluation_language_condition,
    language_task_id_for_goal,
    language_task_id_for_goal_xyz,
    load_language_cache,
)


EVAL_DIAGNOSTIC_METRICS = (
    'ever_success',
    'episode_length',
    'initial_goal_distance',
    'final_goal_distance',
    'minimum_goal_distance',
    'goal_progress',
    'max_block_displacement',
    'max_block_height_delta',
    'min_effector_block_distance',
    'gripper_contact_fraction',
    'likely_object_contact_fraction',
    'gripper_closed_fraction',
    'mean_action_norm',
    'movement_alignment',
    'first_move_detected',
    'red_moved_first',
    'blue_moved_first',
    'first_move_tie',
    'first_cube_reached_target',
    'second_cube_started_before_first_target',
    'second_cube_approached_after_first_target',
    'second_cube_contacted_after_first_target',
    'second_cube_moved_after_first_target',
    'second_cube_reached_target_after_first_target',
    'first_cube_retained_at_end_given_reached',
    'first_cube_lost_after_target_given_reached',
    'cube_0_initial_goal_distance',
    'cube_0_final_goal_distance',
    'cube_0_minimum_goal_distance',
    'cube_0_max_displacement',
    'cube_0_ever_at_target',
    'cube_0_final_at_target',
    'cube_1_initial_goal_distance',
    'cube_1_final_goal_distance',
    'cube_1_minimum_goal_distance',
    'cube_1_max_displacement',
    'cube_1_ever_at_target',
    'cube_1_final_at_target',
)

CONTACT_SIGNAL_THRESHOLD = 0.1
LIKELY_OBJECT_CONTACT_DISTANCE = 0.08
CUBE_TARGET_DISTANCE_THRESHOLD = 0.04
CUBE_MOVEMENT_THRESHOLD = 0.03

FAILURE_MODES = (
    'success',
    'lost_after_success',
    'unclassified',
    'no_contact_or_object_motion',
    'contact_without_object_motion',
    'near_goal_at_timeout',
    'lifted_but_not_placed',
    'moved_opposite_goal',
    'moved_away_from_goal',
    'partial_progress',
    'no_clear_progress',
)


def _info_vector(info, key):
    value = info.get(key)
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return value if value.size else None


def _task_goal_xyzs(env):
    """
    Return the actual target assigned to each physical/color cube.
    """
    unwrapped = env.unwrapped
    target_ids = getattr(unwrapped, '_cube_target_mocap_ids', None)
    data = getattr(unwrapped, '_data', None)
    if target_ids is None or data is None or not hasattr(data, 'mocap_pos'):
        return None
    target_ids = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    if len(target_ids) == 0:
        return None
    mocap_positions = np.asarray(data.mocap_pos, dtype=np.float64)
    if mocap_positions.ndim != 2 or mocap_positions.shape[1] < 3:
        return None
    if np.any(target_ids < 0) or np.any(target_ids >= len(mocap_positions)):
        return None
    return mocap_positions[target_ids, :3].copy()


def _task_goal_xyz(env):
    goal_xyzs = _task_goal_xyzs(env)
    if goal_xyzs is None:
        return None
    unwrapped = env.unwrapped
    target_block = int(getattr(unwrapped, '_target_block', 0))
    target_block = min(max(target_block, 0), len(goal_xyzs) - 1)
    return goal_xyzs[target_block, :3].copy()


def _first_true_step(mask, start=0):
    """Return the first true state index at or after start, or -1."""
    mask = np.asarray(mask, dtype=bool)
    if start >= len(mask):
        return -1
    matches = np.flatnonzero(mask[start:])
    return int(matches[0] + start) if len(matches) else -1


def _cube_sequence_diagnostics(cube_positions, goal_xyzs, effector_positions, contacts):
    """Measure the two-cube completion funnel using physical cube IDs
    """
    summary = {
        'num_cubes_tracked': 0,
        'num_required_cubes': 0,
        'first_moved_cube_id': -1,
        'first_move_step': -1,
        'first_move_detected': 0.0,
        'red_moved_first': 0.0,
        'blue_moved_first': 0.0,
        'first_move_tie': 0.0,
        'first_cube_target_step': -1,
        'first_cube_reached_target': 0.0,
        'second_cube_id': -1,
        'second_cube_approach_step': -1,
        'second_cube_contact_step': -1,
        'second_cube_post_target_move_step': -1,
        'second_cube_post_target_reach_step': -1,
        'second_cube_started_before_first_target': np.nan,
        'second_cube_approached_after_first_target': np.nan,
        'second_cube_contacted_after_first_target': np.nan,
        'second_cube_moved_after_first_target': np.nan,
        'second_cube_reached_target_after_first_target': np.nan,
        'first_cube_retained_at_end_given_reached': np.nan,
        'first_cube_lost_after_target_given_reached': np.nan,
    }
    for cube_id in range(2):
        summary.update(
            {
                f'cube_{cube_id}_goal_x': np.nan,
                f'cube_{cube_id}_goal_y': np.nan,
                f'cube_{cube_id}_goal_z': np.nan,
                f'cube_{cube_id}_required': np.nan,
                f'cube_{cube_id}_initial_goal_distance': np.nan,
                f'cube_{cube_id}_final_goal_distance': np.nan,
                f'cube_{cube_id}_minimum_goal_distance': np.nan,
                f'cube_{cube_id}_max_displacement': np.nan,
                f'cube_{cube_id}_move_step': -1,
                f'cube_{cube_id}_target_step': -1,
                f'cube_{cube_id}_ever_at_target': np.nan,
                f'cube_{cube_id}_final_at_target': np.nan,
            }
        )

    cube_positions = np.asarray(cube_positions, dtype=np.float64)
    goal_xyzs = np.asarray(goal_xyzs, dtype=np.float64)
    if (
        cube_positions.ndim != 3
        or goal_xyzs.ndim != 2
        or cube_positions.shape[0] == 0
        or cube_positions.shape[1] == 0
        or cube_positions.shape[2] < 3
        or goal_xyzs.shape[1] < 3
    ):
        return summary

    num_cubes = min(cube_positions.shape[1], goal_xyzs.shape[0])
    cube_positions = cube_positions[:, :num_cubes, :3]
    goal_xyzs = goal_xyzs[:num_cubes, :3]
    distances = np.linalg.norm(cube_positions - goal_xyzs[None, :, :], axis=-1)
    displacements = np.linalg.norm(cube_positions - cube_positions[0:1], axis=-1)
    required = distances[0] > CUBE_TARGET_DISTANCE_THRESHOLD
    move_steps = np.asarray(
        [
            _first_true_step(displacements[:, cube_id] >= CUBE_MOVEMENT_THRESHOLD, start=1)
            for cube_id in range(num_cubes)
        ],
        dtype=np.int32,
    )
    target_steps = np.asarray(
        [
            _first_true_step(distances[:, cube_id] <= CUBE_TARGET_DISTANCE_THRESHOLD)
            for cube_id in range(num_cubes)
        ],
        dtype=np.int32,
    )

    summary['num_cubes_tracked'] = int(num_cubes)
    summary['num_required_cubes'] = int(np.sum(required))
    for cube_id in range(min(num_cubes, 2)):
        summary.update(
            {
                f'cube_{cube_id}_goal_x': float(goal_xyzs[cube_id, 0]),
                f'cube_{cube_id}_goal_y': float(goal_xyzs[cube_id, 1]),
                f'cube_{cube_id}_goal_z': float(goal_xyzs[cube_id, 2]),
                f'cube_{cube_id}_required': float(required[cube_id]),
                f'cube_{cube_id}_initial_goal_distance': float(distances[0, cube_id]),
                f'cube_{cube_id}_final_goal_distance': float(distances[-1, cube_id]),
                f'cube_{cube_id}_minimum_goal_distance': float(np.min(distances[:, cube_id])),
                f'cube_{cube_id}_max_displacement': float(np.max(displacements[:, cube_id])),
                f'cube_{cube_id}_move_step': int(move_steps[cube_id]),
                f'cube_{cube_id}_target_step': int(target_steps[cube_id]),
                f'cube_{cube_id}_ever_at_target': float(target_steps[cube_id] >= 0),
                f'cube_{cube_id}_final_at_target': float(
                    distances[-1, cube_id] <= CUBE_TARGET_DISTANCE_THRESHOLD
                ),
            }
        )

    movable_required = np.flatnonzero(required & (move_steps >= 0))
    if len(movable_required) == 0:
        return summary
    first_step = int(np.min(move_steps[movable_required]))
    first_candidates = movable_required[move_steps[movable_required] == first_step]
    summary['first_move_step'] = first_step
    summary['first_move_detected'] = 1.0
    if len(first_candidates) != 1:
        summary['first_moved_cube_id'] = 2
        summary['first_move_tie'] = 1.0
        return summary

    first_cube = int(first_candidates[0])
    summary['first_moved_cube_id'] = first_cube
    summary['red_moved_first'] = float(first_cube == 0)
    summary['blue_moved_first'] = float(first_cube == 1)

    first_target_step = _first_true_step(
        distances[:, first_cube] <= CUBE_TARGET_DISTANCE_THRESHOLD,
        start=first_step,
    )
    summary['first_cube_target_step'] = first_target_step
    summary['first_cube_reached_target'] = float(first_target_step >= 0)
    if first_target_step >= 0:
        summary['first_cube_retained_at_end_given_reached'] = float(
            distances[-1, first_cube] <= CUBE_TARGET_DISTANCE_THRESHOLD
        )
        summary['first_cube_lost_after_target_given_reached'] = float(
            np.any(distances[first_target_step:, first_cube] > CUBE_TARGET_DISTANCE_THRESHOLD)
        )

    other_required = np.flatnonzero(required & (np.arange(num_cubes) != first_cube))
    if len(other_required) != 1:
        return summary
    second_cube = int(other_required[0])
    summary['second_cube_id'] = second_cube
    if first_target_step < 0:
        summary.update(
            {
                'second_cube_approached_after_first_target': 0.0,
                'second_cube_contacted_after_first_target': 0.0,
                'second_cube_moved_after_first_target': 0.0,
                'second_cube_reached_target_after_first_target': 0.0,
            }
        )
        return summary

    summary['second_cube_started_before_first_target'] = float(
        move_steps[second_cube] >= 0 and move_steps[second_cube] < first_target_step
    )
    start = first_target_step + 1
    effector_positions = np.asarray(effector_positions, dtype=np.float64)
    contacts = np.asarray(contacts, dtype=np.float64).reshape(-1)
    if effector_positions.shape == (len(cube_positions), 3):
        second_effector_distances = np.linalg.norm(
            effector_positions - cube_positions[:, second_cube], axis=-1
        )
        finite_approach = np.isfinite(second_effector_distances)
        approach_step = _first_true_step(
            finite_approach
            & (second_effector_distances <= LIKELY_OBJECT_CONTACT_DISTANCE),
            start=start,
        )
        summary['second_cube_approach_step'] = approach_step
        summary['second_cube_approached_after_first_target'] = float(approach_step >= 0)
        if len(contacts) == len(cube_positions):
            contact_step = _first_true_step(
                finite_approach
                & (second_effector_distances <= LIKELY_OBJECT_CONTACT_DISTANCE)
                & np.isfinite(contacts)
                & (contacts >= CONTACT_SIGNAL_THRESHOLD),
                start=start,
            )
            summary['second_cube_contact_step'] = contact_step
            summary['second_cube_contacted_after_first_target'] = float(contact_step >= 0)
        else:
            summary['second_cube_contacted_after_first_target'] = 0.0
    else:
        summary['second_cube_approached_after_first_target'] = 0.0
        summary['second_cube_contacted_after_first_target'] = 0.0

    post_target_displacement = np.linalg.norm(
        cube_positions[:, second_cube] - cube_positions[first_target_step, second_cube],
        axis=-1,
    )
    second_move_step = _first_true_step(
        post_target_displacement >= CUBE_MOVEMENT_THRESHOLD,
        start=start,
    )
    second_reach_step = _first_true_step(
        distances[:, second_cube] <= CUBE_TARGET_DISTANCE_THRESHOLD,
        start=start,
    )
    summary['second_cube_post_target_move_step'] = second_move_step
    summary['second_cube_post_target_reach_step'] = second_reach_step
    summary['second_cube_moved_after_first_target'] = float(second_move_step >= 0)
    summary['second_cube_reached_target_after_first_target'] = float(second_reach_step >= 0)
    return summary


def _failure_mode(summary):
    """Assign a deliberately coarse, heuristic Cube failure category."""
    if summary['success']:
        return 'success'
    if summary['ever_success']:
        return 'lost_after_success'
    if not np.isfinite(summary['final_goal_distance']):
        return 'unclassified'
    if summary['max_block_displacement'] < 0.015:
        if summary['likely_object_contact_fraction'] <= 0:
            return 'no_contact_or_object_motion'
        return 'contact_without_object_motion'
    if summary['minimum_goal_distance'] <= 0.06:
        return 'near_goal_at_timeout'
    if summary['max_block_height_delta'] > 0.03:
        return 'lifted_but_not_placed'
    if np.isfinite(summary['movement_alignment']) and summary['movement_alignment'] < 0:
        return 'moved_opposite_goal'
    if summary['goal_progress'] < -0.01:
        return 'moved_away_from_goal'
    if summary['goal_progress'] > 0.01:
        return 'partial_progress'
    return 'no_clear_progress'


def summarize_episode(initial_info, traj, goal_xyz=None, goal_xyzs=None):
    """Summarize Cube behavior, including the Double-Cube completion funnel."""
    step_infos = list(traj.get('info', []))
    infos = [initial_info, *step_infos]
    actions = np.asarray(traj.get('action', []), dtype=np.float64)
    if actions.ndim == 1 and actions.size:
        actions = actions[:, None]

    if goal_xyzs is None and goal_xyz is not None:
        goal_xyzs = np.asarray(goal_xyz, dtype=np.float64).reshape(1, -1)
    elif goal_xyzs is not None:
        goal_xyzs = np.asarray(goal_xyzs, dtype=np.float64)
        if goal_xyzs.ndim == 1:
            goal_xyzs = goal_xyzs.reshape(1, -1)
    if (
        goal_xyzs is None
        or goal_xyzs.ndim != 2
        or goal_xyzs.shape[1] < 3
        or len(goal_xyzs) == 0
    ):
        goal_xyzs = np.empty((0, 3), dtype=np.float64)
    else:
        goal_xyzs = goal_xyzs[:, :3]
        
    cube_positions = []
    if len(goal_xyzs):
        for info in infos:
            row = []
            for cube_id in range(len(goal_xyzs)):
                value = _info_vector(info, f'privileged/block_{cube_id}_pos')
                if value is None or value.size < 3:
                    row = []
                    break
                row.append(value[:3])
            if not row:
                cube_positions = []
                break
            cube_positions.append(row)
    cube_positions = np.asarray(cube_positions, dtype=np.float64)
    if cube_positions.ndim == 3 and cube_positions.shape[1] > 0:
        block_positions = cube_positions[:, 0, :3]
    else:
        block_positions = np.empty((0, 3), dtype=np.float64)

    has_geometry = len(goal_xyzs) > 0 and len(block_positions) == len(infos)
    if has_geometry:
        primary_goal_xyz = goal_xyzs[0]
        goal_distances = np.linalg.norm(block_positions - primary_goal_xyz, axis=-1)
        block_displacements = np.linalg.norm(block_positions - block_positions[0], axis=-1)
        movement = block_positions[-1] - block_positions[0]
        goal_direction = primary_goal_xyz - block_positions[0]
        denominator = np.linalg.norm(movement) * np.linalg.norm(goal_direction)
        movement_alignment = (
            float(np.dot(movement, goal_direction) / denominator)
            if denominator > 1e-8
            else np.nan
        )
    else:
        goal_distances = block_displacements = np.empty(0)
        movement_alignment = np.nan

    effector_positions = np.full((len(infos), 3), np.nan, dtype=np.float64)
    aligned_contacts = np.full(len(infos), np.nan, dtype=np.float64)
    effector_block_distances = []
    contacts = []
    likely_object_contacts = []
    openings = []
    for state_index, info in enumerate(infos):
        block = _info_vector(info, 'privileged/block_0_pos')
        effector = _info_vector(info, 'proprio/effector_pos')
        effector_block_distance = None
        if effector is not None and effector.size >= 3:
            effector_positions[state_index] = effector[:3]
        if block is not None and effector is not None and block.size >= 3 and effector.size >= 3:
            effector_block_distance = float(np.linalg.norm(block[:3] - effector[:3]))
            effector_block_distances.append(effector_block_distance)
        contact = _info_vector(info, 'proprio/gripper_contact')
        if contact is not None:
            aligned_contacts[state_index] = float(contact[0])
            contacts.append(float(contact[0]))
            if effector_block_distance is not None:
                likely_object_contacts.append(
                    contact[0] >= CONTACT_SIGNAL_THRESHOLD
                    and effector_block_distance <= LIKELY_OBJECT_CONTACT_DISTANCE
                )
        opening = _info_vector(info, 'proprio/gripper_opening')
        if opening is not None:
            openings.append(float(opening[0]))

    successes = [bool(info.get('success', False)) for info in step_infos]
    success = bool(successes[-1]) if successes else False
    ever_success = any(successes)
    initial_distance = float(goal_distances[0]) if len(goal_distances) else np.nan
    final_distance = float(goal_distances[-1]) if len(goal_distances) else np.nan
    minimum_distance = float(np.min(goal_distances)) if len(goal_distances) else np.nan
    progress = initial_distance - final_distance if np.isfinite(initial_distance + final_distance) else np.nan
    contacts = np.asarray(contacts, dtype=np.float64)
    openings = np.asarray(openings, dtype=np.float64)
    action_norms = np.linalg.norm(actions, axis=-1) if actions.size else np.empty(0)

    summary = {
        'success': float(success),
        'ever_success': float(ever_success),
        'episode_length': int(len(step_infos)),
        'initial_goal_distance': initial_distance,
        'final_goal_distance': final_distance,
        'minimum_goal_distance': minimum_distance,
        'goal_progress': float(progress),
        'max_block_displacement': float(np.max(block_displacements)) if len(block_displacements) else np.nan,
        'max_block_height_delta': (
            float(np.max(block_positions[:, 2]) - block_positions[0, 2]) if has_geometry else np.nan
        ),
        'min_effector_block_distance': float(np.min(effector_block_distances)) if effector_block_distances else np.nan,
        'gripper_contact_fraction': (
            float(np.mean(contacts >= CONTACT_SIGNAL_THRESHOLD)) if len(contacts) else np.nan
        ),
        'likely_object_contact_fraction': (
            float(np.mean(likely_object_contacts)) if likely_object_contacts else 0.0
        ),
        'gripper_closed_fraction': float(np.mean(openings <= 0.2)) if len(openings) else np.nan,
        'mean_action_norm': float(np.mean(action_norms)) if len(action_norms) else np.nan,
        'movement_alignment': movement_alignment,
    }
    summary.update(
        _cube_sequence_diagnostics(
            cube_positions,
            goal_xyzs,
            effector_positions,
            aligned_contacts,
        )
    )
    summary['failure_mode'] = _failure_mode(summary)
    return summary


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    language_eval_variant='canonical',
    collect_diagnostics=False,
    evaluation_seed=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.
        language_eval_variant: Canonical or held-out language condition to use.
        collect_diagnostics: Whether to add Cube-specific behavioral diagnostics to
        the always-recorded basic episode conditions and outcomes.

    Returns:
        A tuple containing statistics, trajectories, rendered videos, and episode summaries.
    """
    trajs = []
    stats = defaultdict(list)
    policy_conditioning = config.get('policy_conditioning')
    uses_language = policy_conditioning in {'language', 'goal_language'}
    language_cache = None
    if uses_language:
        language_cache = load_language_cache(
            config['language_embedding_path'],
            int(config['num_language_tasks']),
            int(config['language_embedding_dim']),
        )

    renders = []
    episode_summaries = []
    numpy_state = np.random.get_state()
    try:
        episode_iterator = trange(num_eval_episodes + num_video_episodes)
        for i in episode_iterator:
            traj = defaultdict(list)
            should_render = i >= num_eval_episodes
            episode_seed = int(
                np.random.SeedSequence([int(evaluation_seed), int(task_id or 0), int(i)])
                .generate_state(1, dtype=np.uint32)[0]
            )
            # environment resets, actor sampling, and optional Gaussian
            # noise across variants and model seeds
            np.random.seed(episode_seed)
            env.action_space.seed(episode_seed)
            actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(episode_seed))
            gaussian_rng = np.random.default_rng(episode_seed)
            observation, info = env.reset(
                seed=episode_seed,
                options=dict(task_id=task_id, render_goal=should_render),
            )
            initial_info = info.copy()
            goal_xyzs = _task_goal_xyzs(env)
            goal = info.get('goal')
            goal_frame = info.get('goal_rendered')
            language_embedding = None
            target_language_task_id = None
            condition_language_task_id = None
            task_info = getattr(env.unwrapped, 'cur_task_info', None)
            goal_ij = task_info.get('goal_ij') if isinstance(task_info, dict) else None
            if uses_language:
                if goal_ij is not None:
                    target_language_task_id = language_task_id_for_goal(
                        language_cache,
                        goal_ij,
                        fallback_task_id=task_id,
                    )
                elif goal_xyzs is not None and len(goal_xyzs) == 1:
                    target_language_task_id = language_task_id_for_goal_xyz(
                        language_cache,
                        goal_xyzs[0],
                        fallback_task_id=task_id,
                    )
                else:
                    target_language_task_id = int(task_id)
                (
                    language_embedding,
                    language_text,
                    condition_language_task_id,
                ) = evaluation_language_condition(
                    language_cache,
                    target_language_task_id,
                    language_eval_variant,
                    i,
                )
            if policy_conditioning == 'language':
                policy_condition = language_embedding
            else:
                policy_condition = goal
            if not uses_language:
                language_text = ''
            done = False
            step = 0
            render = []
            while not done:
                if policy_conditioning == 'goal_language':
                    action = actor_fn(
                        observations=observation,
                        goals=policy_condition,
                        language_embeddings=language_embedding,
                        temperature=eval_temperature,
                    )
                else:
                    action = actor_fn(
                        observations=observation,
                        goals=policy_condition,
                        temperature=eval_temperature,
                    )
                action = np.array(action)
                if not config.get('discrete'):
                    if eval_gaussian is not None:
                        action = gaussian_rng.normal(action, eval_gaussian)
                    action = np.clip(action, -1, 1)

                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                if should_render and (step % video_frame_skip == 0 or done):
                    frame = env.render().copy()
                    if goal_frame is not None:
                        render.append(np.concatenate([goal_frame, frame], axis=0))
                    else:
                        render.append(frame)

                transition = dict(
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=reward,
                    done=done,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                add_to(traj, transition)
                observation = next_observation
            if i < num_eval_episodes:
                flat_info = flatten(info)
                add_to(stats, flat_info)
                trajs.append(traj)
                summary = {
                    'task_id': int(task_id) if task_id is not None else -1,
                    'environment_task_id': int(task_id) if task_id is not None else -1,
                    'language_task_id': (
                        int(target_language_task_id)
                        if target_language_task_id is not None
                        else -1
                    ),
                    'language_condition_task_id': (
                        int(condition_language_task_id)
                        if condition_language_task_id is not None
                        else -1
                    ),
                    'language_variant': language_eval_variant if uses_language else 'goal',
                    'language_text': language_text,
                    'language_goal_row': (
                        int(language_cache['task_ij'][target_language_task_id - 1, 0])
                        if uses_language
                        and target_language_task_id is not None
                        and 'task_ij' in language_cache
                        else None
                    ),
                    'language_goal_column': (
                        int(language_cache['task_ij'][target_language_task_id - 1, 1])
                        if uses_language
                        and target_language_task_id is not None
                        and 'task_ij' in language_cache
                        else None
                    ),
                    'language_condition_row': (
                        int(language_cache['task_ij'][condition_language_task_id - 1, 0])
                        if uses_language
                        and condition_language_task_id is not None
                        and condition_language_task_id > 0
                        and 'task_ij' in language_cache
                        else None
                    ),
                    'language_condition_column': (
                        int(language_cache['task_ij'][condition_language_task_id - 1, 1])
                        if uses_language
                        and condition_language_task_id is not None
                        and condition_language_task_id > 0
                        and 'task_ij' in language_cache
                        else None
                    ),
                    'episode_index': i,
                    'evaluation_seed': episode_seed,
                    'success': float(bool(flat_info.get('success', False))),
                    'episode_length': int(len(traj['info'])),
                    'goal_i': (
                        int(np.asarray(goal_ij, dtype=np.int32).reshape(-1)[0])
                        if goal_ij is not None and np.asarray(goal_ij).size == 2
                        else None
                    ),
                    'goal_j': (
                        int(np.asarray(goal_ij, dtype=np.int32).reshape(-1)[1])
                        if goal_ij is not None and np.asarray(goal_ij).size == 2
                        else None
                    ),
                }
                if collect_diagnostics:
                    summary.update(
                        summarize_episode(
                            initial_info,
                            traj,
                            goal_xyzs=goal_xyzs,
                        )
                    )
                episode_summaries.append(summary)
            else:
                renders.append(np.array(render))
    finally:
        # Evaluation must not perturb the training dataset/augmentation stream.
        np.random.set_state(numpy_state)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if collect_diagnostics and episode_summaries:
        for metric in EVAL_DIAGNOSTIC_METRICS:
            values = np.asarray([summary[metric] for summary in episode_summaries], dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values):
                stats[metric] = float(np.mean(values))
        modes, counts = np.unique([summary['failure_mode'] for summary in episode_summaries], return_counts=True)
        mode_counts = dict(zip(modes, counts))
        for mode in FAILURE_MODES:
            stats[f'failure/{mode}_fraction'] = float(mode_counts.get(mode, 0) / len(episode_summaries))

    return stats, trajs, renders, episode_summaries
