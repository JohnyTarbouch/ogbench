from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from utils.language import evaluation_language_embedding, evaluation_language_text, load_language_cache


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
)

CONTACT_SIGNAL_THRESHOLD = 0.1
LIKELY_OBJECT_CONTACT_DISTANCE = 0.08

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


def _task_goal_xyz(env):
    unwrapped = env.unwrapped
    task_info = getattr(unwrapped, 'cur_task_info', None)
    if not isinstance(task_info, dict) or 'goal_xyzs' not in task_info:
        return None
    goal_xyzs = np.asarray(task_info['goal_xyzs'], dtype=np.float64)
    if goal_xyzs.ndim != 2 or goal_xyzs.shape[1] < 3 or len(goal_xyzs) == 0:
        return None
    target_block = int(getattr(unwrapped, '_target_block', 0))
    target_block = min(max(target_block, 0), len(goal_xyzs) - 1)
    return goal_xyzs[target_block, :3].copy()


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


def summarize_episode(initial_info, traj, goal_xyz=None):
    """Summarize the signals needed to explain Cube success and failure."""
    step_infos = list(traj.get('info', []))
    infos = [initial_info, *step_infos]
    actions = np.asarray(traj.get('action', []), dtype=np.float64)
    if actions.ndim == 1 and actions.size:
        actions = actions[:, None]

    block_positions = [
        value[:3]
        for info in infos
        if (value := _info_vector(info, 'privileged/block_0_pos')) is not None and value.size >= 3
    ]
    block_positions = np.asarray(block_positions, dtype=np.float64)
    has_geometry = goal_xyz is not None and len(block_positions) > 0
    if has_geometry:
        goal_xyz = np.asarray(goal_xyz, dtype=np.float64)[:3]
        goal_distances = np.linalg.norm(block_positions - goal_xyz, axis=-1)
        block_displacements = np.linalg.norm(block_positions - block_positions[0], axis=-1)
        movement = block_positions[-1] - block_positions[0]
        goal_direction = goal_xyz - block_positions[0]
        denominator = np.linalg.norm(movement) * np.linalg.norm(goal_direction)
        movement_alignment = float(np.dot(movement, goal_direction) / denominator) if denominator > 1e-8 else np.nan
    else:
        goal_distances = block_displacements = np.empty(0)
        movement_alignment = np.nan

    effector_block_distances = []
    contacts = []
    likely_object_contacts = []
    openings = []
    for info in infos:
        block = _info_vector(info, 'privileged/block_0_pos')
        effector = _info_vector(info, 'proprio/effector_pos')
        effector_block_distance = None
        if block is not None and effector is not None and block.size >= 3 and effector.size >= 3:
            effector_block_distance = float(np.linalg.norm(block[:3] - effector[:3]))
            effector_block_distances.append(effector_block_distance)
        contact = _info_vector(info, 'proprio/gripper_contact')
        if contact is not None:
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
        collect_diagnostics: Whether to compute compact per-episode behavioral summaries.

    Returns:
        A tuple containing statistics, trajectories, rendered videos, and episode summaries.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    policy_conditioning = config.get('policy_conditioning')
    language_cache = None
    if policy_conditioning == 'language':
        language_cache = load_language_cache(
            config['language_embedding_path'],
            int(config['num_language_tasks']),
            int(config['language_embedding_dim']),
        )

    renders = []
    episode_summaries = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        initial_info = info.copy()
        goal_xyz = _task_goal_xyz(env)
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        if policy_conditioning == 'language':
            policy_condition = evaluation_language_embedding(
                language_cache,
                task_id,
                language_eval_variant,
                i,
            )
            language_text = evaluation_language_text(language_cache, task_id, language_eval_variant, i)
        else:
            policy_condition = goal
            language_text = ''
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=policy_condition, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
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
            add_to(stats, flatten(info))
            trajs.append(traj)
            if collect_diagnostics:
                summary = summarize_episode(initial_info, traj, goal_xyz=goal_xyz)
                summary.update(
                    {
                        'task_id': int(task_id) if task_id is not None else -1,
                        'language_variant': language_eval_variant if policy_conditioning == 'language' else 'goal',
                        'language_text': language_text,
                        'episode_index': i,
                    }
                )
                episode_summaries.append(summary)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if episode_summaries:
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
