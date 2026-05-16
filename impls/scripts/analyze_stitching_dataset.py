"""
Analyze where an OGBench dataset requires stitching.

Logs that help interpret goal-conditioned baselines and temporal augmentation runs:

- whether evaluation starts/goals are covered by the same dataset trajectories?
- how the default GCBC same-trajectory goal sampling differs from random goalsß
- how often a simple XY-overlap rule can find cross-trajectory stitch candidates?
- where those candidates are only XY-compatible?
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/bigwork/nhwptarj/matplotlib_cache')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import ogbench


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True, default=json_default)


def summary_stats(values):
    values = np.asarray(values)
    if len(values) == 0:
        return dict(count=0)
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'p05': float(np.percentile(values, 5)),
        'p50': float(np.percentile(values, 50)),
        'p95': float(np.percentile(values, 95)),
        'max': float(np.max(values)),
    }


def trajectory_info(terminals):
    terminal_locs = np.flatnonzero(terminals > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    traj_ids = np.searchsorted(terminal_locs, np.arange(len(terminals)))
    lengths = terminal_locs - initial_locs + 1
    return initial_locs, terminal_locs, traj_ids, lengths


def sample_traj_goals(rng, idxs, terminal_locs):
    final_state_idxs = terminal_locs[np.searchsorted(terminal_locs, idxs)]
    distances = rng.random(len(idxs))
    return np.round((np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))).astype(
        int
    )


def task_info(env):
    infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    return [
        {
            'task_id': idx + 1,
            'task_name': info.get('task_name', f'task{idx + 1}'),
            'init_ij': tuple(info.get('init_ij', ())),
            'goal_ij': tuple(info.get('goal_ij', ())),
            'init_xy': np.asarray(info['init_xy'], dtype=np.float64),
            'goal_xy': np.asarray(info['goal_xy'], dtype=np.float64),
        }
        for idx, info in enumerate(infos)
    ]


def analyze_eval_task_coverage(tasks, xy, traj_ids, tree, valid_idxs, radii):
    rows = []
    valid_traj_ids = traj_ids[valid_idxs]
    for task in tasks:
        init_xy = task['init_xy']
        goal_xy = task['goal_xy']
        init_nn_dist, init_nn_pos = tree.query(init_xy, k=1)
        goal_nn_dist, goal_nn_pos = tree.query(goal_xy, k=1)
        init_nn_idx = int(valid_idxs[int(init_nn_pos)])
        goal_nn_idx = int(valid_idxs[int(goal_nn_pos)])
        for radius in radii:
            init_positions = tree.query_ball_point(init_xy, radius)
            goal_positions = tree.query_ball_point(goal_xy, radius)
            init_trajs = set(valid_traj_ids[init_positions].tolist())
            goal_trajs = set(valid_traj_ids[goal_positions].tolist())
            shared_trajs = init_trajs.intersection(goal_trajs)
            rows.append(
                {
                    'task_id': task['task_id'],
                    'task_name': task['task_name'],
                    'radius': float(radius),
                    'init_xy': init_xy,
                    'goal_xy': goal_xy,
                    'init_nearest_dataset_dist': float(init_nn_dist),
                    'goal_nearest_dataset_dist': float(goal_nn_dist),
                    'init_nearest_idx': init_nn_idx,
                    'goal_nearest_idx': goal_nn_idx,
                    'init_near_state_count': int(len(init_positions)),
                    'goal_near_state_count': int(len(goal_positions)),
                    'init_near_traj_count': int(len(init_trajs)),
                    'goal_near_traj_count': int(len(goal_trajs)),
                    'shared_near_traj_count': int(len(shared_trajs)),
                    'single_trajectory_covered': bool(len(shared_trajs) > 0),
                    'xy_task_distance': float(np.linalg.norm(goal_xy - init_xy)),
                }
            )
    return rows


def analyze_goal_sampling(rng, obs, xy, traj_ids, terminal_locs, valid_idxs, sample_size):
    idxs = rng.choice(valid_idxs, size=min(sample_size, len(valid_idxs)), replace=False)
    traj_goal_idxs = sample_traj_goals(rng, idxs, terminal_locs)
    random_goal_idxs = rng.choice(valid_idxs, size=len(idxs), replace=True)

    return {
        'sample_size': int(len(idxs)),
        'default_gcbc_actor_goal_sampling': {
            'description': 'actor_p_trajgoal=1.0: future goal from the same trajectory',
            'same_trajectory_fraction': float(np.mean(traj_ids[idxs] == traj_ids[traj_goal_idxs])),
            'temporal_offset': summary_stats(traj_goal_idxs - idxs),
            'xy_distance': summary_stats(np.linalg.norm(xy[traj_goal_idxs] - xy[idxs], axis=1)),
            'full_observation_distance': summary_stats(np.linalg.norm(obs[traj_goal_idxs] - obs[idxs], axis=1)),
        },
        'random_actor_goal_sampling': {
            'description': 'actor_p_randomgoal>0: goals may come from different trajectories',
            'same_trajectory_fraction': float(np.mean(traj_ids[idxs] == traj_ids[random_goal_idxs])),
            'different_trajectory_fraction': float(np.mean(traj_ids[idxs] != traj_ids[random_goal_idxs])),
            'xy_distance': summary_stats(np.linalg.norm(xy[random_goal_idxs] - xy[idxs], axis=1)),
            'full_observation_distance': summary_stats(np.linalg.norm(obs[random_goal_idxs] - obs[idxs], axis=1)),
        },
    }


def analyze_temporal_stitch_candidates(
    rng,
    obs,
    xy,
    traj_ids,
    terminal_locs,
    tree,
    valid_idxs,
    sample_size,
    radii,
    nearest_k,
):
    idxs = rng.choice(valid_idxs, size=min(sample_size, len(valid_idxs)), replace=False)
    original_goal_idxs = sample_traj_goals(rng, idxs, terminal_locs)
    query_k = min(nearest_k, len(valid_idxs))
    dists, positions = tree.query(xy[original_goal_idxs], k=query_k)
    if query_k == 1:
        dists = dists[:, None]
        positions = positions[:, None]

    valid_traj_ids = traj_ids[valid_idxs]
    rows = []
    examples = []
    for radius in radii:
        accepted = []
        accepted_xy_dist = []
        accepted_full_dist = []
        accepted_future_room = []
        accepted_same_start_traj = []
        for row_idx, goal_idx in enumerate(original_goal_idxs):
            start_traj = traj_ids[idxs[row_idx]]
            goal_traj = traj_ids[goal_idx]
            match = None
            for dist, pos in zip(dists[row_idx], positions[row_idx]):
                if dist > radius:
                    continue
                waypoint_idx = int(valid_idxs[int(pos)])
                waypoint_traj = int(valid_traj_ids[int(pos)])
                if waypoint_traj == goal_traj:
                    continue
                if waypoint_idx >= terminal_locs[waypoint_traj]:
                    continue
                match = (waypoint_idx, waypoint_traj, float(dist))
                break
            if match is None:
                accepted.append(False)
                continue
            waypoint_idx, waypoint_traj, xy_dist = match
            accepted.append(True)
            accepted_xy_dist.append(xy_dist)
            accepted_full_dist.append(float(np.linalg.norm(obs[goal_idx] - obs[waypoint_idx])))
            accepted_future_room.append(int(terminal_locs[waypoint_traj] - waypoint_idx))
            accepted_same_start_traj.append(bool(waypoint_traj == start_traj))
            if len(examples) < 20 and radius == radii[0]:
                examples.append(
                    {
                        'state_idx': int(idxs[row_idx]),
                        'state_traj': int(start_traj),
                        'original_goal_idx': int(goal_idx),
                        'original_goal_traj': int(goal_traj),
                        'waypoint_idx': int(waypoint_idx),
                        'waypoint_traj': int(waypoint_traj),
                        'xy_distance_goal_to_waypoint': xy_dist,
                        'full_observation_distance_goal_to_waypoint': float(np.linalg.norm(obs[goal_idx] - obs[waypoint_idx])),
                        'waypoint_future_steps_available': int(terminal_locs[waypoint_traj] - waypoint_idx),
                        'state_xy': xy[idxs[row_idx]],
                        'original_goal_xy': xy[goal_idx],
                        'waypoint_xy': xy[waypoint_idx],
                    }
                )

        accepted = np.asarray(accepted, dtype=bool)
        rows.append(
            {
                'radius': float(radius),
                'sample_size': int(len(idxs)),
                'accepted_count': int(np.sum(accepted)),
                'accepted_fraction': float(np.mean(accepted)),
                'xy_goal_to_waypoint_distance': summary_stats(accepted_xy_dist),
                'full_goal_to_waypoint_distance': summary_stats(accepted_full_dist),
                'waypoint_future_steps_available': summary_stats(accepted_future_room),
                'accepted_same_as_start_traj_fraction': float(np.mean(accepted_same_start_traj))
                if accepted_same_start_traj
                else 0.0,
            }
        )
    return rows, examples


def save_task_coverage_csv(path, rows):
    fieldnames = [
        'task_id',
        'task_name',
        'radius',
        'init_nearest_dataset_dist',
        'goal_nearest_dataset_dist',
        'init_near_state_count',
        'goal_near_state_count',
        'init_near_traj_count',
        'goal_near_traj_count',
        'shared_near_traj_count',
        'single_trajectory_covered',
        'xy_task_distance',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def save_xy_plot(path, tasks, xy, rng, max_points):
    sample_size = min(max_points, len(xy))
    sample_idxs = rng.choice(np.arange(len(xy)), size=sample_size, replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xy[sample_idxs, 0], xy[sample_idxs, 1], s=1, alpha=0.12, label='dataset states')
    for task in tasks:
        init_xy = task['init_xy']
        goal_xy = task['goal_xy']
        ax.scatter(init_xy[0], init_xy[1], marker='o', s=70, label=f"{task['task_name']} init")
        ax.scatter(goal_xy[0], goal_xy[1], marker='*', s=110, label=f"{task['task_name']} goal")
        ax.arrow(
            init_xy[0],
            init_xy[1],
            goal_xy[0] - init_xy[0],
            goal_xy[1] - init_xy[1],
            length_includes_head=True,
            head_width=0.4,
            alpha=0.7,
        )
    ax.set_title('OGBench stitch dataset XY coverage and evaluation tasks')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='antmaze-medium-stitch-v0')
    parser.add_argument('--dataset_dir', default='/bigwork/nhwptarj/ogbench_data')
    parser.add_argument('--output_dir', default='/bigwork/nhwptarj/ogbench_stitching_analysis')
    parser.add_argument('--sample_size', type=int, default=20000)
    parser.add_argument('--nearest_k', type=int, default=64)
    parser.add_argument('--radii', default='0.25,0.5,1.0,2.0')
    parser.add_argument('--plot_points', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    radii = [float(x) for x in args.radii.split(',') if x]
    output_dir = Path(args.output_dir) / args.env_name
    output_dir.mkdir(parents=True, exist_ok=True)

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        args.env_name,
        dataset_dir=args.dataset_dir,
        compact_dataset=True,
    )
    obs = train_dataset['observations']
    xy = obs[:, :2]
    actions = train_dataset['actions']
    valids = train_dataset['valids'] > 0
    valid_idxs = np.flatnonzero(valids)
    initial_locs, terminal_locs, traj_ids, lengths = trajectory_info(train_dataset['terminals'])
    tree = cKDTree(xy[valid_idxs])
    tasks = task_info(env)

    task_coverage_rows = analyze_eval_task_coverage(tasks, xy, traj_ids, tree, valid_idxs, radii)
    goal_sampling = analyze_goal_sampling(rng, obs, xy, traj_ids, terminal_locs, valid_idxs, args.sample_size)
    stitch_candidates, stitch_examples = analyze_temporal_stitch_candidates(
        rng,
        obs,
        xy,
        traj_ids,
        terminal_locs,
        tree,
        valid_idxs,
        args.sample_size,
        radii,
        args.nearest_k,
    )

    analysis = {
        'env_name': args.env_name,
        'interpretation_notes': {
            'default_gcbc_limitation': (
                'GCBC defaults to actor_p_trajgoal=1.0, so every actor goal used for BC is a future state from the '
                'same trajectory. This does not train the policy directly on cross-trajectory state-goal pairs.'
            ),
            'stitch_dataset_relevance': (
                'OGBench notes that allowing random actor goals is especially important for datasets that require '
                'stitching. Temporal augmentation should be compared against this baseline and against random-goal controls.'
            ),
            'xy_overlap_caution': (
                'For AntMaze, observation[:2] is agent XY. XY overlap is a useful first oracle for navigation, but full '
                'state compatibility also includes ant posture and velocity, so accepted XY matches are not automatically '
                'full-state-compatible.'
            ),
        },
        'dataset': {
            'num_transitions': int(len(obs)),
            'num_valid_transitions': int(len(valid_idxs)),
            'num_trajectories': int(len(terminal_locs)),
            'observation_shape': list(obs.shape),
            'action_shape': list(actions.shape),
            'trajectory_length': summary_stats(lengths),
            'xy_bounds': {
                'min': np.min(xy[valid_idxs], axis=0),
                'max': np.max(xy[valid_idxs], axis=0),
            },
        },
        'eval_tasks': [
            {
                'task_id': task['task_id'],
                'task_name': task['task_name'],
                'init_ij': task['init_ij'],
                'goal_ij': task['goal_ij'],
                'init_xy': task['init_xy'],
                'goal_xy': task['goal_xy'],
                'xy_distance': float(np.linalg.norm(task['goal_xy'] - task['init_xy'])),
            }
            for task in tasks
        ],
        'eval_task_dataset_coverage': task_coverage_rows,
        'goal_sampling_diagnostics': goal_sampling,
        'temporal_stitch_candidate_diagnostics': stitch_candidates,
        'temporal_stitch_candidate_examples': stitch_examples,
    }

    write_json(output_dir / 'stitching_analysis.json', analysis)
    save_task_coverage_csv(output_dir / 'eval_task_dataset_coverage.csv', task_coverage_rows)
    write_json(output_dir / 'temporal_stitch_candidates.json', stitch_candidates)
    save_xy_plot(output_dir / 'xy_coverage_tasks.png', tasks, xy[valid_idxs], rng, args.plot_points)

    print(f'Wrote stitching analysis to: {output_dir}')
    print('Key files:')
    print(f'  {output_dir / "stitching_analysis.json"}')
    print(f'  {output_dir / "eval_task_dataset_coverage.csv"}')
    print(f'  {output_dir / "temporal_stitch_candidates.json"}')
    print(f'  {output_dir / "xy_coverage_tasks.png"}')


if __name__ == '__main__':
    main()
