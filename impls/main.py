import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from importlib import metadata

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


def _json_default(value):
    # Convert to JSON
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    return str(value)


def _write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)


def _array_summary(array):
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
    }


def _dataset_summary(name, dataset):
    summary = {
        'name': name,
        'size': int(dataset.size),
        'keys': sorted(dataset.keys()),
        'arrays': {key: _array_summary(value) for key, value in dataset.items()},
    }
    if 'valids' in dataset:
        summary['num_valid'] = int(np.sum(dataset['valids'] > 0))
    if 'terminals' in dataset:
        terminal_idxs = np.flatnonzero(dataset['terminals'] > 0)
        summary['num_trajectories'] = int(len(terminal_idxs))
        if len(terminal_idxs) > 0:
            initial_idxs = np.concatenate([[0], terminal_idxs[:-1] + 1])
            lengths = terminal_idxs - initial_idxs + 1
            summary['trajectory_length'] = {
                'min': int(np.min(lengths)),
                'mean': float(np.mean(lengths)),
                'max': int(np.max(lengths)),
            }
    return summary


def _goal_dataset_summary(name, goal_dataset):
    summary = {
        'name': name,
        'size': int(goal_dataset.size),
        'num_trajectories': int(len(goal_dataset.terminal_locs)),
    }
    if len(goal_dataset.terminal_locs) > 0:
        lengths = goal_dataset.terminal_locs - goal_dataset.initial_locs + 1
        summary['trajectory_length'] = {
            'min': int(np.min(lengths)),
            'mean': float(np.mean(lengths)),
            'max': int(np.max(lengths)),
        }
    return summary


def _env_summary(env):
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else getattr(env, 'task_infos', [])
    return {
        'env_class': f'{type(env).__module__}.{type(env).__name__}',
        'unwrapped_env_class': f'{type(env.unwrapped).__module__}.{type(env.unwrapped).__name__}',
        'observation_space': str(env.observation_space),
        'action_space': str(env.action_space),
        'num_tasks': len(task_infos),
        'tasks': [
            {
                'task_id': idx + 1,
                'task_name': task_info.get('task_name', f'task{idx + 1}'),
            }
            for idx, task_info in enumerate(task_infos)
        ],
    }


def _runtime_summary():
    package_names = [
        'ogbench',
        'jax',
        'jaxlib',
        'flax',
        'optax',
        'distrax',
        'mujoco',
        'dm_control',
        'gymnasium',
        'wandb',
        'numpy',
    ]
    packages = {}
    for package_name in package_names:
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None

    try:
        jax_devices = [str(device) for device in jax.devices()]
        jax_backend_error = None
    except Exception as exc: #login nodes with CUDA JAX
        jax_devices = []
        jax_backend_error = repr(exc)

    return {
        'python': sys.version,
        'platform': platform.platform(),
        'hostname': platform.node(),
        'cwd': os.getcwd(),
        'packages': packages,
        'jax_devices': jax_devices,
        'jax_backend_error': jax_backend_error,
        'env_vars': {
            key: os.environ.get(key)
            for key in [
                'CUDA_VISIBLE_DEVICES',
                'JAX_PLATFORMS',
                'XLA_PYTHON_CLIENT_PREALLOCATE',
                'MUJOCO_GL',
                'WANDB_MODE',
                'WANDB_DIR',
                'WANDB_CACHE_DIR',
                'WANDB_CONFIG_DIR',
                'SLURM_JOB_ID',
                'SLURM_JOB_NAME',
                'SLURM_PROCID',
                'SLURM_NODELIST',
            ]
        },
    }


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    _write_json(os.path.join(FLAGS.save_dir, 'flags.json'), flag_dict)
    _write_json(os.path.join(FLAGS.save_dir, 'runtime.json'), _runtime_summary())
    _write_json(
        os.path.join(FLAGS.save_dir, 'wandb.json'),
        {
            'mode': getattr(wandb.run.settings, 'mode', None),
            'id': wandb.run.id,
            'name': wandb.run.name,
            'project': wandb.run.project,
            'group': wandb.run.group,
            'dir': wandb.run.dir,
            'path': wandb.run.path,
        },
    )

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    _write_json(
        os.path.join(FLAGS.save_dir, 'dataset_raw.json'),
        {
            'env': _env_summary(env),
            'train_dataset': _dataset_summary('train', train_dataset),
            'val_dataset': _dataset_summary('val', val_dataset),
        },
    )

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)
    _write_json(
        os.path.join(FLAGS.save_dir, 'dataset_goal_conditioned.json'),
        {
            'dataset_class': config['dataset_class'],
            'train_dataset': _goal_dataset_summary('train', train_dataset),
            'val_dataset': _goal_dataset_summary('val', val_dataset) if val_dataset is not None else None,
            'goal_sampling': {
                key: config[key]
                for key in [
                    'value_p_curgoal',
                    'value_p_trajgoal',
                    'value_p_randomgoal',
                    'value_geom_sample',
                    'actor_p_curgoal',
                    'actor_p_trajgoal',
                    'actor_p_randomgoal',
                    'actor_geom_sample',
                    'gc_negative',
                    'p_aug',
                    'frame_stack',
                ]
            },
        },
    )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
