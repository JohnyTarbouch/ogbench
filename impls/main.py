import csv
import glob
import hashlib
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
from utils.datasets import (
    AtomicGCDataset,
    AtomicLanguageDataset,
    Dataset,
    EndpointLanguageDataset,
    FutureGoalImageLanguageDataset,
    FutureGoalLanguageDataset,
    GCDataset,
    HGCDataset,
)
from utils.env_utils import make_env_and_datasets
from utils.evaluation import EVAL_DIAGNOSTIC_METRICS, evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.stitch_datasets import TemporalStitchGCDataset, VisualFeatureTemporalStitchGCDataset
from utils.stitch_datasets_advanced import AdvancedTemporalStitchGCDataset
from utils.visual_feature_knn_stitch_datasets import VisualFeatureLocalKnnTemporalStitchGCDataset

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_bool('eval_only', False, 'Evaluate a restored checkpoint without updating or saving it.')

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
flags.DEFINE_bool('eval_at_start', True, 'Whether to evaluate the randomly initialized policy at step 1.')
flags.DEFINE_bool(
    'eval_goal_noise',
    None,
    'Whether locomaze evaluation goals use within-cell noise; None preserves the environment default.',
)
flags.DEFINE_bool('eval_diagnostics', False, 'Whether to save compact rollout diagnostics.')
flags.DEFINE_integer('eval_seed', 0, 'Base seed for paired, reproducible evaluation episodes.')

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


def _file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_restore_path(pattern, epoch, seed):
    formatted = pattern.replace('%SEED3%', f'{seed:03d}').replace('%SEED%', str(seed))
    candidates = glob.glob(formatted)
    if len(candidates) != 1:
        raise ValueError(f'Restore pattern matched {len(candidates)} directories: {formatted!r}')
    directory = os.path.abspath(candidates[0])
    checkpoint = os.path.join(directory, f'params_{epoch}.pkl')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'Restore checkpoint not found: {checkpoint}')
    return formatted, checkpoint


def _append_csv_rows(path, rows, step):
    if not rows:
        return
    fieldnames = ['step'] + list(rows[0].keys())
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({'step': step, **row})


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
        endpoint_idxs = np.flatnonzero(dataset['valids'] == 0)
        if len(endpoint_idxs):
            initial_idxs = np.concatenate([[0], endpoint_idxs[:-1] + 1])
            lengths = endpoint_idxs - initial_idxs + 1
            summary['num_trajectories'] = int(len(endpoint_idxs))
            summary['trajectory_length'] = {
                'min': int(np.min(lengths)),
                'mean': float(np.mean(lengths)),
                'max': int(np.max(lengths)),
            }
    elif 'terminals' in dataset:
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
    if hasattr(goal_dataset, 'manifest_summary'):
        # print(f"Language data summary for {name}: {goal_dataset.manifest_summary}")
        summary['atomic_manifest'] = goal_dataset.manifest_summary
    if hasattr(goal_dataset, 'future_label_summary'):
        # print(f"Future goal language label summary for {name}: {goal_dataset.future_label_summary}")
        summary['future_goal_language_labels'] = goal_dataset.future_label_summary
    if hasattr(goal_dataset, 'endpoint_manifest_summary'):
        summary['endpoint_manifest'] = goal_dataset.endpoint_manifest_summary
        summary['num_trajectories'] = goal_dataset.endpoint_manifest_summary['num_retained_episodes']
    if hasattr(goal_dataset, 'language_summary'):
        summary['language'] = goal_dataset.language_summary
    if hasattr(goal_dataset, 'endpoint_goal_indices'):
        starts = goal_dataset.raw_episode_starts[goal_dataset.endpoint_episode_ids]
        lengths = goal_dataset.endpoint_goal_indices - starts
        summary['trajectory_length'] = {
            'min': int(np.min(lengths)),
            'mean': float(np.mean(lengths)),
            'max': int(np.max(lengths)),
        }
    elif hasattr(goal_dataset, 'raw_goal_indices'):
        lengths = goal_dataset.raw_goal_indices - goal_dataset.raw_episode_starts + 1
        summary['trajectory_length'] = {
            'min': int(np.min(lengths)),
            'mean': float(np.mean(lengths)),
            'max': int(np.max(lengths)),
        }
    elif 'valids' in goal_dataset.dataset:
        endpoint_idxs = np.flatnonzero(goal_dataset.dataset['valids'] == 0)
        initial_idxs = np.concatenate([[0], endpoint_idxs[:-1] + 1])
        lengths = endpoint_idxs - initial_idxs + 1
        summary['num_trajectories'] = int(len(endpoint_idxs))
        summary['trajectory_length'] = {
            'min': int(np.min(lengths)),
            'mean': float(np.mean(lengths)),
            'max': int(np.max(lengths)),
        }
    elif len(goal_dataset.terminal_locs) > 0:
        lengths = goal_dataset.terminal_locs - goal_dataset.initial_locs + 1
        summary['trajectory_length'] = {
            'min': int(np.min(lengths)),
            'mean': float(np.mean(lengths)),
            'max': int(np.max(lengths)),
        }
    return summary


def _goal_sampling_summary(config):
    base_keys = [
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
        'policy_conditioning',
        'language_dataset_mode',
        'endpoint_dataset_mode',
        'endpoint_sampling',
        'endpoint_train_manifest_path',
        'endpoint_val_manifest_path',
        'num_language_tasks',
        'atomic_train_manifest_path',
        'atomic_val_manifest_path',
        'atomic_goal_stack_mode',
        'atomic_require_source_fingerprint',
        'future_language_train_labels_path',
        'future_language_val_labels_path',
        'language_embedding_path',
        'language_embedding_model',
        'language_embedding_sha256',
        'language_embedding_dim',
        'language_min_train_retrieval_top1',
        'language_min_heldout_retrieval_top1',
        'language_train_variant',
        'language_train_control',
        'language_eval_variants',
        'language_final_eval_variants',
    ]
    stitch_keys = sorted(
        key for key in config.keys() 
        if key.startswith('stitch_')
    )
    keys = []
    seen = set()
    for key in base_keys + stitch_keys:
        if key not in seen and key in config:
            keys.append(key)
            seen.add(key)
    return {key: config[key] for key in keys}


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
    config = FLAGS.agent
    config['run_seed'] = FLAGS.seed

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
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=config['frame_stack'],
        add_noise_to_goal=FLAGS.eval_goal_noise,
    )
    _write_json(
        os.path.join(FLAGS.save_dir, 'dataset_raw.json'),
        {
            'env': _env_summary(env),
            'train_dataset': _dataset_summary('train', train_dataset),
            'val_dataset': _dataset_summary('val', val_dataset),
        },
    )

    dataset_class = {
        'AtomicGCDataset': AtomicGCDataset,
        'AtomicLanguageDataset': AtomicLanguageDataset,
        'EndpointLanguageDataset': EndpointLanguageDataset,
        'FutureGoalImageLanguageDataset': FutureGoalImageLanguageDataset,
        'FutureGoalLanguageDataset': FutureGoalLanguageDataset,
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'TemporalStitchGCDataset': TemporalStitchGCDataset,
        'VisualFeatureTemporalStitchGCDataset': VisualFeatureTemporalStitchGCDataset,
        'VisualFeatureLocalKnnTemporalStitchGCDataset': VisualFeatureLocalKnnTemporalStitchGCDataset,
        'AdvancedTemporalStitchGCDataset': AdvancedTemporalStitchGCDataset,
    }[config['dataset_class']]
    # set up training dataset with arguments
    train_dataset_kwargs = {}
    if dataset_class in {AtomicGCDataset, AtomicLanguageDataset}:
        train_dataset_kwargs['manifest_path'] = config['atomic_train_manifest_path']
        train_dataset_kwargs['source_dataset_name'] = FLAGS.env_name
        train_dataset_kwargs['source_split'] = 'train'
        dataset_dir = os.environ.get('OGBENCH_DATASET_DIR') or os.environ.get('OGBENCH_DATA_DIR')
        if dataset_dir:
            train_dataset_kwargs['source_path'] = os.path.join(dataset_dir, f'{FLAGS.env_name}.npz')
    elif dataset_class is EndpointLanguageDataset:
        train_dataset_kwargs['manifest_path'] = config['endpoint_train_manifest_path']
        dataset_dir = os.environ.get('OGBENCH_DATASET_DIR') or os.environ.get('OGBENCH_DATA_DIR')
        if dataset_dir:
            train_dataset_kwargs['source_path'] = os.path.join(dataset_dir, f'{FLAGS.env_name}.npz')
    elif dataset_class in {FutureGoalLanguageDataset, FutureGoalImageLanguageDataset}:
        train_dataset_kwargs['labels_path'] = config['future_language_train_labels_path']
        dataset_dir = os.environ.get('OGBENCH_DATASET_DIR') or os.environ.get('OGBENCH_DATA_DIR')
        if dataset_dir:
            train_dataset_kwargs['source_path'] = os.path.join(dataset_dir, f'{FLAGS.env_name}.npz')
    train_dataset = dataset_class(Dataset.create(**train_dataset), config, **train_dataset_kwargs)
    if val_dataset is not None:
        stitch_dataset_classes = {
            'TemporalStitchGCDataset',
            'VisualFeatureTemporalStitchGCDataset',
            'VisualFeatureLocalKnnTemporalStitchGCDataset',
            'AdvancedTemporalStitchGCDataset',
        }
        val_dataset_class = GCDataset if config['dataset_class'] in stitch_dataset_classes else dataset_class
        # set up validation dataset with arguments
        val_dataset_kwargs = {}
        if val_dataset_class in {AtomicGCDataset, AtomicLanguageDataset}:
            val_dataset_kwargs['manifest_path'] = config['atomic_val_manifest_path']
            val_dataset_kwargs['source_dataset_name'] = FLAGS.env_name
            val_dataset_kwargs['source_split'] = 'val'
            dataset_dir = os.environ.get('OGBENCH_DATASET_DIR') or os.environ.get('OGBENCH_DATA_DIR')
            if dataset_dir:
                val_dataset_kwargs['source_path'] = os.path.join(
                    dataset_dir, f'{FLAGS.env_name}-val.npz'
                )
        elif val_dataset_class is EndpointLanguageDataset:
            val_dataset_kwargs['manifest_path'] = config['endpoint_val_manifest_path']
            dataset_dir = os.environ.get('OGBENCH_DATASET_DIR') or os.environ.get('OGBENCH_DATA_DIR')
            if dataset_dir:
                val_dataset_kwargs['source_path'] = os.path.join(
                    dataset_dir, f'{FLAGS.env_name}-val.npz'
                )
        elif val_dataset_class in {FutureGoalLanguageDataset, FutureGoalImageLanguageDataset}:
            val_dataset_kwargs['labels_path'] = config['future_language_val_labels_path']
            dataset_dir = os.environ.get('OGBENCH_DATASET_DIR') or os.environ.get('OGBENCH_DATA_DIR')
            if dataset_dir:
                val_dataset_kwargs['source_path'] = os.path.join(
                    dataset_dir, f'{FLAGS.env_name}-val.npz'
                )
        val_dataset = val_dataset_class(Dataset.create(**val_dataset), config, **val_dataset_kwargs)
    _write_json(
        os.path.join(FLAGS.save_dir, 'dataset_goal_conditioned.json'),
        {
            'dataset_class': config['dataset_class'],
            'train_dataset_class': train_dataset.__class__.__name__,
            'val_dataset_class': val_dataset.__class__.__name__ if val_dataset is not None else None,
            'train_dataset': _goal_dataset_summary('train', train_dataset),
            'val_dataset': _goal_dataset_summary('val', val_dataset) if val_dataset is not None else None,
            'goal_sampling': _goal_sampling_summary(config),
            'stitching': getattr(train_dataset, 'stitch_summary', None),
        },
    )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if hasattr(train_dataset, 'get_and_reset_diagnostics'):
        train_dataset.get_and_reset_diagnostics()
    if hasattr(train_dataset, 'get_and_reset_debug_records'):
        train_dataset.get_and_reset_debug_records()
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
    if 'representation_type' in agent.config:
        _write_json(
            os.path.join(FLAGS.save_dir, 'representation_transfer.json'),
            {
                key: agent.config.get(key)
                for key in [
                    'representation_type',
                    'representation_source',
                    'encoder_transfer_mode',
                    'source_checkpoint',
                    'resolved_source_checkpoint',
                    'source_module',
                ]
                if key in agent.config
            },
        )

    # Restore agent.
    if FLAGS.eval_only and (FLAGS.restore_path is None or FLAGS.restore_epoch is None):
        raise ValueError('--eval_only requires both --restore_path and --restore_epoch.')
    if FLAGS.restore_path is not None:
        formatted_restore_path, checkpoint_path = _resolve_restore_path(
            FLAGS.restore_path, FLAGS.restore_epoch, FLAGS.seed
        )
        if os.path.dirname(checkpoint_path) == os.path.abspath(FLAGS.save_dir):
            raise ValueError('Evaluation output directory may not equal the source checkpoint directory.')
        agent = restore_agent(agent, formatted_restore_path, FLAGS.restore_epoch)
        _write_json(
            os.path.join(FLAGS.save_dir, 'checkpoint_restore.json'),
            {
                'requested_pattern': FLAGS.restore_path,
                'formatted_pattern': formatted_restore_path,
                'checkpoint_path': checkpoint_path,
                'checkpoint_sha256': _file_sha256(checkpoint_path),
                'restore_epoch': FLAGS.restore_epoch,
                'eval_only': FLAGS.eval_only,
            },
        )

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    eval_episode_path = os.path.join(FLAGS.save_dir, 'eval_episodes.csv')
    stitch_debug_path = os.path.join(FLAGS.save_dir, 'stitch_debug.csv')
    first_time = time.time()
    last_time = time.time()
    iteration_steps = [FLAGS.restore_epoch] if FLAGS.eval_only else range(1, FLAGS.train_steps + 1)
    for i in tqdm.tqdm(iteration_steps, smoothing=0.1, dynamic_ncols=True):
        if not FLAGS.eval_only:
            # Update agent.
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = agent.update(batch)

        # Log metrics.
        if not FLAGS.eval_only and i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'], evaluation=True)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            if hasattr(train_dataset, 'get_and_reset_diagnostics'):
                train_metrics.update(train_dataset.get_and_reset_diagnostics())
            if hasattr(train_dataset, 'get_and_reset_debug_records'):
                _append_csv_rows(stitch_debug_path, train_dataset.get_and_reset_debug_records(), i)
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        should_evaluate = (
            FLAGS.eval_only
            or (FLAGS.eval_at_start and i == 1)
            or i % FLAGS.eval_interval == 0
            or i == FLAGS.train_steps
        )
        if should_evaluate:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            # evaluate on each task
            if config.get('policy_conditioning') in {'language', 'goal_language'}:
                periodic_variants = tuple(config.get('language_eval_variants', ('canonical',)))
                final_variants = config.get('language_final_eval_variants')
                if final_variants is None:
                    final_variants = periodic_variants
                use_final_variants = FLAGS.eval_only or i == FLAGS.train_steps
                eval_variants = tuple(final_variants if use_final_variants else periodic_variants)
            else:
                eval_variants = (None,)
            for eval_variant in eval_variants:
                overall_metrics = defaultdict(list)
                metric_prefix = '' if eval_variant in (None, 'canonical') else f'{eval_variant}/'
                for task_id in tqdm.trange(1, num_tasks + 1):
                    task_name = task_infos[task_id - 1]['task_name']
                    eval_info, trajs, cur_renders, episode_summaries = evaluate(
                        agent=eval_agent,
                        env=env,
                        task_id=task_id,
                        config=config,
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=(
                            FLAGS.video_episodes if eval_variant in (None, 'canonical') else 0
                        ),
                        video_frame_skip=FLAGS.video_frame_skip,
                        eval_temperature=FLAGS.eval_temperature,
                        eval_gaussian=FLAGS.eval_gaussian,
                        language_eval_variant=eval_variant or 'canonical',
                        collect_diagnostics=FLAGS.eval_diagnostics,
                        evaluation_seed=FLAGS.eval_seed,
                    )
                    renders.extend(cur_renders)
                    for row in episode_summaries:
                        row['task_name'] = task_name
                    _append_csv_rows(eval_episode_path, episode_summaries, i)
                    metric_names = {'success', *EVAL_DIAGNOSTIC_METRICS}
                    eval_metrics.update(
                        {
                            f'evaluation/{metric_prefix}{task_name}_{k}': v
                            for k, v in eval_info.items()
                            if k in metric_names or k.startswith('failure/')
                        }
                    )
                    for k, v in eval_info.items():
                        if k in metric_names or k.startswith('failure/'):
                            overall_metrics[k].append(v)
                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/{metric_prefix}overall_{k}'] = np.mean(v)
            # print(f"Evaluation metrics at step {i}: {eval_metrics}")
            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if not FLAGS.eval_only and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
