from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=Path, required=True, help='OGBench experiment save directory.')
    parser.add_argument('--output_dir', type=Path, default=None, help='Directory for plots and summaries.')
    parser.add_argument('--title', type=str, default=None, help='Title prefix for figures.')
    return parser.parse_args()


def safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_')


def infer_seed(path: Path) -> int:
    match = re.search(r'sd(\d+)', str(path))
    if match is None:
        raise ValueError(f'Could not infer seed from path: {path}')
    return int(match.group(1))


def read_csv_rows(path: Path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if value == '':
                item[key] = math.nan
            else:
                try:
                    item[key] = float(value)
                except ValueError:
                    item[key] = value
        parsed.append(item)
    return parsed


def find_csvs(run_dir: Path, name: str):
    return sorted(run_dir.glob(f'OGBench/*/sd*/{name}.csv'))


def mean(values):
    xs = [x for x in values if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def std(values):
    xs = [x for x in values if not math.isnan(x)]
    if not xs:
        return math.nan
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def collect_long(csv_files):
    records = []
    metrics = set()
    for path in csv_files:
        seed = infer_seed(path)
        for row in read_csv_rows(path):
            step = int(row['step'])
            for key, value in row.items():
                if key == 'step' or not isinstance(value, float):
                    continue
                metrics.add(key)
                records.append({'seed': seed, 'step': step, 'metric': key, 'value': value, 'source': str(path)})
    return records, sorted(metrics)


def summarize_by_step(records, metrics):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record['metric'], record['step'])].append(record['value'])

    rows = []
    for metric in metrics:
        steps = sorted(step for (m, step) in grouped if m == metric)
        for step in steps:
            values = grouped[(metric, step)]
            rows.append(
                {
                    'metric': metric,
                    'step': step,
                    'mean': mean(values),
                    'std': std(values),
                    'num_seeds': len(values),
                    'min': min(values),
                    'max': max(values),
                }
            )
    return rows


def final_summary(csv_files):
    per_seed = []
    for path in csv_files:
        rows = read_csv_rows(path)
        if not rows:
            continue
        row = rows[-1]
        seed = infer_seed(path)
        for key, value in row.items():
            if key == 'step' or not isinstance(value, float):
                continue
            per_seed.append({'seed': seed, 'step': int(row['step']), 'metric': key, 'value': value})

    grouped = defaultdict(list)
    for record in per_seed:
        grouped[record['metric']].append(record['value'])

    aggregate = []
    for metric in sorted(grouped):
        values = grouped[metric]
        aggregate.append(
            {
                'metric': metric,
                'mean': mean(values),
                'std': std(values),
                'num_seeds': len(values),
                'min': min(values),
                'max': max(values),
            }
        )
    return per_seed, aggregate


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=2)


def records_for_metric(records, metric):
    by_seed = defaultdict(list)
    for record in records:
        if record['metric'] == metric:
            by_seed[record['seed']].append((record['step'], record['value']))
    return {seed: sorted(values) for seed, values in by_seed.items()}


def summary_for_metric(summary_rows, metric):
    rows = [row for row in summary_rows if row['metric'] == metric]
    return sorted(rows, key=lambda row: row['step'])


def plot_metric(records, summary_rows, metric, output_path, title_prefix=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for seed, values in records_for_metric(records, metric).items():
        xs = [x for x, _ in values]
        ys = [y for _, y in values]
        ax.plot(xs, ys, color='0.75', linewidth=1.0, alpha=0.75)

    rows = summary_for_metric(summary_rows, metric)
    xs = [row['step'] for row in rows]
    mus = [row['mean'] for row in rows]
    sds = [row['std'] for row in rows]
    lower = [mu - sd for mu, sd in zip(mus, sds)]
    upper = [mu + sd for mu, sd in zip(mus, sds)]
    ax.plot(xs, mus, color='#1f77b4', linewidth=2.4, label='mean')
    ax.fill_between(xs, lower, upper, color='#1f77b4', alpha=0.18, label='std')

    ax.set_xlabel('training step')
    ax.set_ylabel(ylabel or metric)
    title = metric if title_prefix is None else f'{title_prefix}: {metric}'
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_combined(summary_rows, metrics, output_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for metric in metrics:
        rows = summary_for_metric(summary_rows, metric)
        if not rows:
            continue
        xs = [row['step'] for row in rows]
        ys = [row['mean'] for row in rows]
        ax.plot(xs, ys, linewidth=2.0, marker='o', markersize=3, label=metric.split('/')[-1])
    ax.set_xlabel('training step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_bars(final_aggregate, output_path, title, ylabel):
    metrics = [row['metric'] for row in final_aggregate]
    means = [row['mean'] for row in final_aggregate]
    errors = [row['std'] for row in final_aggregate]
    labels = [metric.split('/')[-1].replace('_success', '') for metric in metrics]

    fig_width = max(8, 1.25 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    ax.bar(labels, means, yerr=errors, capsize=4, color='#4c78a8', alpha=0.9)
    ax.set_ylim(0, 1.0 if all('success' in metric for metric in metrics) else None)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.25)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_per_seed(final_per_seed, output_path, title):
    rows = [row for row in final_per_seed if row['metric'] == 'evaluation/overall_success']
    rows = sorted(rows, key=lambda row: row['seed'])
    if not rows:
        return
    labels = [f'seed {row["seed"]}' for row in rows]
    values = [row['value'] for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(labels, values, color='#59a14f', alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel('overall success')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(path, run_dir, eval_final_aggregate, train_final_aggregate):
    overall = next((row for row in eval_final_aggregate if row['metric'] == 'evaluation/overall_success'), None)
    task_rows = [row for row in eval_final_aggregate if row['metric'].startswith('evaluation/task')]
    task_rows = sorted(task_rows, key=lambda row: row['metric'])

    lines = [
        '# OGBench Run Summary',
        '',
        f'Run directory: `{run_dir}`',
        '',
    ]
    if overall is not None:
        lines.extend(
            [
                '## Final Evaluation',
                '',
                f"- Overall success: `{overall['mean']:.3f} ± {overall['std']:.3f}` over {overall['num_seeds']} seeds.",
                '',
                '| Metric | Mean | Std | Min | Max |',
                '|---|---:|---:|---:|---:|',
            ]
        )
        for row in [overall] + task_rows:
            lines.append(
                f"| `{row['metric']}` | {row['mean']:.3f} | {row['std']:.3f} | {row['min']:.3f} | {row['max']:.3f} |"
            )
        lines.append('')

    if train_final_aggregate:
        selected = [
            'training/actor/actor_loss',
            'training/actor/mse',
            'validation/actor/actor_loss',
            'validation/actor/mse',
            'training/grad/norm',
        ]
        train_by_metric = {row['metric']: row for row in train_final_aggregate}
        lines.extend(['## Final Training Metrics', '', '| Metric | Mean | Std |', '|---|---:|---:|'])
        for metric in selected:
            row = train_by_metric.get(metric)
            if row is not None:
                lines.append(f"| `{metric}` | {row['mean']:.4f} | {row['std']:.4f} |")
        lines.append('')

    path.write_text('\n'.join(lines))


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / 'plots').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = args.title or run_dir.name

    eval_files = find_csvs(run_dir, 'eval')
    train_files = find_csvs(run_dir, 'train')
    if not eval_files:
        raise FileNotFoundError(f'No eval.csv files found under {run_dir}')
    if not train_files:
        raise FileNotFoundError(f'No train.csv files found under {run_dir}')

    eval_records, eval_metrics = collect_long(eval_files)
    train_records, train_metrics = collect_long(train_files)
    eval_summary = summarize_by_step(eval_records, eval_metrics)
    train_summary = summarize_by_step(train_records, train_metrics)
    eval_final_per_seed, eval_final_aggregate = final_summary(eval_files)
    train_final_per_seed, train_final_aggregate = final_summary(train_files)

    write_csv(output_dir / 'eval_long.csv', eval_records, ['seed', 'step', 'metric', 'value', 'source'])
    write_csv(output_dir / 'eval_summary_by_step.csv', eval_summary, ['metric', 'step', 'mean', 'std', 'num_seeds', 'min', 'max'])
    write_csv(output_dir / 'final_eval_per_seed.csv', eval_final_per_seed, ['seed', 'step', 'metric', 'value'])
    write_csv(output_dir / 'final_eval_summary.csv', eval_final_aggregate, ['metric', 'mean', 'std', 'num_seeds', 'min', 'max'])

    write_csv(output_dir / 'train_long.csv', train_records, ['seed', 'step', 'metric', 'value', 'source'])
    write_csv(output_dir / 'train_summary_by_step.csv', train_summary, ['metric', 'step', 'mean', 'std', 'num_seeds', 'min', 'max'])
    write_csv(output_dir / 'final_train_per_seed.csv', train_final_per_seed, ['seed', 'step', 'metric', 'value'])
    write_csv(output_dir / 'final_train_summary.csv', train_final_aggregate, ['metric', 'mean', 'std', 'num_seeds', 'min', 'max'])

    write_json(
        output_dir / 'summary.json',
        {
            'run_dir': str(run_dir),
            'num_eval_files': len(eval_files),
            'num_train_files': len(train_files),
            'eval_metrics': eval_metrics,
            'train_metrics': train_metrics,
            'final_eval_summary': eval_final_aggregate,
            'final_train_summary': train_final_aggregate,
        },
    )

    metric_dir = output_dir / 'metric_plots'
    metric_dir.mkdir(exist_ok=True)
    for metric in eval_metrics:
        plot_metric(
            eval_records,
            eval_summary,
            metric,
            metric_dir / f'eval_{safe_name(metric)}.png',
            title_prefix=title_prefix,
            ylabel='success',
        )
    for metric in train_metrics:
        plot_metric(
            train_records,
            train_summary,
            metric,
            metric_dir / f'train_{safe_name(metric)}.png',
            title_prefix=title_prefix,
            ylabel=metric,
        )

    eval_task_metrics = [m for m in eval_metrics if m.startswith('evaluation/task')]
    eval_success_metrics = eval_task_metrics + ['evaluation/overall_success']
    plot_combined(
        eval_summary,
        eval_success_metrics,
        output_dir / 'eval_success_all_metrics.png',
        f'{title_prefix}: evaluation success',
        'success',
    )
    plot_metric(
        eval_records,
        eval_summary,
        'evaluation/overall_success',
        output_dir / 'eval_overall_success.png',
        title_prefix=title_prefix,
        ylabel='overall success',
    )
    plot_final_bars(
        eval_final_aggregate,
        output_dir / 'final_eval_success_bars.png',
        f'{title_prefix}: final evaluation success',
        'success',
    )
    plot_final_per_seed(
        eval_final_per_seed,
        output_dir / 'final_overall_success_per_seed.png',
        f'{title_prefix}: final overall success per seed',
    )

    plot_combined(
        train_summary,
        ['training/actor/actor_loss', 'validation/actor/actor_loss'],
        output_dir / 'train_validation_actor_loss.png',
        f'{title_prefix}: actor loss',
        'negative log likelihood',
    )
    plot_combined(
        train_summary,
        ['training/actor/mse', 'validation/actor/mse'],
        output_dir / 'train_validation_actor_mse.png',
        f'{title_prefix}: actor MSE',
        'MSE',
    )
    plot_metric(
        train_records,
        train_summary,
        'training/grad/norm',
        output_dir / 'train_grad_norm.png',
        title_prefix=title_prefix,
        ylabel='gradient norm',
    )

    write_markdown_report(output_dir / 'report.md', run_dir, eval_final_aggregate, train_final_aggregate)

    print(f'Wrote plots and summaries to: {output_dir}')
    print('Key files:')
    for name in [
        'report.md',
        'summary.json',
        'final_eval_summary.csv',
        'eval_success_all_metrics.png',
        'eval_overall_success.png',
        'final_eval_success_bars.png',
        'final_overall_success_per_seed.png',
        'train_validation_actor_loss.png',
        'train_validation_actor_mse.png',
        'train_grad_norm.png',
    ]:
        print(f'  {output_dir / name}')


if __name__ == '__main__':
    main()
