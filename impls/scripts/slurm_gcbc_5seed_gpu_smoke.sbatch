#!/bin/bash
#SBATCH --job-name=ogb-gcbc-5smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=00:45:00

set -euo pipefail

PROJECT_ROOT="/bigwork/nhwptarj/Johny-Thesis"
OGBENCH_IMPLS="$PROJECT_ROOT/external/ogbench/impls"
VENV="$PROJECT_ROOT/ogbench_venv"

ENV_NAME="${ENV_NAME:-antmaze-medium-stitch-v0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
RUN_GROUP="${RUN_GROUP:-OGBench_GCBC_5seed_GPU_smoke}"
SAVE_DIR="${SAVE_DIR:-/bigwork/nhwptarj/ogbench_5seed_gpu_smoke_exp}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_TASKS="${EVAL_TASKS:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-2}"
VIDEO_EPISODES="${VIDEO_EPISODES:-1}"
AGENT_CONFIG="${AGENT_CONFIG:-agents/gcbc.py}"
AGENT_EXTRA_ARGS="${AGENT_EXTRA_ARGS:-}"

export HOME="/bigwork/nhwptarj"
export TMPDIR="${TMPDIR:-/bigwork/nhwptarj/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/bigwork/nhwptarj/wandb_ogbench}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/bigwork/nhwptarj/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/bigwork/nhwptarj/wandb_config}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

RUN_ROOT="$SAVE_DIR/slurm_${SLURM_JOB_ID:-manual}"
PROCESS_LOG_DIR="$RUN_ROOT/process_logs"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$SAVE_DIR" "$PROCESS_LOG_DIR"

source "$VENV/bin/activate"
cd "$OGBENCH_IMPLS"

echo "date: $(date --iso-8601=seconds)"
echo "host: $(hostname)"
echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "env_name: $ENV_NAME"
echo "seeds: $SEEDS"
echo "run_group: $RUN_GROUP"
echo "save_dir: $SAVE_DIR"
echo "run_root: $RUN_ROOT"
echo "wandb_mode: $WANDB_MODE"
echo "wandb_dir: $WANDB_DIR"
echo "mujoco_gl: $MUJOCO_GL"
echo "train_steps: $TRAIN_STEPS"
echo "eval_interval: $EVAL_INTERVAL"
echo "eval_tasks: $EVAL_TASKS"
echo "eval_episodes: $EVAL_EPISODES"
echo "video_episodes: $VIDEO_EPISODES"
echo "agent_config: $AGENT_CONFIG"
echo "agent_extra_args: $AGENT_EXTRA_ARGS"
echo "xla_preallocate: $XLA_PYTHON_CLIENT_PREALLOCATE"

python - <<'PY'
import jax
print("jax_devices:", jax.devices())
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

declare -a pids=()
declare -A pid_to_seed=()
read -r -a agent_extra_args <<< "$AGENT_EXTRA_ARGS"

for seed in $SEEDS; do
  seed_log="$PROCESS_LOG_DIR/seed_${seed}.out"
  echo "starting seed $seed, log: $seed_log"
  (
    export WANDB_RUN_GROUP="$RUN_GROUP"
    python main.py \
      --env_name="$ENV_NAME" \
      --seed="$seed" \
      --agent="$AGENT_CONFIG" \
      --train_steps="$TRAIN_STEPS" \
      --log_interval="$LOG_INTERVAL" \
      --eval_interval="$EVAL_INTERVAL" \
      --eval_tasks="$EVAL_TASKS" \
      --eval_episodes="$EVAL_EPISODES" \
      --video_episodes="$VIDEO_EPISODES" \
      --save_interval=1000000 \
      --save_dir="$SAVE_DIR" \
      --run_group="$RUN_GROUP" \
      "${agent_extra_args[@]}"
  ) >"$seed_log" 2>&1 &
  pid=$!
  pids+=("$pid")
  pid_to_seed["$pid"]="$seed"
done

status=0
for pid in "${pids[@]}"; do
  seed="${pid_to_seed[$pid]}"
  if wait "$pid"; then
    echo "seed $seed completed"
  else
    rc=$?
    echo "seed $seed failed with exit code $rc"
    status=1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo "process logs:"
find "$PROCESS_LOG_DIR" -maxdepth 1 -type f -print | sort
echo "run metadata files:"
find "$SAVE_DIR" -path "*$RUN_GROUP*" -name runtime.json -print | sort

exit "$status"
