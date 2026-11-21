#!/usr/bin/env bash
# Single-node training: one process per visible GPU, one shared optimization job.
set -euo pipefail

if [[ $# -lt 1 || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/train_multi_gpu.sh GPU_COUNT [--dry-run] [vla-ttt train arguments...]"
    echo "Example: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multi_gpu.sh 4"
    exit 0
fi
task_gpus="$1"
shift
if [[ ! "$task_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_COUNT must be a positive integer." >&2
    exit 2
fi
task_dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    task_dry_run=true
    shift
fi
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${VLA_TTT_PYTHON:-python}"
cd -- "$task_root"
task_command=("$task_python" -m torch.distributed.run --standalone --nnodes=1
    --nproc-per-node="$task_gpus" --max-restarts=0 -m vla_ttt train
    --config configs/train/pretrain.yaml "$@")
if "$task_dry_run"; then
    printf '%q ' "${task_command[@]}"
    printf '\n'
    exit 0
fi
"$task_python" - "$task_gpus" <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Activate the project's Python 3.12 environment first.")
import torch
import torchvision
if torch.__version__.split('+')[0] != '2.9.1' or torchvision.__version__.split('+')[0] != '0.24.1':
    raise SystemExit("This project requires torch 2.9.1 and torchvision 0.24.1.")
count = int(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() < count:
    raise SystemExit(f"Requested {count} GPUs; only {torch.cuda.device_count()} CUDA GPUs are visible.")
if not torch.distributed.is_nccl_available():
    raise SystemExit("GPU training requires a PyTorch build with NCCL support.")
for index in range(count):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
PY
exec "${task_command[@]}"
