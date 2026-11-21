# VLA-TTT

VLA-TTT is a stateful vision-language-action model for **RoboMME-Plus**. It independently implements fast-weight memory, sequence flow matching, truncated backpropagation through time (TBPTT), and inference-time memory updates from [RoboTTT: Context Scaling for Robot Policies](https://research.nvidia.com/labs/gear/robottt/).

The backbone is a pinned revision of the public `nvidia/GR00T-N1.7-3B` checkpoint. The project calls this a `public_adaptation` because the public model uses Cosmos-Reason2-2B/Qwen3-VL and the paper describes Eagle. It retains all 32 upstream blocks, which alternate between cross-attention and self-attention, and adds a TTT layer after each of the 16 pairs. Reproduction of the paper's results remains unverified.

The repository includes the model, a RoboMME-Plus HDF5 pipeline, commands for training and resuming, and a single-episode inference interface. Validation so far excludes full training runs, simulator runs, standalone tests of differentiable memory, toy tasks, ablations, and performance experiments. See the [implementation notes](implementation.md).

## Environment and installation

Use one Python environment for all components.

| Component | Version |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.9.1 |
| torchvision | 0.24.1 |
| Transformers | 4.57.3 |
| NumPy | 1.26.4 |

For the full model and simulator, use Linux with an NVIDIA GPU. Run these commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[groot,robomme]'
bash scripts/install_groot.sh
```

The `robomme` extra pins the Git revisions of both RoboMME and its ManiSkill fork because pip ignores the upstream `tool.uv.sources` setting and could otherwise install a different ManiSkill version. For data processing and training alone, install `'.[groot]'`. The simulator also requires an NVIDIA graphics driver and Vulkan. See the [official RoboMME installation instructions](https://github.com/RoboMME/robomme_benchmark/blob/1f08ca5a4573eda164906dbfcf2f3511a7b62e90/readme.md#-installation).

Upstream GR00T pins torch 2.9.0 and torchvision 0.24.0. The installation script overrides only those two pins, selecting 2.9.1 and 0.24.1 respectively. The adapter verifies hashes of key model source files against the pinned commit. The script installs the GR00T source with `--no-deps` and uses this project's model dependencies, omitting the full upstream deployment stack. See the [dependency versions](pyproject.toml) and the [GR00T installation script](scripts/install_groot.sh).

The first explicit model construction downloads the pinned GR00T and Cosmos weights. Installation, data preparation, and configuration checks do not construct the model. See the [offline setup instructions](docs/backbone.md).

```bash
python -c 'import sys, torch, torchvision; print(sys.version); print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
vla-ttt check-config --config configs/train/posttrain.yaml
vla-ttt audit-backbone --config configs/model/groot_public.yaml
```

By default, `audit-backbone` prints the expected model sources and configuration. To verify that the weights load, supply local training statistics, use a GPU, and add `--load-model`.

## Data for two-stage training

Pretraining uses the 1,600 episodes in the [original RoboMME HDF5 dataset](https://huggingface.co/datasets/Yinpei/robomme_data_h5). Post-training uses the 50 episodes in the [Plus HDF5 dataset](https://huggingface.co/datasets/Yinpei/robomme_ood_data_h5). Extract the files into their respective directories.

```text
data/robomme/h5/record_dataset_<Task>.h5
data/robomme_plus/h5/record_dataset_<Task>.h5
```

The loader reads HDF5 directly. A LeRobot converter is not yet available. Supply only files from the training split, or list the training episodes explicitly in `data.split_manifest`. Compute normalization statistics from the selected training data.

```bash
vla-ttt validate-data --config configs/train/pretrain.yaml
vla-ttt prepare-data --config configs/train/pretrain.yaml
vla-ttt validate-data --config configs/train/posttrain.yaml
```

The default output is `data/robomme/train_statistics.json`. Both training stages and Plus inference use this file to keep state and action normalization consistent. Plus adaptation reuses the statistics without refitting them. The `prepare-data` command defaults to the pretraining configuration and rejects attempts to overwrite the shared statistics using the post-training configuration. If you choose another path, pass the same `--statistics /path/to/statistics.json` to both stages and inference. The code checks the source and hash of the statistics file.

The training target, `action/joint_action[8]`, contains seven absolute joint angles in radians and one gripper command (`-1` for closed, `+1` for open). The state contains seven joint angles and the mean of the two finger widths in meters. MoveCube's `native_joint_action[15]` has a separate layout. Taking its first eight entries does not produce the standard action.

Both stages retain the entire demonstration prefix and execution trajectory by default. Demonstration frames update memory only. During execution, an action chunk starts every 40 recorded frames, and the intervening frames still update memory. Plus data is recorded at 20 Hz. When the original dataset lacks frequency metadata, the loader records the frequency as unknown and does not infer it from the playback video.

For the original dataset, the default `observation_timing: post_action` pairs each observation with the action in the next row of the same continuous segment. Plus uses `pre_action`, which pairs the observation and action from the same row. The original dataset default follows the pinned recorder source. The recorder version used to generate the released HDF5 samples has yet to be verified. Set `pre_action` explicitly if your files already align observations with the actions that follow them. See the [data documentation](docs/data.md).

## Training

Run the training commands explicitly. Installation and imports do not start training.

```bash
# On the original RoboMME data, freeze GR00T and train the new memory parameters.
vla-ttt train --config configs/train/pretrain.yaml

# On Plus, load outputs/robomme/pretrain/checkpoints/latest.pt by default and fine-tune all parameters.
# Use --init-checkpoint to select different pretrained weights.
vla-ttt train --config configs/train/posttrain.yaml

# Resume an interrupted run with its optimizer, scheduler, RNG state, and data position.
vla-ttt train --config configs/train/posttrain.yaml \
  --resume outputs/robomme_plus/posttrain/checkpoints/latest.pt
```

The `pretrain` stage freezes public GR00T and trains only the added parameters. It uses `video` mode on the original dataset to retain demonstrations for tasks that depend on them. The `posttrain` stage uses `video` mode on Plus and fine-tunes all parameters.

TBPTT limits how far a task gradient can reach. With a long video prefix, the first supervised action may come after the first segment boundary. In that case the sequence supplies no task gradient to the initial fast weights (`W0`), even though memory carries the demonstration forward. A sequence with no demonstration prefix and supervision in the first segment retains that gradient.

This project pretrains on original RoboMME and adapts to Plus. The paper's EgoScale data is outside this pipeline. The configured limits are 30,000 pretraining steps and 20,000 post-training steps. The appropriate training duration for these datasets has yet to be established. Set `training.epochs` to limit the number of passes through the data. Post-training uses FP32 slow parameters and AdamW state with BF16 autocast. Fast weights and inner-loop gradients always use FP32. Training uses the math backend for DiT scaled dot-product attention (SDPA).

The `configs/train/video.yaml` configuration explicitly retains demonstrations. The `bc.yaml` configuration removes them. The `dagger.yaml` configuration supervises only human takeover actions and requires genuine per-frame takeover labels. The official successful demonstrations alone do not supply those labels. The project defaults to `segment_steps=8`, `batch_size=1`, and `context_steps=null`. Their memory use and training performance have yet to be validated.

### Single-node multi-GPU training

The launcher runs pretraining by default. It checks the Python and PyTorch versions, visible GPU count, and NCCL availability, then starts one process per GPU with [torchrun](https://docs.pytorch.org/docs/2.9/elastic/run.html).

```bash
# Train one model on four GPUs. Adjust the device list and count for two or eight GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multi_gpu.sh 4

# Resume pretraining with the same GPU count.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multi_gpu.sh 4 \
  --resume outputs/robomme/pretrain/checkpoints/latest.pt

# Switch to post-training and load only the previous stage's model parameters.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_multi_gpu.sh 4 \
  --config configs/train/posttrain.yaml \
  --init-checkpoint outputs/robomme/pretrain/checkpoints/latest.pt

# Print the launch command without loading the model or starting training processes.
bash scripts/train_multi_gpu.sh 4 --dry-run
```

The launcher runs commands from the repository root. Use absolute paths for custom data locations. Set `VLA_TTT_PYTHON` to select an interpreter. The startup log reports `world_size`, `per_rank_batch_size`, and `global_batch_size`. The per-GPU `training.batch_size` defaults to 1, giving a global batch of 4 on four GPUs. With `total_steps=30000`, the job performs 30,000 synchronized updates. Changing the GPU count does not automatically scale the learning rate.

Each GPU holds a full model replica and processes its own trajectories, with separate fast weights for each episode. If the final sampling round is short, the sampler repeats examples to fill it. Each trajectory stays on one GPU. Faster ranks wait for slower ones when trajectory lengths differ. The default `distributed_timeout_seconds` is 3600. Multi-GPU training with NCCL has yet to be verified on CUDA hardware.

After backpropagating through the segments of a logical batch, each rank normalizes its gradients using the global number of supervised sequences. The ranks synchronize outer parameter gradients and take one optimizer step per logical batch. Rank 0 writes a single checkpoint at logical batch boundaries and writes JSONL logs in the output directory.

Use `--resume` to continue the same run with the same GPU count. Strict resume checks provenance, normalization, key configuration settings, world size, and dataset order. Use `--init-checkpoint` to switch stages or start a new run with a different GPU count. It loads model parameters only, so the new run starts with a fresh optimizer and data position.

## Inputs, outputs, and the RoboMME-Plus interface

A single observation has the following structure.

```python
observation = {
    "images": {"front": front_rgb, "wrist": wrist_rgb},  # uint8 [H,W,3]
    "state": {"joint_position": joint_angles, "gripper": mean_finger_width},  # [7], [1]
    "language": task_goal,
    "embodiment_id": "new_embodiment",
}

policy.configure_inference()
state = policy.init_state(batch_size=1)
state = policy.observe_context(demo_observations, state)  # Observations in chronological order
actions, state = policy.act(observation, state)          # Actions in physical units [1,40,8]
```

The internal training target is normalized flow velocity with shape `[B,S,40,132]`. Supervision covers the first eight physical action dimensions at valid action timesteps. The returned `state` holds the per-layer fast weights, position counters, and commit counters. Pass it back on the next call. Reset it at the start of each episode and carry it from demonstration into execution.

`RoboMMEPlusPolicy` handles the environment's list of observations and caches actions.

```python
from vla_ttt.robomme import RoboMMEPlusPolicy

wrapper = RoboMMEPlusPolicy(policy, task_id="VideoRepick", execution_horizon=40)
wrapper.reset()
action = wrapper.predict(obs, task_goal, error_message=None)  # numpy float32 [8]
```

The wrapper first processes the demonstration list returned by the environment's `reset()`, then generates actions from the final, current observation. While executing cached actions, it retains incoming images and writes them to memory in order before planning again. All denoising iterations read the same prior memory, and only the final candidate state is committed, once. Inference supports `torch.no_grad()`. Keep the TTT inner loop outside `torch.inference_mode()`.

To start the simulator and run a single episode with a trained checkpoint, invoke the inference command explicitly.

```bash
vla-ttt infer --config configs/inference/default.yaml \
  --checkpoint outputs/robomme_plus/posttrain/checkpoints/latest.pt \
  --task VideoRepick --episode 0 --split train
```

This command runs one episode. It does not run an evaluation suite or calculate success rates. Full VLA-TTT inference requires a checkpoint with the project's trained TTT parameters, which the base GR00T checkpoint lacks. For MoveCube, the official control adapter expands the standard eight-dimensional action into the native 15-dimensional format with zero velocity feedforward. This provides interface compatibility. Reproducing the original trajectory or completing the task is not guaranteed.

## Implementation and verification

- `models/` contains the GR00T adapter, 16 differentiable fast-weight MLP layers, register tokens, RoPE, gating, and explicit memory state.
- `data/` handles HDF5 episodes, lazy image loading, BC/video/DAgger masks, and training statistics.
- `training/` implements sequence flow matching, TBPTT, the two-stage parameter-freezing policy, distributed gradient synchronization, and checkpointing.
- `robomme/` handles conversion of observation lists, action caching, the memory lifecycle, and single-episode execution.

Completed checks cover Python syntax and imports, configuration parsing, installation script syntax, and reads, writes, masks, and statistics on temporary HDF5 files. The development machine used different Python and PyTorch versions from the target environment and lacked CUDA. Loading the full model, training, and running the simulator in the specified environment still require end-to-end verification.

Credit for the original method and its citation belongs to the [RoboTTT authors](https://research.nvidia.com/labs/gear/robottt/robottt.pdf). The public backbone comes from [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), and the environment comes from [RoboMME](https://github.com/RoboMME/robomme_benchmark). Third-party code, datasets, and model weights retain their respective licenses.
