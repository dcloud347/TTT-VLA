"""Explicit GR00T N1.7 DiT execution with interleaved VLA-TTT memory.

The upstream modules and checkpoint are retained; this adapter exposes each
cross/self-attention pair instead of using forward hooks. See docs/backbone.md.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

GROOT_COMMIT = "3b49067943eb1ab2eb2203b93d461c4390c78da3"
GROOT_MODEL_ID = "nvidia/GR00T-N1.7-3B"
GROOT_REVISION = "373d34e2723d928870157d07dc216a19a624b5bb"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
UPSTREAM_SOURCE_HASHES = {
    "gr00t.model.gr00t_n1d7.gr00t_n1d7": "1bce76ec4594af1971e40bda91ec416ae1f06403905c0286333fdda1ece63f57",
    "gr00t.model.modules.dit": "2c88f29f3c84061fc605ded3d8948ec1a4086ac26310143de23bce1a6540d2de",
    "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7": "2ea735d9dce1bd9a8f4b03edef71a203bceff6dbdd5410390191394b24584dd3",
}
FIELDS = (("joint_position", 7), ("gripper", 1))
EMBODIMENT = "new_embodiment"


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _state_validity(observation: Mapping[str, Any]) -> np.ndarray:
    validity = observation.get("modality_valid", {}).get("state")
    if validity is None:
        return np.full(8, observation.get("state") is not None, dtype=np.bool_)
    if isinstance(validity, Mapping):
        parts = []
        for name, size in FIELDS:
            value = validity.get(name, validity.get("gripper_position", False) if name == "gripper" else False)
            parts.append(np.broadcast_to(_numpy(value).astype(np.bool_), (size,)))
        return np.concatenate(parts)
    return np.broadcast_to(_numpy(validity).astype(np.bool_), (8,)).copy()


def _states(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Keep physical state for upstream normalization and action decoding."""
    state = observation.get("state")
    if state is None:
        if not observation.get("is_video_demo", False):
            raise ValueError("RoboMME observations require the physical eight-dimensional state")
        state = np.zeros(8, dtype=np.float32)
    if not isinstance(state, Mapping):
        array = _numpy(state).astype(np.float32)
        if array.shape[-1] != 8:
            raise ValueError(f"Expected state8, received {array.shape}")
        state = {"joint_position": array[..., :7], "gripper": array[..., 7:8]}
    state = dict(state)
    if "gripper" not in state and "gripper_position" in state:
        state["gripper"] = state["gripper_position"]
    result = {}
    validity = _state_validity(observation)
    offset = 0
    for name, dimension in FIELDS:
        value = _numpy(state[name]).astype(np.float32)
        if value.ndim == 1:
            value = value[None, :]
        if value.shape != (1, dimension):
            raise ValueError(f"state.{name} must have shape ({dimension},) or (1,{dimension})")
        field_valid = validity[offset:offset + dimension][None]
        if np.any(~np.isfinite(value) & field_valid):
            raise ValueError(f"Valid state.{name} contains non-finite values")
        value = np.where(field_valid, value, 0.0).astype(np.float32)
        result[name] = value
        offset += dimension
    return result


def _statistics(path: str | Path, use_percentiles: bool) -> dict:
    """Read train-set statistics; never synthesize statistics from test data."""
    source = json.loads(Path(path).read_text())
    statistics = source.get("statistics", source)
    if EMBODIMENT in statistics:
        statistics = statistics[EMBODIMENT]
    result: dict[str, dict] = {}
    required = ("min", "max", "mean", "std") + (("q01", "q99") if use_percentiles else ())
    for modality in ("state", "action"):
        values = statistics[modality]
        result[modality] = {}
        offset = 0
        for name, size in FIELDS:
            if name in values:
                field_stats = values[name]
            elif name == "gripper" and "gripper_position" in values:
                field_stats = values["gripper_position"]
            else:
                field_stats = {key: value[offset:offset + size] for key, value in values.items()
                               if key in ("min", "max", "mean", "std", "q01", "q99")}
            checked = {}
            for key in required:
                array = np.asarray(field_stats[key], dtype=np.float64)
                if array.shape != (size,) or not np.isfinite(array).all():
                    raise ValueError(f"Statistics {modality}.{name}.{key} must contain {size} finite values")
                checked[key] = array.tolist()
            if np.any(np.asarray(checked["max"]) < np.asarray(checked["min"])):
                raise ValueError(f"Inverted normalization range for {modality}.{name}")
            result[modality][name] = checked
            offset += size
    return {EMBODIMENT: result}


def _modalities(horizon: int) -> dict:
    absolute = {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"}
    return {EMBODIMENT: {
        "video": {"delta_indices": [0], "modality_keys": ["front", "wrist"]},
        "state": {"delta_indices": [0], "modality_keys": [name for name, _ in FIELDS]},
        "action": {"delta_indices": list(range(horizon)), "modality_keys": [name for name, _ in FIELDS],
                   "action_configs": [dict(absolute), dict(absolute)]},
        "language": {"delta_indices": [0], "modality_keys": ["annotation.human.action.task_description"]},
    }}


class GR00TAdapter(nn.Module):
    """GR00T checkpoint + official processor, specialized to RoboMME joint_angle8."""

    def __init__(self, model: nn.Module, processor: Any, *, provenance: dict | None = None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.provenance = provenance or {}
        self.hidden_size = int(self.dit.inner_dim)
        self.action_dim = int(self.head.action_dim)
        self.action_horizon = int(self.head.action_horizon)
        self.num_memory_layers = len(self.dit.transformer_blocks) // 2
        self.stage = "pretrain"
        if not self.dit.config.interleave_self_attention or len(self.dit.transformer_blocks) % 2:
            raise ValueError("VLA-TTT requires complete alternating cross/self-attention pairs")
        if (self.hidden_size, self.action_dim, self.action_horizon, self.num_memory_layers) != (1536, 132, 40, 16):
            raise ValueError("Checkpoint architecture differs from the pinned N1.7 adaptation; audit it before use")
        if int(model.config.state_history_length) != 1:
            raise ValueError("This adapter accepts the current state only; state_history_length must be one")
        self.configure_trainable("pretrain")

    @property
    def head(self):
        return self.model.action_head

    @property
    def dit(self):
        return self.model.action_head.model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GR00TAdapter":
        """Load pinned upstream weights only when explicitly constructing a policy.

        `statistics_path` must describe the RoboMME training split. The optional
        local paths allow a fully offline installation after artifact download.
        """
        upstream_path = config.get("upstream_path")
        if upstream_path:
            sys.path.insert(0, str(Path(upstream_path).expanduser().resolve()))
        try:
            from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
            from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
            from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
            from transformers.utils import cached_file
        except ImportError as exc:
            raise ImportError("Install the [groot] dependencies and run scripts/install_groot.sh first") from exc
        for module_name, expected in UPSTREAM_SOURCE_HASHES.items():
            source = Path(importlib.import_module(module_name).__file__)
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Unpinned GR00T source at {source}; install upstream commit {GROOT_COMMIT}")
        if not config.get("statistics_path"):
            raise ValueError("model.statistics_path must point to RoboMME training-set statistics")
        percentiles = bool(config.get("use_percentiles", False))
        stats = _statistics(config["statistics_path"], percentiles)
        checkpoint = str(config.get("checkpoint", config.get("checkpoint_path", GROOT_MODEL_ID)))
        revision = str(config.get("revision", GROOT_REVISION))
        cache_dir = config.get("cache_dir")
        local_only = bool(config.get("local_files_only", False))
        hub = {"revision": revision, "local_files_only": local_only, "cache_dir": cache_dir}
        model_config = Gr00tN1d7Config.from_pretrained(checkpoint, **hub)
        model_config.use_flash_attention = bool(config.get("use_flash_attention", False))
        # Only eight physical dimensions are supervised. Padded dimensions must
        # not influence attention or memory, including when sampled from noise.
        model_config.strict_action_padding_mask = True
        model_config.strict_vl_padding_mask = True
        model_config.use_relative_action = False
        dtype_name = str(config.get("dtype", "bfloat16")).removeprefix("torch.")
        if dtype_name not in ("float32", "bfloat16", "float16"):
            raise ValueError(f"Unsupported model dtype: {dtype_name}")
        dtype = getattr(torch, dtype_name)
        model_config.load_bf16 = dtype == torch.bfloat16
        backbone_path = config.get("backbone_path")
        if backbone_path:
            # Upstream get_backbone_cls recognizes the model name. Keep its
            # canonical name and use the HF cache for offline operation instead.
            raise ValueError("Use cache_dir + local_files_only for Cosmos; upstream dispatch requires its canonical model ID")
        backbone_hub = {"revision": COSMOS_REVISION, "local_files_only": local_only,
                        "cache_dir": cache_dir, "trust_remote_code": False}
        model, loading = Gr00tN1d7.from_pretrained(
            checkpoint, config=model_config, torch_dtype=dtype, output_loading_info=True,
            transformers_loading_kwargs=backbone_hub, **hub,
        )
        bad = {key: loading.get(key, []) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        if any(bad.values()):
            raise RuntimeError(f"Checkpoint loading is not exact: {bad}")
        processor_source = str(config.get("processor_path", checkpoint))
        config_file = Path(processor_source) / "processor_config.json"
        if not config_file.is_file():
            config_file = Path(cached_file(processor_source, "processor_config.json", **hub))
        processor_kwargs = json.loads(config_file.read_text())["processor_kwargs"]
        processor_kwargs.update(
            modality_configs=_modalities(int(model_config.action_horizon)), statistics=stats,
            use_relative_action=False, use_percentiles=percentiles, apply_sincos_state_encoding=False,
            use_mean_std=False, max_state_dim=int(model_config.max_state_dim),
            max_action_dim=int(model_config.max_action_dim), max_action_horizon=int(model_config.action_horizon),
            strict_action_padding_mask=True, exclude_state=False, state_dropout_prob=0.0,
            model_name=model_config.model_name, model_type=model_config.backbone_model_type,
            embodiment_id_mapping={EMBODIMENT: 10}, transformers_loading_kwargs=backbone_hub,
        )
        processor = Gr00tN1d7Processor(**processor_kwargs)
        processor.set_statistics(stats, override=True)
        model.to(device=config.get("device", "cuda"), dtype=dtype)
        provenance = {"github_commit": GROOT_COMMIT, "checkpoint": checkpoint, "revision": revision,
                      "cosmos_revision": COSMOS_REVISION, "statistics_sha256": hashlib.sha256(
                          Path(config["statistics_path"]).read_bytes()).hexdigest(),
                      "reproduction_level": "public_backbone_adaptation"}
        result = cls(model, processor, provenance=provenance)
        result.configure_trainable(str(config.get("stage", "pretrain")))
        return result

    from_pretrained = from_config

    def configure_trainable(self, stage: str) -> None:
        if stage not in ("pretrain", "posttrain", "finetune", "inference"):
            raise ValueError(f"Unknown training stage {stage!r}")
        self.stage = stage
        train_backbone = stage in ("posttrain", "finetune")
        self.model.requires_grad_(train_backbone)
        self.model.backbone.set_trainable_parameters(train_backbone, train_backbone, 0)
        self.head.set_trainable_parameters(train_backbone, train_backbone, train_backbone)
        if not train_backbone:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.stage in ("pretrain", "inference"):
            self.model.eval()
        self.processor.train() if mode and self.stage != "inference" else self.processor.eval()
        return self

    def prepare_observations(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Encode one batched physical observation; preserve training autograd."""
        if not observations or any(obs is None for obs in observations):
            raise ValueError("prepare_observations requires non-padding observations")
        features, states, state_valid = [], [], []
        for obs in observations:
            if obs.get("embodiment_id", EMBODIMENT) != EMBODIMENT:
                raise ValueError("This adapter is configured for RoboMME new_embodiment only")
            state = _states(obs)
            states.append(state)
            normalized = self.processor.state_action_processor.apply_state(state, EMBODIMENT)
            state_tensor = torch.from_numpy(np.concatenate([normalized[name] for name, _ in FIELDS], axis=-1))
            dimension_validity = _state_validity(obs)
            state_tensor = state_tensor * torch.from_numpy(dimension_validity)[None]
            state_tensor = F.pad(state_tensor, (0, int(self.model.config.max_state_dim) - 8))
            images = {}
            camera_validity = obs.get("modality_valid", {}).get("images", {})
            for camera in ("front", "wrist"):
                if not camera_validity.get(camera, camera in obs["images"]):
                    continue
                value = _numpy(obs["images"][camera])
                if value.ndim == 3:
                    value = value[None]
                if value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != 3 or value.dtype != np.uint8:
                    raise ValueError(f"images.{camera} must be RGB uint8 [H,W,3] or [1,H,W,3]")
                from PIL import Image
                images[camera] = [Image.fromarray(value[0])]
            if not images:
                raise ValueError("A RoboMME observation must contain at least one valid RGB camera")
            language = str(obs["language"])
            if self.processor.formalize_language:
                import re
                language = re.sub(r"[^\w\s]", "", language.lower())
            transform = self.processor.train_image_transform if self.processor.training else self.processor.eval_image_transform
            sample = self.processor._get_vlm_inputs(list(images), images, None, transform, language)
            action_mask = torch.zeros(self.action_horizon, self.action_dim)
            action_mask[:, :8] = 1
            sample.update(state=state_tensor.float(), embodiment_id=10, action_mask=action_mask)
            features.append(sample)
            state_valid.append(bool(dimension_validity.any()))
        inputs = self.processor.collator(features)["inputs"]
        backbone_inputs, action_inputs = self.model.prepare_input(inputs)
        backbone_trainable = any(p.requires_grad for p in self.model.backbone.parameters())
        with nullcontext() if backbone_trainable else torch.no_grad():
            backbone_output = self.model.backbone(backbone_inputs)
        encoded = self.head._encode_features(backbone_output, action_inputs)
        state_features = encoded["state_features"]
        state_mask = torch.as_tensor(state_valid, device=state_features.device, dtype=torch.bool)
        if self.training and self.stage in ("posttrain", "finetune") and self.head.state_dropout_prob > 0:
            state_mask = state_mask & (torch.rand_like(state_mask.float()) >= self.head.state_dropout_prob)
        state_features = state_features * state_mask[:, None, None]
        return {"backbone_features": encoded["backbone_features"], "state_features": state_features,
                "backbone_attention_mask": backbone_output["backbone_attention_mask"],
                "image_mask": backbone_output.get("image_mask"), "state_valid": state_mask,
                "embodiment_id": action_inputs["embodiment_id"], "action_mask": action_inputs["action_mask"],
                "physical_states": states, "batch_size": len(observations)}

    def encode_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return self.prepare_observations(observation["samples"] if "samples" in observation else [observation])

    def encode_tokens(self, prepared: Mapping[str, Any], noisy_actions: Tensor, tau: Tensor,
                      register_tokens: Tensor) -> dict[str, Any]:
        if noisy_actions.shape != (prepared["batch_size"], self.action_horizon, self.action_dim):
            raise ValueError(f"Expected [B,{self.action_horizon},{self.action_dim}] noisy actions")
        batch = noisy_actions.shape[0]
        tau = torch.as_tensor(tau, device=self.device).reshape(-1)
        if tau.numel() == 1:
            tau = tau.expand(batch)
        if tau.numel() != batch or torch.any((tau < 0) | (tau > 1)):
            raise ValueError("tau must provide one [0,1] flow time per sample")
        discretized = (tau * self.head.num_timestep_buckets).long()
        action_mask = prepared["action_mask"].to(device=self.device, dtype=self.dtype)
        noisy_actions = noisy_actions.to(device=self.device, dtype=self.dtype) * action_mask
        action_features = self.head.action_encoder(noisy_actions, discretized, prepared["embodiment_id"])
        if self.head.config.add_pos_embed:
            positions = torch.arange(self.action_horizon, device=self.device)
            action_features = action_features + self.head.position_embedding(positions)[None]
        if register_tokens.ndim == 2:
            register_tokens = register_tokens[None].expand(batch, -1, -1)
        if register_tokens.shape[0] != batch or register_tokens.shape[-1] != self.hidden_size:
            raise ValueError("Register tokens must be [N,D] or [B,N,D] in the DiT hidden dimension")
        registers = register_tokens.to(device=self.device, dtype=self.dtype)
        tokens = torch.cat((registers, prepared["state_features"], action_features), dim=1)
        mask = torch.cat((torch.ones(batch, registers.shape[1], device=self.device, dtype=torch.bool),
                          prepared["state_valid"][:, None], action_mask.any(-1)), dim=1)
        return {**prepared, "tokens": tokens * mask[..., None], "hidden_attention_mask": mask, "token_mask": mask,
                "temb": self.dit.timestep_encoder(discretized), "action_mask": action_mask}

    def run_layer(self, index: int, payload: Mapping[str, Any], hidden_states: Tensor) -> Tensor:
        """Execute upstream cross/self pair; caller inserts TTT after this returns."""
        if not 0 <= index < self.num_memory_layers:
            raise IndexError(index)
        hidden_mask = payload["hidden_attention_mask"]
        for block_index in (2 * index, 2 * index + 1):
            block = self.dit.transformer_blocks[block_index]
            if block_index % 2:
                hidden_states = block(hidden_states, attention_mask=hidden_mask,
                                      encoder_hidden_states=None, encoder_attention_mask=None,
                                      temb=payload["temb"])
            else:
                encoder_mask = payload["backbone_attention_mask"]
                if self.head.config.use_alternate_vl_dit:
                    image_mask = payload["image_mask"]
                    if image_mask is None:
                        raise ValueError("AlternateVLDiT requires the backbone's image-token mask")
                    use_text = block_index % (2 * self.dit.attend_text_every_n_blocks) == 0
                    encoder_mask = encoder_mask & (~image_mask if use_text else image_mask)
                hidden_states = block(hidden_states, attention_mask=None,
                                      encoder_hidden_states=payload["backbone_features"],
                                      encoder_attention_mask=encoder_mask, temb=payload["temb"])
            hidden_states = hidden_states * hidden_mask[..., None]
        return hidden_states

    def decode_actions(self, hidden_states: Tensor, payload: Mapping[str, Any]) -> Tensor:
        """Decode normalized flow velocity after the final TTT layer."""
        shift, scale = self.dit.proj_out_1(F.silu(payload["temb"])).chunk(2, dim=1)
        hidden_states = self.dit.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = hidden_states * payload["hidden_attention_mask"][..., None]
        output = self.dit.proj_out_2(hidden_states)
        output = output * payload["hidden_attention_mask"][..., None]
        velocity = self.head.action_decoder(output, payload["embodiment_id"])[:, -self.action_horizon:]
        return velocity * payload["action_mask"]

    def normalize_actions(self, raw: Tensor, observations: Sequence[Sequence[Mapping[str, Any] | None]]) -> tuple[Tensor, Tensor]:
        """Physical joint targets [B,S,H,8] -> padded normalized targets and mask."""
        if raw.ndim != 4 or raw.shape[-1] != 8 or raw.shape[-2] > self.action_horizon:
            raise ValueError("normalize_actions expects [B,S,H<=40,8] physical joint targets")
        array = raw.detach().float().cpu().numpy()
        output = np.zeros((*array.shape[:2], self.action_horizon, self.action_dim), dtype=np.float32)
        valid = np.zeros_like(output, dtype=np.bool_)
        if len(observations) != array.shape[0] or any(len(row) != array.shape[1] for row in observations):
            raise ValueError("Observation shape must match [B,S] action axes")
        for batch, row in enumerate(observations):
            for time, obs in enumerate(row):
                if obs is None or obs.get("is_video_demo", False):
                    continue
                values = array[batch, time]
                if not np.isfinite(values).all():
                    raise ValueError("Physical action targets contain non-finite values")
                normalized = self.processor.state_action_processor.apply_action(
                    {"joint_position": values[:, :7], "gripper": values[:, 7:8]}, EMBODIMENT,
                    state=_states(obs),
                )
                output[batch, time, :values.shape[0], :8] = np.concatenate(
                    [normalized[name] for name, _ in FIELDS], axis=-1)
                valid[batch, time, :values.shape[0], :8] = True
        return torch.as_tensor(output, device=raw.device, dtype=raw.dtype), torch.as_tensor(valid, device=raw.device)

    def decode_action(self, normalized: Tensor, observations: Sequence[Mapping[str, Any]]) -> Tensor:
        """Normalized [B,H,132] -> joint_angle8 physical targets, no relative EEF conversion."""
        from gr00t.data.embodiment_tags import EmbodimentTag
        if normalized.ndim != 3 or normalized.shape[0] != len(observations) or normalized.shape[-1] != self.action_dim:
            raise ValueError("Expected normalized [B,H,132] and B raw observations")
        states = [_states(obs) for obs in observations]
        batched = {name: np.stack([state[name] for state in states]) for name, _ in FIELDS}
        physical = self.processor.decode_action(normalized.detach().float().cpu().numpy(),
                                                EmbodimentTag.NEW_EMBODIMENT, batched)
        action = np.concatenate([physical[name] for name, _ in FIELDS], axis=-1)
        return torch.as_tensor(action, device=normalized.device, dtype=torch.float32)

    def audit(self) -> dict[str, Any]:
        modules = ("gr00t.model.gr00t_n1d7.gr00t_n1d7", "gr00t.model.modules.dit",
                   "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
        source_hashes = {}
        for name in modules:
            filename = Path(importlib.import_module(name).__file__)
            source_hashes[name] = {"path": str(filename), "sha256": hashlib.sha256(filename.read_bytes()).hexdigest()}
        return {**self.provenance, "hidden_size": self.hidden_size, "action_dim": self.action_dim,
                "physical_action_dim": 8, "action_horizon": self.action_horizon,
                "num_memory_layers": self.num_memory_layers, "num_upstream_blocks": len(self.dit.transformer_blocks),
                "vlm": self.model.config.model_name, "embodiment_projector": 10,
                "sources": source_hashes, "num_parameters": sum(p.numel() for p in self.parameters()),
                "use_relative_action": self.processor.use_relative_action}


GrootAdapter = GR00TAdapter
