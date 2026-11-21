"""Stateful VLA policy shared by sequence training and online inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .fast_weights import MemoryState
from .sequence_dit import SequenceDiT
from .ttt_block import TTTConfig


@dataclass(frozen=True)
class SequenceOutput:
    velocity: Tensor
    state: MemoryState


class VLATTTPolicy(nn.Module):
    def __init__(self, adapter: nn.Module, ttt_config: TTTConfig | dict | None = None, *, denoising_steps: int = 4):
        super().__init__()
        self.adapter = adapter
        if isinstance(ttt_config, dict) or ttt_config is None:
            ttt_config = TTTConfig.from_dict(ttt_config or {})
        self.ttt_config = ttt_config
        self.sequence_dit = SequenceDiT(adapter.hidden_size, adapter.num_memory_layers, ttt_config)
        self.sequence_dit.to(device=self.device)
        if denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        self.denoising_steps = int(denoising_steps)
        self.stage = getattr(adapter, "stage", "pretrain")

    @classmethod
    def from_config(cls, config: dict) -> VLATTTPolicy:
        from .adapters.groot import GR00TAdapter

        adapter = GR00TAdapter.from_config(config.get("model", config))
        policy = cls(adapter, config.get("ttt", {}), denoising_steps=config.get("inference", {}).get("denoising_steps", 4))
        stage = config.get("training", {}).get("stage", config.get("model", {}).get("stage", "posttrain"))
        if stage == "inference":
            policy.configure_inference()
        else:
            policy.configure_trainable(stage)
        return policy

    @property
    def device(self) -> torch.device:
        return next(self.adapter.parameters()).device

    @property
    def action_horizon(self) -> int:
        return self.adapter.action_horizon

    @property
    def action_dim(self) -> int:
        return self.adapter.action_dim

    def configure_trainable(self, stage: str) -> None:
        import os

        # Higher-order training must not inherit an upstream inference override.
        os.environ["GR00T_DIT_SDPA_MODE"] = "math"
        if stage not in {"pretrain", "posttrain"}:
            raise ValueError("stage must be pretrain or posttrain")
        self.stage = stage
        self.adapter.requires_grad_(stage == "posttrain")
        self.sequence_dit.requires_grad_(True)
        if hasattr(self.adapter, "configure_trainable"):
            self.adapter.configure_trainable(stage)
        self.train()

    def configure_inference(self) -> None:
        self.stage = "inference"
        self.requires_grad_(False)
        if hasattr(self.adapter, "configure_trainable"):
            self.adapter.configure_trainable("inference")
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "stage", "posttrain") == "pretrain":
            # Frozen modules still run with autograd where needed; eval disables dropout only.
            self.adapter.eval()
        return self

    def init_state(self, batch_size: int = 1, device=None, session_ids=()) -> MemoryState:
        if session_ids and len(session_ids) != batch_size:
            raise ValueError("One session ID is required per batch member")
        return self.sequence_dit.init_state(
            batch_size, training=self.training and torch.is_grad_enabled(), device=device or self.device,
            session_ids=session_ids,
        )

    @staticmethod
    def detach_state(state: MemoryState, requires_grad: bool = True) -> MemoryState:
        return state.detach(requires_grad)

    def reset_state(self, state: MemoryState, reset_mask: Tensor) -> MemoryState:
        return self.sequence_dit.reset(state, reset_mask, training=self.training and torch.is_grad_enabled())

    def normalize_actions(self, actions: Tensor, observations) -> tuple[Tensor, Tensor]:
        return self.adapter.normalize_actions(actions, observations)

    def forward_sequence(self, segment, state=None, noisy_actions=None, tau=None) -> SequenceOutput:
        if noisy_actions is None or tau is None:
            raise ValueError("Training forward requires explicit noisy_actions and tau")
        batch_size, sequence_length = noisy_actions.shape[:2]
        if tau.shape != (batch_size, sequence_length):
            raise ValueError("tau must have shape [B,S]")
        state = state or self.init_state(batch_size)
        predictions = []
        create_graph = self.training and torch.is_grad_enabled()
        for step in range(sequence_length):
            valid = segment.valid_mask[:, step].to(self.device).bool()
            reset = segment.reset_mask[:, step].to(self.device).bool() & valid
            if reset.any():
                state = self.reset_state(state, reset)
            if not valid.any():
                predictions.append(torch.zeros_like(noisy_actions[:, step]))
                continue
            observations = [segment.observations[b][step] for b in range(batch_size)]
            example = next(obs for obs, is_valid in zip(observations, valid.tolist()) if is_valid and obs is not None)
            observations = [obs if obs is not None else example for obs in observations]
            prepared = self.adapter.prepare_observations(observations)
            # The same mask must govern loss, attention and memory. Missing
            # tail actions and context-only placeholders are not observations.
            raw_mask = segment.action_valid_mask[:, step].to(self.device).bool()
            padded_mask = torch.zeros(
                batch_size, self.action_horizon, self.action_dim, device=self.device, dtype=torch.bool,
            )
            padded_mask[:, :raw_mask.shape[1], :raw_mask.shape[2]] = raw_mask
            prepared["action_mask"] = prepared["action_mask"] * padded_mask
            velocity, state = self.sequence_dit.forward_step(
                self.adapter, prepared, noisy_actions[:, step], tau[:, step], state,
                valid_mask=valid, update_mask=segment.state_update_mask[:, step].to(self.device).bool(),
                create_graph=create_graph,
            )
            predictions.append(velocity)
        return SequenceOutput(torch.stack(predictions, dim=1), state)

    @staticmethod
    def _observations(observation: dict | list[dict]) -> list[dict]:
        if isinstance(observation, list):
            return observation
        if "samples" in observation:
            return observation["samples"]
        return [observation]

    @torch.no_grad()
    def act(self, observation: dict | list[dict], state: MemoryState | None = None, *, generator=None):
        if self.training:
            raise RuntimeError("Call policy.eval() before online inference")
        observations = self._observations(observation)
        batch_size = len(observations)
        state = state or self.init_state(batch_size)
        if state.batch_size != batch_size:
            raise ValueError("Observation batch and memory batch differ")
        prepared = self.adapter.prepare_observations(observations)
        action = torch.randn(
            batch_size, self.action_horizon, self.action_dim, device=self.device, generator=generator,
        )
        valid = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        candidate = state
        # Every noise level reads the same previous memory. Commit the final
        # candidate only once, after completing the whole action chunk.
        for index in range(self.denoising_steps):
            tau = torch.full((batch_size,), index / self.denoising_steps, device=self.device)
            velocity, candidate = self.sequence_dit.forward_step(
                self.adapter, prepared, action, tau, state,
                valid_mask=valid, update_mask=valid, create_graph=False,
            )
            action = action + velocity.float() / self.denoising_steps
        physical = self.adapter.decode_action(action, observations)
        return physical, candidate.detach(requires_grad=True)

    @torch.no_grad()
    def observe_context(self, observations, state: MemoryState | None = None) -> MemoryState:
        """Write chronological observations without generating or executing actions.

        Context uses zero normalized action placeholders and tau=0. This is an
        explicit implementation choice for unlabelled demo/video observations.
        """
        if self.training:
            raise RuntimeError("Call policy.eval() before consuming online context")
        if isinstance(observations, dict):
            observations = [observations]
        for observation in observations:
            samples = self._observations(observation)
            state = state or self.init_state(len(samples))
            if state.batch_size != len(samples):
                raise ValueError("Context batch and memory batch differ")
            prepared = self.adapter.prepare_observations(samples)
            prepared["action_mask"] = torch.zeros_like(prepared["action_mask"])
            actions = torch.zeros(len(samples), self.action_horizon, self.action_dim, device=self.device)
            valid = torch.ones(len(samples), dtype=torch.bool, device=self.device)
            _, state = self.sequence_dit.forward_step(
                self.adapter, prepared, actions, torch.zeros(len(samples), device=self.device), state,
                valid_mask=valid, update_mask=valid, create_graph=False,
            )
            state = state.detach(requires_grad=True)
        if state is None:
            state = self.init_state(1)
        return state

    def parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "ttt": sum(p.numel() for p in self.sequence_dit.parameters()),
        }
