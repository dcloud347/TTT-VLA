"""Compose the original GR00T DiT blocks with explicit recurrent memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .fast_weights import MemoryState
from .ttt_block import TTTBlock, TTTConfig


class SequenceDiT(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, config: TTTConfig):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.empty(config.num_register_tokens, hidden_size))
        nn.init.normal_(self.register_tokens, std=0.02)
        self.memory_layers = nn.ModuleList(TTTBlock(hidden_size, config) for _ in range(num_layers))

    def init_state(self, batch_size: int, *, training: bool, device=None, session_ids=()) -> MemoryState:
        device = device or self.register_tokens.device
        return MemoryState(
            tuple(layer.fast.init_state(batch_size, training=training, device=device) for layer in self.memory_layers),
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device), tuple(session_ids),
        )

    def reset(self, state: MemoryState, mask: Tensor, *, training: bool) -> MemoryState:
        mask = mask.to(device=state.positions.device, dtype=torch.bool)
        if mask.shape != state.positions.shape:
            raise ValueError("reset_mask must have shape [batch_size]")
        fresh = self.init_state(state.batch_size, training=training, device=state.positions.device)
        return MemoryState(
            tuple(old.select(mask, new) for old, new in zip(state.layers, fresh.layers)),
            torch.where(mask, 0, state.positions), torch.where(mask, 0, state.commits), state.session_ids,
        )

    def forward_step(
        self, adapter, prepared: dict, noisy_actions: Tensor, tau: Tensor, state: MemoryState,
        *, valid_mask: Tensor, update_mask: Tensor, create_graph: bool,
    ) -> tuple[Tensor, MemoryState]:
        payload = adapter.encode_tokens(prepared, noisy_actions, tau, self.register_tokens)
        hidden = payload["tokens"]
        token_mask = payload.get("token_mask")
        if token_mask is None:
            token_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        token_mask = token_mask.to(hidden.device).bool() & valid_mask[:, None]
        update_mask = update_mask & valid_mask
        if len(state.layers) != len(self.memory_layers):
            raise ValueError("Memory layer count does not match the loaded backbone")
        next_layers = []
        for index, (layer, memory) in enumerate(zip(self.memory_layers, state.layers)):
            hidden = adapter.run_layer(index, payload, hidden)
            hidden, memory = layer(
                hidden, memory, state.positions, token_mask=token_mask,
                update_mask=update_mask, create_graph=create_graph,
            )
            next_layers.append(memory)
        velocity = adapter.decode_actions(hidden, payload)
        next_state = MemoryState(
            tuple(next_layers), state.positions + valid_mask.long() * hidden.shape[1],
            state.commits + update_mask.long(), state.session_ids,
        )
        return velocity, next_state

