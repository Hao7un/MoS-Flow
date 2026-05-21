"""
Context-conditioned gate over flow-matching source distributions.

Mixes K candidate sources (noise / vision / history) via softmax gating
driven by [obs_latents ; history_latents]. No text modality.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceGate(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_sources: int,
        gate_hidden_dim: int = 256,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.temperature = temperature

        # context = [obs_latents ; history_latents]  -> 2*latent_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, num_sources),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs_latents: torch.Tensor, history_latents: torch.Tensor) -> torch.Tensor:
        """
        Returns per-sample gate probabilities of shape (B, num_sources).
        """
        ctx = torch.cat([obs_latents, history_latents], dim=-1)
        logits = self.net(ctx) / self.temperature
        return F.softmax(logits, dim=-1)


def mix_sources(gates: torch.Tensor, sources: list[torch.Tensor]) -> torch.Tensor:
    """
    gates:   (B, K)
    sources: list of K tensors each (B, latent_dim)
    returns: (B, latent_dim)
    """
    stacked = torch.stack(sources, dim=1)  # (B, K, latent_dim)
    return (gates.unsqueeze(-1) * stacked).sum(dim=1)


def load_balance_loss(gates: torch.Tensor) -> torch.Tensor:
    """
    Penalise deviation of batch-mean gate usage from uniform.
    L = sum_i (P_i - 1/K)^2
    """
    K = gates.shape[-1]
    mean_usage = gates.mean(dim=0)  # (K,)
    target = torch.full_like(mean_usage, 1.0 / K)
    return ((mean_usage - target) ** 2).sum()


def gate_entropy(gates: torch.Tensor) -> torch.Tensor:
    """
    Mean per-sample entropy of the gate distribution.
    Used only as a monitoring metric (and optional regulariser).
    """
    p = gates.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1).mean()


class StochasticGate(nn.Module):
    """Sigmoid gate controlling the *randomness ratio* α used in

        x_0 = α ⊙ noise  +  (1 − α) ⊙ mix(source_gate, informative_sources)

    Output shape is controlled by ``output_dim``:
      - ``output_dim=1``          → (B, 1) scalar α per sample.
      - ``output_dim=latent_dim`` → (B, latent_dim) per-sample **per-dimension** α.
        Each latent axis gets its own noise ratio, so semantic directions can
        be independently stochastic or deterministic.
      - Any other value broadcasts elementwise along the last dim.

    The gate is context-conditioned on [obs_latents ; history_latents], same
    input as SourceGate — so α is task-dependent (e.g. deterministic for a
    simple reach, noisy for a multi-modal grasp).
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int = 1,
        gate_hidden_dim: int = 256,
        dropout: float = 0.0,
        init_logit: float = 0.0,  # sigmoid(0) = 0.5 at initialisation
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, self.output_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Bias the final logit so initial α = sigmoid(init_logit).
        # 0.0 → 0.5 (balanced); negative → deterministic-biased; positive → noisy-biased.
        self.net[-1].bias.data.fill_(init_logit)

    def forward(self, obs_latents: torch.Tensor, history_latents: torch.Tensor) -> torch.Tensor:
        """Returns α of shape (B, output_dim) in (0, 1)."""
        ctx = torch.cat([obs_latents, history_latents], dim=-1)
        return torch.sigmoid(self.net(ctx))
