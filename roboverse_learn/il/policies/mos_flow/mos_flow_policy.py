"""
MoS-Flow: Mixture-of-Sources Flow Matching Policy.

Generalises the DP -> VITA -> A2A evolution chain by letting the policy
learn, per-sample, how to mix candidate source distributions. The active
configuration (the "main" MoS-Flow used in the paper) is hierarchical
gating with per-dim α:

    x_0 = α(c) ⊙ noise  +  (1 − α(c)) ⊙ mix(g_src(c), {vision, history})
    global_cond = (1 − g_vision(c)) ⊙ obs_latents

where
- α: StochasticGate, sigmoid, **per-dim** in (0,1)^D — each latent axis
  picks its own noise ratio. Provides anisotropic implicit Jacobian
  regularisation (Camuto et al. 2020 generalised to per-direction).
- g_src: softmax over informative sources {vision, history}.
- The (1 − g_vision) attenuation prevents double-dipping when vision is
  already fed through x_0.

Ablation switches preserved (used in the paper's ablation table):
- hierarchical_gate=False        → flat softmax over all active_sources
- per_dim_alpha=False            → scalar α
- active_sources=[history]       → "no vision" ablation (E6)
- active_sources=[vision]        → "no history" ablation (E5)

All non-MoS hyperparameters mirror policy_config/a2a.yaml so differences
attribute to the mixture-source design, not capacity / schedule.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from roboverse_learn.il.utils.normalizer import LinearNormalizer
from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.policies.base_image_policy import BaseImagePolicy

from roboverse_learn.il.utils.models.flow_net import SimpleFlowNet
from roboverse_learn.il.policies.a2a.action_ae import CNNActionEncoder, SimpleActionDecoder
from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
from roboverse_learn.il.utils.flow.flow_matchers import TorchFlowMatcher
from roboverse_learn.il.policies.mos_flow.source_gate import (
    SourceGate, StochasticGate,
    mix_sources, load_balance_loss, gate_entropy,
)


# Order matters: indices are logged as gate_usage/{name}.
SOURCE_NAMES = ("noise", "vision", "history")


class MoSFlowImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        flow_net,
        flow_matcher: TorchFlowMatcher,
        decode_flow_latents=True,
        consistency_weight=1.0,
        enc_contrastive_weight=0.0,
        flow_contrastive_weight=0.0,
        latent_dim=512,
        action_ae=None,
        history_noise_std=0.0,
        # MoS-specific
        gate_hidden_dim: int = 256,
        gate_temperature: float = 1.0,
        gate_dropout: float = 0.0,
        load_balance_weight: float = 1e-2,
        entropy_weight: float = 0.0,           # constant weight (used when warmup_steps <= 0)
        entropy_weight_start: float = 0.0,     # starting weight for linear annealing
        entropy_weight_end: float = 0.0,       # final weight for linear annealing
        entropy_warmup_steps: int = 0,         # 0 -> use constant entropy_weight (back-compat)
        active_sources: tuple = SOURCE_NAMES,
        # Hierarchical gating (handles noise-vs-informative dichotomy):
        #   x_0 = α ⊙ noise  +  (1 − α) ⊙ mix(source_gate, informative_sources)
        # α comes from StochasticGate; source_gate softmax covers only the
        # informative sources (vision/history). Any "noise" entry in
        # active_sources is dropped automatically — noise enters only via α.
        hierarchical_gate: bool = True,
        # Per-dimension α (output_dim = latent_dim). Each latent axis picks its
        # own noise ratio, providing anisotropic implicit Jacobian regularisation.
        # Scalar α (False) is kept for ablation; per-dim is the paper's default.
        per_dim_alpha: bool = True,
        alpha_init_logit: float = 0.0,         # sigmoid(0) = 0.5 → balanced start
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.decode_flow_latents = decode_flow_latents
        self.consistency_weight = consistency_weight
        self.enc_contrastive_weight = enc_contrastive_weight
        self.flow_contrastive_weight = flow_contrastive_weight
        self.latent_dim = latent_dim
        self.num_sampling_steps = flow_matcher.num_sampling_steps
        self.history_noise_std = history_noise_std
        self.load_balance_weight = load_balance_weight
        self.entropy_weight = entropy_weight
        self.entropy_weight_start = entropy_weight_start
        self.entropy_weight_end = entropy_weight_end
        self.entropy_warmup_steps = int(entropy_warmup_steps)
        # Training step counter for annealing schedules. Persisted in checkpoints.
        self.register_buffer("_train_step", torch.zeros(1, dtype=torch.long))

        # Validate and register active sources
        for s in active_sources:
            assert s in SOURCE_NAMES, f"unknown source '{s}', allowed={SOURCE_NAMES}"
        active_sources = tuple(active_sources)

        # Hierarchical gating: noise is handled separately by α, must NOT be in
        # the softmax budget. Drop it automatically (with a warning-ish assert
        # if it was explicitly present).
        self.hierarchical_gate = bool(hierarchical_gate)
        if self.hierarchical_gate and "noise" in active_sources:
            active_sources = tuple(s for s in active_sources if s != "noise")
        if self.hierarchical_gate:
            assert len(active_sources) >= 1, (
                "hierarchical_gate=True requires ≥1 informative source "
                "(e.g. vision or history); noise is handled by the α gate."
            )

        self.active_sources = active_sources
        self.num_sources = len(self.active_sources)
        assert self.num_sources >= 1
        self._vision_gate_idx = (
            self.active_sources.index("vision") if "vision" in self.active_sources else None
        )

        self.per_dim_alpha = bool(per_dim_alpha)
        # per_dim_alpha only makes sense when the α path exists. Enforce
        # rather than silently ignore.
        if self.per_dim_alpha and not self.hierarchical_gate:
            raise ValueError(
                "per_dim_alpha requires hierarchical_gate=True "
                "(it operates on the α-gated noise path)."
            )

        self.flow_matcher = flow_matcher
        self.action_ae = action_ae

        # Visual observation encoder (same as A2A/VITA)
        self.obs_encoder = obs_encoder
        obs_flat_dim = obs_feature_dim * n_obs_steps
        self.obs_projector = nn.Linear(obs_flat_dim, latent_dim)

        # Flow network with obs_latents as global_cond (same wiring as A2A)
        self.flow_net = SimpleFlowNet(
            input_dim=latent_dim,
            hidden_dim=flow_net.hidden_dim,
            output_dim=latent_dim,
            num_layers=flow_net.num_layers,
            mlp_ratio=flow_net.mlp_ratio,
            dropout=flow_net.dropout,
            condition_dim=latent_dim,
        )

        # History state encoder (same as A2A)
        self.history_action_encoder = CNNActionEncoder(
            pred_horizon=n_obs_steps,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )

        future_horizon = n_action_steps
        self.future_horizon = future_horizon

        self.action_encoder = CNNActionEncoder(
            pred_horizon=future_horizon,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=action_ae.net.dec_hidden_dim,
            latent_dim=latent_dim,
            pred_horizon=future_horizon,
            action_dim=action_dim,
            num_layers=action_ae.net.num_layers,
            dropout=action_ae.net.dropout,
        )

        self.source_gate = SourceGate(
            latent_dim=latent_dim,
            num_sources=self.num_sources,
            gate_hidden_dim=gate_hidden_dim,
            temperature=gate_temperature,
            dropout=gate_dropout,
        )

        # Hierarchical gating: α controls the noise ratio. Either scalar (B, 1)
        # or per-dim (B, latent_dim) — per-dim lets each semantic axis set its
        # own stochasticity level (paper's default).
        if self.hierarchical_gate:
            self.stochastic_gate = StochasticGate(
                latent_dim=latent_dim,
                output_dim=(latent_dim if self.per_dim_alpha else 1),
                gate_hidden_dim=gate_hidden_dim,
                dropout=gate_dropout,
                init_logit=alpha_init_logit,
            )
        else:
            self.stochastic_gate = None

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

    # ------------------------------------------------------------------ utils

    def _add_history_noise(self, history_states: torch.Tensor) -> torch.Tensor:
        if self.history_noise_std > 0:
            return history_states + torch.randn_like(history_states) * self.history_noise_std
        return history_states

    def _encode_context(self, nobs: Dict[str, torch.Tensor], batch_size: int):
        """
        Encode visual obs + history. Returns (obs_latents, history_latents).
        obs_latents feeds both the vision source in x_0 and (after attenuation
        by 1 − g_vision in _effective_cond) the flow net's global_cond.
        """
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, -1)
        obs_latents = self.obs_projector(nobs_features)

        history_states = nobs["agent_pos"][:, :self.n_obs_steps, :]
        history_states = self._add_history_noise(history_states)
        history_latents = self.history_action_encoder(history_states)
        return obs_latents, history_latents

    def _build_sources(
        self,
        obs_latents: torch.Tensor,
        history_latents: torch.Tensor,
        target_latents_like: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Assemble the list of candidate source tensors matching self.active_sources."""
        sources = []
        for name in self.active_sources:
            if name == "noise":
                sources.append(torch.randn_like(target_latents_like))
            elif name == "vision":
                sources.append(obs_latents)
            elif name == "history":
                sources.append(history_latents)
            else:  # pragma: no cover
                raise ValueError(f"unknown source {name}")
        return sources

    def _effective_cond(self, obs_latents: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """
        Attenuate the global_cond by (1 − g_vision) to prevent double-dipping
        when the gate routes to the vision source. g_vision=1 recovers VITA
        (source-only); g_vision=0 recovers full A2A-style conditioning; the
        gate smoothly interpolates between the two.

        No-op if vision isn't in active_sources (nothing to double-dip on).
        """
        if self._vision_gate_idx is None:
            return obs_latents
        g_vision = gates[:, self._vision_gate_idx:self._vision_gate_idx + 1]  # (B, 1)
        return obs_latents * (1.0 - g_vision)

    def _current_entropy_weight(self) -> float:
        """Linear annealing from entropy_weight_start to entropy_weight_end
        over entropy_warmup_steps training steps.

        If entropy_warmup_steps <= 0, falls back to the constant entropy_weight
        (backward compatible with the original interface).
        """
        if self.entropy_warmup_steps <= 0:
            return self.entropy_weight
        step = int(self._train_step.item())
        alpha = min(1.0, step / float(self.entropy_warmup_steps))
        return (1.0 - alpha) * self.entropy_weight_start + alpha * self.entropy_weight_end

    # -------------------------------------------------------------- train/eval

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        obs_latents, history_latents = self._encode_context(nobs, batch_size)

        # Future actions (same slicing as A2A)
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        future_action_latents = self.action_encoder(future_actions)

        # Gate is conditioned on the shared obs projection + history.
        gates = self.source_gate(obs_latents, history_latents)  # (B, K_info)
        sources = self._build_sources(obs_latents, history_latents, future_action_latents)
        info_mix = mix_sources(gates, sources)

        # Hierarchical: blend N(0, I) noise with the informative mix via α.
        # α is (B, 1) for scalar mode, (B, latent_dim) for per-dim — broadcasting
        # handles both uniformly.
        if self.stochastic_gate is not None:
            alpha = self.stochastic_gate(obs_latents, history_latents)
            noise = torch.randn_like(future_action_latents)
            mixed_start = alpha * noise + (1.0 - alpha) * info_mix
        else:
            alpha = None
            mixed_start = info_mix

        effective_cond = self._effective_cond(obs_latents, gates)

        # Flow matching loss
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_action_latents,
            start=mixed_start,
            global_cond=effective_cond,
        )
        loss = flow_loss
        metrics["flow_loss"] = flow_loss.item()

        # Gating regularisers + monitoring metrics
        lb = load_balance_loss(gates)
        metrics["load_balance_loss"] = lb.item()
        if self.load_balance_weight > 0:
            loss = loss + self.load_balance_weight * lb

        ent = gate_entropy(gates)
        metrics["gate_entropy"] = ent.item()
        # Effective entropy weight (annealed if warmup_steps > 0, constant otherwise).
        # Positive weight -> encourage high entropy (exploration, prevents gate collapse).
        # Negative weight -> encourage low entropy (specialisation).
        entropy_w = self._current_entropy_weight()
        metrics["effective_entropy_weight"] = entropy_w
        if entropy_w != 0.0:
            loss = loss - entropy_w * ent

        with torch.no_grad():
            mean_usage = gates.mean(dim=0)
            for i, name in enumerate(self.active_sources):
                metrics[f"gate_usage/{name}"] = mean_usage[i].item()

        # Hierarchical α metrics.
        if alpha is not None:
            # Works for both scalar (B, 1) and per-dim (B, latent_dim) α.
            metrics["alpha/mean"] = alpha.mean().item()
            metrics["alpha/std"] = alpha.std().item()
            metrics["alpha/min"] = alpha.min().item()
            metrics["alpha/max"] = alpha.max().item()
            if self.per_dim_alpha:
                # How spread the per-sample α is across latent dims on average —
                # a small value means the gate is effectively scalar even with
                # capacity to diverge, a large value means dims are specialising.
                metrics["alpha/per_sample_dim_std"] = alpha.std(dim=-1).mean().item()

        # Encoder contrastive loss (disabled by default, kept for parity with A2A)
        if self.enc_contrastive_weight > 0:
            image_features = obs_latents.view(batch_size, -1)
            action_features = future_action_latents.view(batch_size, -1)
            contrastive = self._compute_contrastive_loss(image_features, action_features)
            loss = loss + self.enc_contrastive_weight * contrastive
            metrics["enc_contrastive_loss"] = contrastive.item()

        # Flow latent decoding block (mirror A2A)
        if self.decode_flow_latents:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=obs_latents.device,
                start=mixed_start,
                num_steps=self.num_sampling_steps,
                global_cond=effective_cond,
            )
            if self.consistency_weight > 0:
                consistency = F.mse_loss(action_latents_pred, future_action_latents)
                loss = loss + self.consistency_weight * consistency
                metrics["consistency_loss"] = consistency.item()

            if self.flow_contrastive_weight > 0:
                image_features = obs_latents.view(batch_size, -1)
                action_features = action_latents_pred.view(batch_size, -1)
                contrastive = self._compute_contrastive_loss(image_features, action_features)
                loss = loss + self.flow_contrastive_weight * contrastive
                metrics["flow_contrastive_loss"] = contrastive.item()

            if self.action_ae["flow_recon_weight"] > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                recon = F.l1_loss(actions_recon, future_actions)
                loss = loss + self.action_ae["flow_recon_weight"] * recon
                metrics["flow_action_recon_loss"] = recon.item()

        if self.action_ae["enc_recon_weight"] > 0:
            actions_recon = self.action_decoder(future_action_latents)
            recon = F.l1_loss(actions_recon, future_actions)
            loss = loss + self.action_ae["enc_recon_weight"] * recon
            metrics["enc_action_recon_loss"] = recon.item()

        # Advance training step (used by entropy annealing schedule).
        # Increment once per compute_loss call — close enough to "optimizer step"
        # for schedule purposes; gradient accumulation would over-count slightly
        # but the warmup window is long enough that it doesn't matter.
        self._train_step += 1

        # Stash for the runner: it auto-logs every entry as `train_{k}` to wandb
        # (see default_runner.py). Gives us gate_usage/noise|vision|history,
        # gate_entropy, load_balance_loss, effective_entropy_weight curves.
        self._last_metrics = metrics

        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        obs_latents, history_latents = self._encode_context(nobs, B)

        gates = self.source_gate(obs_latents, history_latents)
        # Noise source is drawn fresh at inference (same distribution as training).
        noise_like = torch.randn(B, self.latent_dim, device=obs_latents.device)
        sources = self._build_sources(obs_latents, history_latents, noise_like)
        info_mix = mix_sources(gates, sources)

        if self.stochastic_gate is not None:
            alpha = self.stochastic_gate(obs_latents, history_latents)
            noise = torch.randn_like(info_mix)
            mixed_start = alpha * noise + (1.0 - alpha) * info_mix
        else:
            mixed_start = info_mix

        effective_cond = self._effective_cond(obs_latents, gates)

        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(B, self.latent_dim),
            device=obs_latents.device,
            num_steps=self.num_sampling_steps,
            start=mixed_start,
            global_cond=effective_cond,
            return_traces=False,
        )

        with torch.no_grad():
            action_pred = self.action_decoder(action_latents_pred)

        action_pred = self.normalizer["action"].unnormalize(action_pred)
        action = action_pred[:, :self.n_action_steps]
        return {"action": action, "action_pred": action_pred, "gate": gates.detach()}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @torch.no_grad()
    def get_gate_visualizations(self, batch, max_samples: int = 128) -> dict:
        """
        Produce 2 wandb-loggable gate figures:
          - gate/heatmap         : (samples x sources) per-sample probability map
          - gate/dominant_hist   : bar chart of argmax(gate) source usage over the batch

        Returned dict is `{name: wandb.Image}`. The runner can log it directly.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import wandb

        nobs = self.normalizer.normalize(batch["obs"])
        B = next(iter(nobs.values())).shape[0]
        B = min(B, max_samples)

        # Slice the batch down to B samples to keep viz cheap
        sliced = {k: v[:B] for k, v in nobs.items()}
        obs_lat, hist = self._encode_context(sliced, B)
        gates = self.source_gate(obs_lat, hist).cpu().numpy()  # (B, K)

        figures = {}

        # Figure 1: per-sample gate heatmap
        fig, ax = plt.subplots(figsize=(3.5, max(2.0, B * 0.04)))
        im = ax.imshow(gates, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(self.active_sources)))
        ax.set_xticklabels(self.active_sources, rotation=0)
        ax.set_xlabel("source")
        ax.set_ylabel("val sample")
        ax.set_title(f"gate probs (B={B})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        figures["gate/heatmap"] = wandb.Image(fig)
        plt.close(fig)

        # Figure 2: dominant-source bar chart + mean-usage overlay
        dominant = np.argmax(gates, axis=1)
        counts = np.bincount(dominant, minlength=len(self.active_sources)) / B
        mean_usage = gates.mean(axis=0)

        fig, ax = plt.subplots(figsize=(4, 3))
        x = np.arange(len(self.active_sources))
        ax.bar(x - 0.2, counts, width=0.4, label="argmax share", color="#4C78A8")
        ax.bar(x + 0.2, mean_usage, width=0.4, label="mean prob", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(self.active_sources)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fraction")
        ax.set_title(f"source usage (B={B})")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        figures["gate/dominant_hist"] = wandb.Image(fig)
        plt.close(fig)

        return figures

    @staticmethod
    def _compute_contrastive_loss(image_features, action_features, temperature=0.07):
        batch_size = image_features.size(0)
        image_features = F.normalize(image_features, dim=1)
        action_features = F.normalize(action_features, dim=1)
        logits = torch.matmul(image_features, action_features.T) / temperature
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.T, labels)
        return (loss_i2a + loss_a2i) / 2
