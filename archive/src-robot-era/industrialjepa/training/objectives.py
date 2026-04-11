# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Training Objectives for Industrial World Model.

Implements multiple self-supervised and supervised objectives:
1. MTP: Masked Token Prediction (BERT-style)
2. NSP: Next State Prediction (GPT-style)
3. Contrastive: Domain-aware contrastive learning
4. Reconstruction: VQ-VAE reconstruction loss
5. World Model: Combined dynamics + stochastic objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LossOutput:
    """Container for loss values and components."""

    total: torch.Tensor
    components: Dict[str, torch.Tensor]
    metrics: Dict[str, float]


class MaskedTokenPrediction(nn.Module):
    """
    Masked Token Prediction (MTP) objective.

    Similar to BERT's MLM, but for discrete time series tokens.
    Masks random tokens and predicts them from context.
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 0,
        random_token_prob: float = 0.1,
        keep_token_prob: float = 0.1,
        codebook_size: int = 8192,
    ):
        """
        Args:
            mask_prob: Probability of masking each token
            mask_token_id: Token ID to use for masking
            random_token_prob: Prob of replacing with random token
            keep_token_prob: Prob of keeping original token
            codebook_size: Size of token vocabulary
        """
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.random_token_prob = random_token_prob
        self.keep_token_prob = keep_token_prob
        self.codebook_size = codebook_size

    def create_mask(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masked tokens for MTP.

        Args:
            tokens: [B, T] original token indices

        Returns:
            masked_tokens: [B, T] tokens with some masked
            mask: [B, T] boolean mask (True = masked)
            labels: [B, T] original tokens (-100 for non-masked)
        """
        B, T = tokens.shape
        device = tokens.device

        # Decide which tokens to mask
        mask = torch.rand(B, T, device=device) < self.mask_prob

        # Create labels (-100 for non-masked positions)
        labels = tokens.clone()
        labels[~mask] = -100

        # Create masked tokens
        masked_tokens = tokens.clone()

        # Probabilities for mask/random/keep
        probs = torch.rand(B, T, device=device)

        # 80% of masked: replace with [MASK]
        mask_mask = mask & (probs < (1 - self.random_token_prob - self.keep_token_prob))
        masked_tokens[mask_mask] = self.mask_token_id

        # 10% of masked: replace with random token
        random_mask = mask & (probs >= (1 - self.random_token_prob - self.keep_token_prob)) & (
            probs < (1 - self.keep_token_prob)
        )
        random_tokens = torch.randint(
            0, self.codebook_size, (random_mask.sum().item(),), device=device
        )
        masked_tokens[random_mask] = random_tokens

        # 10% of masked: keep original (implicit, already done)

        return masked_tokens, mask, labels

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> LossOutput:
        """
        Compute MTP loss.

        Args:
            predictions: [B, T, codebook_size] logits
            labels: [B, T] target token indices
            mask: [B, T] boolean mask

        Returns:
            LossOutput with MTP loss
        """
        # Flatten for cross entropy
        B, T, V = predictions.shape

        # Only compute loss on masked positions
        predictions_flat = predictions.view(-1, V)
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(
            predictions_flat,
            labels_flat,
            ignore_index=-100,
            reduction="mean",
        )

        # Compute accuracy on masked positions
        with torch.no_grad():
            pred_tokens = predictions.argmax(dim=-1)  # [B, T]
            correct = (pred_tokens == labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()

        return LossOutput(
            total=loss,
            components={"mtp": loss},
            metrics={"mtp_accuracy": accuracy.item()},
        )


class NextStatePrediction(nn.Module):
    """
    Next State Prediction (NSP) objective.

    Autoregressive prediction of next token/state.
    Core world model training signal.
    """

    def __init__(
        self,
        prediction_type: str = "token",  # "token" or "continuous"
        horizon: int = 1,
        use_quantile_loss: bool = True,
        num_quantiles: int = 9,
    ):
        """
        Args:
            prediction_type: Whether to predict tokens or continuous values
            horizon: Number of steps to predict
            use_quantile_loss: Whether to use quantile regression
            num_quantiles: Number of quantiles for uncertainty
        """
        super().__init__()
        self.prediction_type = prediction_type
        self.horizon = horizon
        self.use_quantile_loss = use_quantile_loss
        self.num_quantiles = num_quantiles

        if use_quantile_loss:
            quantiles = torch.linspace(0.1, 0.9, num_quantiles)
            self.register_buffer("quantiles", quantiles)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantile_predictions: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute NSP loss.

        Args:
            predictions: [B, T, D] or [B, T, V] predicted states/logits
            targets: [B, T, D] or [B, T] target states/tokens
            quantile_predictions: [B, T, D, Q] quantile predictions

        Returns:
            LossOutput with NSP loss
        """
        components = {}
        metrics = {}

        if self.prediction_type == "token":
            # Cross entropy loss
            B, T, V = predictions.shape
            predictions_flat = predictions[:, :-1].reshape(-1, V)
            targets_flat = targets[:, 1:].reshape(-1)

            nsp_loss = F.cross_entropy(predictions_flat, targets_flat)
            components["nsp_ce"] = nsp_loss

            # Accuracy
            with torch.no_grad():
                pred_tokens = predictions[:, :-1].argmax(dim=-1)
                correct = (pred_tokens == targets[:, 1:]).float()
                metrics["nsp_accuracy"] = correct.mean().item()

        else:
            # MSE loss for continuous predictions
            # Shift for next-step prediction
            preds = predictions[:, :-1]
            tgts = targets[:, 1:]

            nsp_loss = F.mse_loss(preds, tgts)
            components["nsp_mse"] = nsp_loss

            # RMSE metric
            with torch.no_grad():
                rmse = torch.sqrt(nsp_loss)
                metrics["nsp_rmse"] = rmse.item()

        # Quantile loss for uncertainty
        if self.use_quantile_loss and quantile_predictions is not None:
            q_loss = self._quantile_loss(quantile_predictions, targets)
            components["nsp_quantile"] = q_loss

        total = sum(components.values())

        return LossOutput(
            total=total,
            components=components,
            metrics=metrics,
        )

    def _quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quantile regression loss.

        Args:
            predictions: [B, T, D, Q] quantile predictions
            targets: [B, T, D] targets

        Returns:
            Scalar quantile loss
        """
        # Shift for next-step prediction
        preds = predictions[:, :-1]  # [B, T-1, D, Q]
        tgts = targets[:, 1:].unsqueeze(-1)  # [B, T-1, D, 1]

        errors = tgts - preds  # [B, T-1, D, Q]

        # Pinball loss
        loss = torch.max(
            self.quantiles * errors,
            (self.quantiles - 1) * errors,
        )

        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Domain-aware Contrastive Learning.

    Learns representations that:
    - Pull together same-domain samples
    - Push apart different-domain samples
    - Capture temporal structure
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_domain_labels: bool = True,
        projection_dim: int = 128,
        hidden_dim: int = 512,
    ):
        """
        Args:
            temperature: Contrastive temperature
            use_domain_labels: Whether to use domain as positive
            projection_dim: Dimension of projection head output
            hidden_dim: Hidden dimension for projection head
        """
        super().__init__()
        self.temperature = temperature
        self.use_domain_labels = use_domain_labels

        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        augmented_hidden: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute contrastive loss.

        Args:
            hidden_states: [B, T, D] or [B, D] hidden states
            domain_labels: [B] domain indices
            augmented_hidden: [B, T, D] augmented view (if available)

        Returns:
            LossOutput with contrastive loss
        """
        # Pool if sequence
        if hidden_states.dim() == 3:
            h = hidden_states.mean(dim=1)  # [B, D]
        else:
            h = hidden_states

        # Project
        z = F.normalize(self.projector(h), dim=-1)  # [B, projection_dim]

        B = z.shape[0]

        if augmented_hidden is not None:
            # SimCLR-style: positive pairs are augmented views
            if augmented_hidden.dim() == 3:
                h_aug = augmented_hidden.mean(dim=1)
            else:
                h_aug = augmented_hidden

            z_aug = F.normalize(self.projector(h_aug), dim=-1)

            # Concatenate for 2B samples
            z_all = torch.cat([z, z_aug], dim=0)  # [2B, D]

            # Compute similarities
            sim = torch.mm(z_all, z_all.t()) / self.temperature  # [2B, 2B]

            # Mask self-similarities
            mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
            sim.masked_fill_(mask, float("-inf"))

            # Positive pairs: (i, i+B) and (i+B, i)
            labels = torch.cat([
                torch.arange(B, 2 * B, device=z.device),
                torch.arange(B, device=z.device),
            ])

            loss = F.cross_entropy(sim, labels)

        elif domain_labels is not None and self.use_domain_labels:
            # Domain-based contrastive: same domain = positive
            sim = torch.mm(z, z.t()) / self.temperature  # [B, B]

            # Create label matrix: same domain = 1
            domain_eq = domain_labels.unsqueeze(0) == domain_labels.unsqueeze(1)
            domain_eq.fill_diagonal_(False)  # Exclude self

            # Supervised contrastive loss
            exp_sim = torch.exp(sim)
            exp_sim.fill_diagonal_(0)

            # For each sample, sum over positives / sum over all
            pos_sum = (exp_sim * domain_eq.float()).sum(dim=1)
            all_sum = exp_sim.sum(dim=1)

            # Avoid log(0)
            loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8).mean()

        else:
            # Default: InfoNCE within batch
            sim = torch.mm(z, z.t()) / self.temperature
            labels = torch.arange(B, device=z.device)
            loss = F.cross_entropy(sim, labels)

        return LossOutput(
            total=loss,
            components={"contrastive": loss},
            metrics={},
        )


class ReconstructionLoss(nn.Module):
    """
    VQ-VAE Reconstruction Loss.

    Combines:
    - Reconstruction loss (MSE or spectral)
    - VQ commitment loss
    - Codebook utilization regularization
    """

    def __init__(
        self,
        use_spectral_loss: bool = True,
        commitment_weight: float = 0.25,
        entropy_weight: float = 0.1,
        fft_weight: float = 0.1,
    ):
        """
        Args:
            use_spectral_loss: Whether to add frequency domain loss
            commitment_weight: Weight for VQ commitment loss
            entropy_weight: Weight for codebook entropy regularization
            fft_weight: Weight for FFT loss
        """
        super().__init__()
        self.use_spectral_loss = use_spectral_loss
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight
        self.fft_weight = fft_weight

    def forward(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        vq_loss: torch.Tensor,
        codebook_entropy: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute reconstruction loss.

        Args:
            reconstruction: [B, C, L] reconstructed signal
            original: [B, C, L] original signal
            vq_loss: Scalar VQ commitment loss
            codebook_entropy: Scalar entropy of codebook usage

        Returns:
            LossOutput with reconstruction losses
        """
        components = {}
        metrics = {}

        # Time domain reconstruction
        recon_loss = F.mse_loss(reconstruction, original)
        components["recon_mse"] = recon_loss

        # Spectral loss (frequency domain)
        if self.use_spectral_loss:
            fft_loss = self._spectral_loss(reconstruction, original)
            components["recon_fft"] = self.fft_weight * fft_loss

        # VQ commitment loss
        components["vq_commit"] = self.commitment_weight * vq_loss

        # Codebook entropy regularization (encourage usage)
        if codebook_entropy is not None:
            # Negative entropy = penalty for low utilization
            entropy_loss = -self.entropy_weight * codebook_entropy
            components["codebook_entropy"] = entropy_loss
            metrics["codebook_entropy"] = codebook_entropy.item()

        total = sum(components.values())

        # Metrics
        with torch.no_grad():
            metrics["recon_rmse"] = torch.sqrt(recon_loss).item()

        return LossOutput(
            total=total,
            components=components,
            metrics=metrics,
        )

    def _spectral_loss(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss in frequency domain."""
        # FFT
        recon_fft = torch.fft.rfft(reconstruction, dim=-1)
        orig_fft = torch.fft.rfft(original, dim=-1)

        # Magnitude spectrum loss
        recon_mag = torch.abs(recon_fft)
        orig_mag = torch.abs(orig_fft)

        # Log magnitude for better gradient flow
        recon_log = torch.log(recon_mag + 1e-8)
        orig_log = torch.log(orig_mag + 1e-8)

        return F.mse_loss(recon_log, orig_log)


class WorldModelLoss(nn.Module):
    """
    Complete World Model Training Objective.

    Combines all losses for end-to-end training:
    - Dynamics prediction (NSP)
    - Reconstruction (VQ-VAE)
    - KL divergence (stochastic latent)
    - Contrastive (domain awareness)
    - Optional: MTP for pretraining
    """

    def __init__(
        self,
        # Loss weights
        dynamics_weight: float = 1.0,
        recon_weight: float = 1.0,
        kl_weight: float = 0.1,
        contrastive_weight: float = 0.1,
        mtp_weight: float = 0.0,  # 0 = disabled
        # Loss configs
        use_quantile: bool = True,
        use_spectral: bool = True,
        contrastive_temp: float = 0.07,
    ):
        """
        Args:
            dynamics_weight: Weight for dynamics prediction loss
            recon_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence
            contrastive_weight: Weight for contrastive loss
            mtp_weight: Weight for MTP (0 to disable)
            use_quantile: Use quantile regression
            use_spectral: Use spectral reconstruction loss
            contrastive_temp: Contrastive temperature
        """
        super().__init__()

        self.weights = {
            "dynamics": dynamics_weight,
            "recon": recon_weight,
            "kl": kl_weight,
            "contrastive": contrastive_weight,
            "mtp": mtp_weight,
        }

        # Initialize sub-losses
        self.nsp = NextStatePrediction(
            prediction_type="continuous",
            use_quantile_loss=use_quantile,
        )

        self.recon = ReconstructionLoss(
            use_spectral_loss=use_spectral,
        )

        self.contrastive = ContrastiveLoss(
            temperature=contrastive_temp,
        )

        if mtp_weight > 0:
            self.mtp = MaskedTokenPrediction()
        else:
            self.mtp = None

    def forward(
        self,
        model_output: "IndustrialWorldLMOutput",
        targets: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        mtp_predictions: Optional[torch.Tensor] = None,
        mtp_labels: Optional[torch.Tensor] = None,
        mtp_mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute combined world model loss.

        Args:
            model_output: Output from IndustrialWorldLM
            targets: [B, C, L] target time series
            domain_labels: [B] domain indices
            mtp_predictions: [B, T, V] MTP logits (if using MTP)
            mtp_labels: [B, T] MTP labels
            mtp_mask: [B, T] MTP mask

        Returns:
            LossOutput with all loss components
        """
        components = {}
        metrics = {}

        # 1. Dynamics prediction loss
        if model_output.next_state_pred is not None:
            nsp_output = self.nsp(
                model_output.next_state_pred,
                model_output.quantized_features,
                model_output.quantile_pred,
            )
            components["dynamics"] = self.weights["dynamics"] * nsp_output.total
            metrics.update(nsp_output.metrics)

        # 2. Reconstruction loss
        if model_output.reconstruction is not None:
            vq_loss = model_output.loss_components.get("vq", torch.tensor(0.0))
            recon_output = self.recon(
                model_output.reconstruction,
                targets,
                vq_loss,
            )
            components["recon"] = self.weights["recon"] * recon_output.total
            metrics.update(recon_output.metrics)

        # 3. KL divergence (from stochastic latent)
        if "kl" in model_output.loss_components:
            components["kl"] = self.weights["kl"] * model_output.loss_components["kl"]
            metrics["kl_div"] = model_output.loss_components["kl"].item()

        # 4. Contrastive loss
        if self.weights["contrastive"] > 0 and model_output.hidden_states is not None:
            contrast_output = self.contrastive(
                model_output.hidden_states,
                domain_labels,
            )
            components["contrastive"] = self.weights["contrastive"] * contrast_output.total

        # 5. Masked token prediction (optional)
        if self.mtp is not None and mtp_predictions is not None:
            mtp_output = self.mtp(mtp_predictions, mtp_labels, mtp_mask)
            components["mtp"] = self.weights["mtp"] * mtp_output.total
            metrics.update(mtp_output.metrics)

        # Total loss
        total = sum(components.values())

        return LossOutput(
            total=total,
            components=components,
            metrics=metrics,
        )


class RULPredictionLoss(nn.Module):
    """
    Remaining Useful Life (RUL) Prediction Loss.

    Task-specific loss for turbofan/bearing prognostics.
    """

    def __init__(
        self,
        use_asymmetric: bool = True,
        late_penalty: float = 13.0,
        early_penalty: float = 10.0,
    ):
        """
        Args:
            use_asymmetric: Use asymmetric scoring (penalize late more)
            late_penalty: Exponent for late predictions
            early_penalty: Exponent for early predictions
        """
        super().__init__()
        self.use_asymmetric = use_asymmetric
        self.late_penalty = late_penalty
        self.early_penalty = early_penalty

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossOutput:
        """
        Compute RUL prediction loss.

        Args:
            predictions: [B] or [B, T] predicted RUL
            targets: [B] or [B, T] target RUL

        Returns:
            LossOutput with RUL loss
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)

        if self.use_asymmetric:
            # NASA scoring function
            errors = predictions - targets  # Positive = late, negative = early

            # Asymmetric penalty
            late_mask = errors > 0
            scores = torch.zeros_like(errors)
            scores[late_mask] = torch.exp(errors[late_mask] / self.late_penalty) - 1
            scores[~late_mask] = torch.exp(-errors[~late_mask] / self.early_penalty) - 1

            score_loss = scores.mean()
            total = mse_loss + 0.1 * score_loss

            return LossOutput(
                total=total,
                components={"rul_mse": mse_loss, "rul_score": score_loss},
                metrics={
                    "rul_rmse": torch.sqrt(mse_loss).item(),
                    "rul_score": score_loss.item(),
                },
            )

        return LossOutput(
            total=mse_loss,
            components={"rul_mse": mse_loss},
            metrics={"rul_rmse": torch.sqrt(mse_loss).item()},
        )


class JEPALoss(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA) Loss.

    Based on V-JEPA and EchoJEPA, adapted for industrial time series.

    Key insight: Instead of reconstructing raw sensor values (which wastes
    capacity on unpredictable noise), predict in latent embedding space.
    This naturally filters sensor noise, similar to how EchoJEPA filters
    ultrasound speckle artifacts.

    Components:
    1. Context encoder: Processes visible time segments
    2. Predictor: Predicts embeddings of masked segments from context
    3. EMA target encoder: Provides stable prediction targets

    The loss is L1 distance between predicted and target embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        predictor_hidden_dim: int = 384,
        num_predictor_layers: int = 4,
        num_predictor_heads: int = 8,
        mask_ratio: float = 0.6,
        mask_block_size: int = 16,
        ema_momentum: float = 0.999,
        multi_scale_masks: bool = True,
        use_action_conditioning: bool = True,
        action_dim: int = 32,
    ):
        """
        Args:
            hidden_dim: Dimension of encoder outputs
            predictor_hidden_dim: Hidden dimension of predictor network
            num_predictor_layers: Number of transformer layers in predictor
            num_predictor_heads: Number of attention heads in predictor
            mask_ratio: Fraction of sequence to mask
            mask_block_size: Size of mask blocks (tubelet-like masking)
            ema_momentum: Momentum for EMA target encoder
            multi_scale_masks: Use multi-scale masking (like V-JEPA 2)
            use_action_conditioning: Condition predictor on actions
            action_dim: Dimension of action embeddings
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.ema_momentum = ema_momentum
        self.multi_scale_masks = multi_scale_masks
        self.use_action_conditioning = use_action_conditioning

        # Predictor network: lightweight transformer
        self.predictor_embed = nn.Linear(hidden_dim, predictor_hidden_dim)

        # Learnable mask tokens (one per masked position)
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_hidden_dim))

        # Positional encoding for predictor
        max_len = 1024
        self.predictor_pos_embed = nn.Parameter(
            torch.randn(1, max_len, predictor_hidden_dim) * 0.02
        )

        # Predictor transformer layers
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_hidden_dim,
            nhead=num_predictor_heads,
            dim_feedforward=predictor_hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=num_predictor_layers,
        )

        # Project back to target dimension
        self.predictor_proj = nn.Linear(predictor_hidden_dim, hidden_dim)

        # Action conditioning (FiLM-style)
        if use_action_conditioning:
            self.action_gamma = nn.Linear(action_dim, predictor_hidden_dim)
            self.action_beta = nn.Linear(action_dim, predictor_hidden_dim)

        # Layer norm for targets
        self.target_norm = nn.LayerNorm(hidden_dim)

    def create_block_mask(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block-wise masking pattern.

        Args:
            seq_len: Sequence length
            batch_size: Batch size
            device: Device to create mask on

        Returns:
            [B, L] boolean mask (True = masked/to predict)
        """
        if self.multi_scale_masks:
            # Multi-scale masking: combine coarse and fine masks
            # Similar to V-JEPA 2's approach
            mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

            for b in range(batch_size):
                # Coarse mask: 2-4 large blocks
                num_coarse = torch.randint(2, 5, (1,)).item()
                coarse_size = seq_len // 4

                for _ in range(num_coarse):
                    start = torch.randint(0, seq_len - coarse_size, (1,)).item()
                    if torch.rand(1).item() < 0.5:  # 50% chance for each block
                        mask[b, start:start + coarse_size] = True

                # Fine mask: many small blocks
                num_fine = int(seq_len * self.mask_ratio * 0.3 / self.mask_block_size)
                for _ in range(num_fine):
                    start = torch.randint(0, seq_len - self.mask_block_size, (1,)).item()
                    mask[b, start:start + self.mask_block_size] = True

        else:
            # Simple block masking
            num_blocks = int(seq_len * self.mask_ratio / self.mask_block_size)
            mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

            for b in range(batch_size):
                for _ in range(num_blocks):
                    start = torch.randint(0, max(1, seq_len - self.mask_block_size), (1,)).item()
                    mask[b, start:start + self.mask_block_size] = True

        return mask

    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute JEPA loss.

        Args:
            context_embeddings: [B, L, D] encoder output for visible context
            target_embeddings: [B, L, D] EMA encoder output (targets)
            mask: [B, L] boolean mask (True = masked positions to predict)
            action: [B, action_dim] optional action for conditioning

        Returns:
            LossOutput with JEPA loss components
        """
        B, L, D = context_embeddings.shape
        device = context_embeddings.device

        # Create mask if not provided
        if mask is None:
            mask = self.create_block_mask(L, B, device)

        # Project to predictor dimension
        context_proj = self.predictor_embed(context_embeddings)  # [B, L, D_pred]

        # Create predictor input:
        # - Keep context embeddings for visible positions
        # - Replace masked positions with learnable mask tokens
        mask_tokens = self.mask_token.expand(B, L, -1)  # [B, L, D_pred]
        predictor_input = torch.where(
            mask.unsqueeze(-1).expand_as(context_proj),
            mask_tokens,
            context_proj,
        )

        # Add positional encoding
        if L <= self.predictor_pos_embed.shape[1]:
            predictor_input = predictor_input + self.predictor_pos_embed[:, :L, :]
        else:
            # Extend positional encoding if sequence is longer
            pos_embed = F.interpolate(
                self.predictor_pos_embed.permute(0, 2, 1),
                size=L,
                mode='linear',
            ).permute(0, 2, 1)
            predictor_input = predictor_input + pos_embed

        # Apply action conditioning (FiLM)
        if action is not None and self.use_action_conditioning:
            gamma = self.action_gamma(action).unsqueeze(1)  # [B, 1, D_pred]
            beta = self.action_beta(action).unsqueeze(1)  # [B, 1, D_pred]
            predictor_input = gamma * predictor_input + beta

        # Run predictor
        predicted = self.predictor(predictor_input)  # [B, L, D_pred]

        # Project back to target dimension
        predicted = self.predictor_proj(predicted)  # [B, L, D]

        # Normalize targets (stop gradient - targets come from EMA encoder)
        target_normalized = self.target_norm(target_embeddings.detach())

        # Compute L1 loss only on masked positions
        # This is the key JEPA objective
        mask_expanded = mask.unsqueeze(-1).expand_as(predicted)

        # Masked L1 loss
        l1_diff = torch.abs(predicted - target_normalized)
        masked_diff = l1_diff * mask_expanded.float()

        # Average over masked positions only
        num_masked = mask.sum().float().clamp(min=1)
        jepa_loss = masked_diff.sum() / (num_masked * D)

        # Optional: cosine similarity loss for additional regularization
        pred_flat = predicted[mask].view(-1, D)
        target_flat = target_normalized[mask].view(-1, D)

        if pred_flat.shape[0] > 0:
            pred_norm = F.normalize(pred_flat, dim=-1)
            target_norm_flat = F.normalize(target_flat, dim=-1)
            cosine_loss = 1 - (pred_norm * target_norm_flat).sum(dim=-1).mean()
        else:
            cosine_loss = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = jepa_loss + 0.1 * cosine_loss

        # Compute metrics
        with torch.no_grad():
            # Prediction error on masked positions
            mse_masked = (l1_diff ** 2 * mask_expanded.float()).sum() / (num_masked * D)

            # Variance of predictions (should be high = diverse predictions)
            pred_var = predicted[mask].var().item() if mask.sum() > 0 else 0.0

        return LossOutput(
            total=total_loss,
            components={
                "jepa_l1": jepa_loss,
                "jepa_cosine": 0.1 * cosine_loss,
            },
            metrics={
                "jepa_mse": mse_masked.item(),
                "pred_variance": pred_var,
                "mask_ratio": mask.float().mean().item(),
            },
        )


class AnomalyDetectionLoss(nn.Module):
    """
    Anomaly Detection Loss.

    For SWaT and similar anomaly detection tasks.
    """

    def __init__(
        self,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        pos_weight: Optional[float] = None,
    ):
        """
        Args:
            use_focal: Use focal loss for class imbalance
            focal_gamma: Focal loss gamma
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossOutput:
        """
        Compute anomaly detection loss.

        Args:
            predictions: [B, T] anomaly scores (logits)
            targets: [B, T] binary labels (0=normal, 1=anomaly)

        Returns:
            LossOutput with anomaly detection loss
        """
        if self.use_focal:
            loss = self._focal_loss(predictions, targets)
        else:
            pos_weight = torch.tensor([self.pos_weight]) if self.pos_weight else None
            loss = F.binary_cross_entropy_with_logits(
                predictions, targets, pos_weight=pos_weight
            )

        # Compute metrics
        with torch.no_grad():
            probs = torch.sigmoid(predictions)
            preds_binary = (probs > 0.5).float()

            # Precision, recall, F1
            tp = ((preds_binary == 1) & (targets == 1)).sum().float()
            fp = ((preds_binary == 1) & (targets == 0)).sum().float()
            fn = ((preds_binary == 0) & (targets == 1)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return LossOutput(
            total=loss,
            components={"anomaly_bce": loss},
            metrics={
                "anomaly_precision": precision.item(),
                "anomaly_recall": recall.item(),
                "anomaly_f1": f1.item(),
            },
        )

    def _focal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss for class imbalance."""
        probs = torch.sigmoid(predictions)

        # Focal weight
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.focal_gamma

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )

        return (focal_weight * bce).mean()
