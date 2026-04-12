"""
CC-JEPA: Causal Codebook JEPA

Our primary NeurIPS extension. Merges:
- Trajectory JEPA's causal encoder (genuine future prediction)
- MTS-JEPA's soft codebook (collapse prevention + interpretability)

Key differences from MTS-JEPA:
1. Causal masking in encoder (not bidirectional) — ensures prediction, not detection
2. Variable-horizon prediction — not fixed next-window
3. Multivariate encoding (not channel-independent) — captures cross-variable dynamics
4. Same soft codebook + dual entropy regularization

Key differences from Trajectory JEPA V11:
1. Soft codebook quantization — prevents collapse, enables interpretability
2. Multi-resolution views — fine + coarse temporal scales
3. KL + MSE dual prediction loss — not just MSE
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import (
    SoftCodebook, FinePredictor, CoarsePredictor,
    ReconstructionDecoder, TransformerEncoderLayer,
    kl_divergence, embedding_mse, codebook_alignment_loss,
    dual_entropy_loss, reconstruction_loss, CNNTokenizer,
)


class CausalMultivariateEncoder(nn.Module):
    """
    Causal multivariate encoder.

    Unlike MTS-JEPA's channel-independent encoder, this processes all variables
    jointly, enabling cross-variable interaction from the start.

    Uses causal masking so position t can only attend to positions 0..t.
    """
    def __init__(self, n_vars, d_model=128, d_out=128, n_layers=3,
                 n_heads=4, patch_length=20, n_patches=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.n_patches = n_patches
        self.patch_length = patch_length

        # Input projection: each patch (L, V) -> d_model
        self.input_proj = nn.Linear(patch_length * n_vars, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Causal Transformer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_out),
        )

    def forward(self, x, causal=True):
        """
        x: (B, P, L, V) multi-scale view
        Returns: (B, P, D) patch-level representations
        """
        B, P, L, V = x.shape

        # Flatten each patch: (B, P, L*V)
        x = x.reshape(B, P, L * V)

        # Project to d_model
        h = self.input_proj(x)  # (B, P, d_model)
        h = h + self.pos_embed[:, :P, :]

        # Causal mask: upper triangular = True (masked positions)
        if causal:
            mask = torch.triu(torch.ones(P, P, device=h.device), diagonal=1).bool()
        else:
            mask = None

        for layer in self.layers:
            if mask is not None:
                # Apply causal mask via manual attention computation
                h2 = layer.norm1(h)
                h = h + layer.dropout(layer.attn(h2, h2, h2, attn_mask=mask)[0])
                h = h + layer.ffn(layer.norm2(h))
            else:
                h = layer(h)

        h = self.norm(h)
        return self.projection(h)  # (B, P, D)


class CCJEPA(nn.Module):
    """
    Causal Codebook JEPA.

    Architecture:
    - Causal multivariate encoder (online)
    - Bidirectional multivariate encoder (EMA target)
    - Soft codebook (shared between online and target)
    - Fine predictor + Coarse predictor
    - Reconstruction decoder
    """
    def __init__(self, n_vars, d_model=128, d_out=128, n_codes=64, tau=0.1,
                 patch_length=20, n_patches=5, n_encoder_layers=3,
                 n_heads=4, dropout=0.1, ema_rho=0.996):
        super().__init__()
        self.n_vars = n_vars
        self.d_out = d_out
        self.n_codes = n_codes
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.ema_rho = ema_rho

        # Causal encoder (online)
        self.encoder = CausalMultivariateEncoder(
            n_vars, d_model, d_out, n_encoder_layers,
            n_heads, patch_length, n_patches, dropout
        )

        # Codebook
        self.codebook = SoftCodebook(K=n_codes, D=d_out, tau=tau)

        # Predictors (operate on code distributions)
        self.fine_predictor = FinePredictor(
            K=n_codes, n_patches=n_patches, n_layers=2, n_heads=4,
            d_ff=d_model, dropout=dropout
        )
        self.coarse_predictor = CoarsePredictor(
            K=n_codes, n_patches=n_patches, n_layers=2, n_heads=4,
            d_ff=d_model, dropout=dropout
        )

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_length * n_vars),
        )

        # EMA target encoder (bidirectional)
        self.ema_encoder = CausalMultivariateEncoder(
            n_vars, d_model, d_out, n_encoder_layers,
            n_heads, patch_length, n_patches, dropout
        )
        self.ema_codebook = copy.deepcopy(self.codebook)

        # Initialize EMA = online
        self._init_ema()

    def _init_ema(self):
        for p_o, p_e in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            p_e.data.copy_(p_o.data)
            p_e.requires_grad = False
        for p_o, p_e in zip(self.codebook.parameters(), self.ema_codebook.parameters()):
            p_e.data.copy_(p_o.data)
            p_e.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        rho = self.ema_rho
        for p_o, p_e in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            p_e.data.mul_(rho).add_(p_o.data, alpha=1 - rho)
        for p_o, p_e in zip(self.codebook.parameters(), self.ema_codebook.parameters()):
            p_e.data.mul_(rho).add_(p_o.data, alpha=1 - rho)

    def online_params(self):
        params = list(self.encoder.parameters())
        params += list(self.codebook.parameters())
        params += list(self.fine_predictor.parameters())
        params += list(self.coarse_predictor.parameters())
        params += list(self.decoder.parameters())
        return params

    def forward(self, x_context, x_target, return_details=False):
        """
        x_context: (B, T_w, V)
        x_target: (B, T_w, V)
        """
        from data_utils import create_views

        B, T, V = x_context.shape

        # Create multi-scale views
        ctx_fine, ctx_coarse = create_views(x_context, self.n_patches, self.patch_length)
        tgt_fine, tgt_coarse = create_views(x_target, self.n_patches, self.patch_length)

        # Online branch: CAUSAL encode context
        h_ctx = self.encoder(ctx_fine, causal=True)  # (B, P, D)
        p_ctx, z_ctx = self.codebook(h_ctx)  # p: (B, P, K), z: (B, P, D)

        # Predict target codes
        p_pred_fine = self.fine_predictor(p_ctx)  # (B, P, K)
        p_pred_coarse = self.coarse_predictor(p_ctx)  # (B, 1, K)

        # Target branch: BIDIRECTIONAL encode target (EMA, no grad)
        with torch.no_grad():
            h_tgt_fine = self.ema_encoder(tgt_fine, causal=False)  # (B, P, D)
            h_tgt_coarse = self.ema_encoder(tgt_coarse, causal=False)  # (B, 1, D)
            p_tgt_fine, z_tgt_fine = self.ema_codebook(h_tgt_fine)
            p_tgt_coarse, z_tgt_coarse = self.ema_codebook(h_tgt_coarse)

        # Predicted embeddings for MSE
        z_pred_fine = torch.matmul(p_pred_fine, self.codebook.prototypes)
        z_pred_coarse = torch.matmul(p_pred_coarse, self.codebook.prototypes)

        # Reconstruction
        x_rec = self.decoder(z_ctx)  # (B, P, L*V)
        x_ctx_flat = ctx_fine.reshape(B, self.n_patches, -1)  # (B, P, L*V)

        # Losses
        losses = {}
        losses['kl_fine'] = kl_divergence(p_pred_fine, p_tgt_fine.detach())
        losses['mse_fine'] = embedding_mse(z_pred_fine, z_tgt_fine.detach())
        losses['kl_coarse'] = kl_divergence(p_pred_coarse, p_tgt_coarse.detach())

        L_emb, L_com = codebook_alignment_loss(h_ctx, z_ctx)
        losses['emb'] = L_emb
        losses['com'] = L_com

        p_all = p_ctx.reshape(-1, self.n_codes)
        sample_ent, batch_ent = dual_entropy_loss(p_all)
        losses['ent_sample'] = sample_ent
        losses['ent_batch'] = batch_ent

        losses['rec'] = F.mse_loss(x_rec, x_ctx_flat)

        with torch.no_grad():
            assignments = p_all.argmax(dim=-1)
            losses['codebook_utilization'] = assignments.unique().numel() / self.n_codes
            avg_p = p_all.mean(dim=0)
            losses['codebook_perplexity'] = torch.exp(-(avg_p * torch.log(avg_p + 1e-8)).sum()).item()

        if return_details:
            losses['p_ctx'] = p_ctx
            losses['z_ctx'] = z_ctx

        return losses

    def encode_for_downstream(self, x):
        """
        Encode context window for downstream classification.
        x: (B, T, V)
        Returns: (B, P*K) code representation
        """
        from data_utils import create_views
        B, T, V = x.shape
        fine_view, _ = create_views(x, self.n_patches, self.patch_length)

        with torch.no_grad():
            h = self.encoder(fine_view, causal=True)
            p, z = self.codebook(h)  # p: (B, P, K)

        return p.reshape(B, -1)  # (B, P*K)
