"""
MTS-JEPA: Multi-Scale Temporal Self-Supervised Learning with JEPA
Full architecture implementation following the paper specification.

Components:
  1. Residual CNN tokenizer
  2. Channel-independent Transformer encoder
  3. Soft codebook with K=128 prototypes
  4. Fine predictor (Transformer)
  5. Coarse predictor (cross-attention + Transformer)
  6. Reconstruction decoder
  7. Full MTS-JEPA with EMA target branch
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Residual CNN Tokenizer
# ============================================================================

class ResidualBlock(nn.Module):
    """1D residual block for the CNN tokenizer."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class CNNTokenizer(nn.Module):
    """
    Residual CNN tokenizer: converts (L, 1) patches to d_model tokens.
    Channel-independent: operates on univariate patches.

    Input: (B*V, P, L) — P patches of length L for each variable
    Output: (B*V, P, d_model) — token embeddings
    """
    def __init__(self, patch_length=20, d_model=256, n_residual_blocks=2):
        super().__init__()
        self.patch_length = patch_length
        self.d_model = d_model

        # Initial projection from univariate patch to hidden dim
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(d_model)

        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(d_model) for _ in range(n_residual_blocks)
        ])

        # Pool over time dimension to get single token per patch
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: (B*V, P, L) — univariate patches
        Returns: (B*V, P, d_model)
        """
        BV, P, L = x.shape
        # Process each patch independently
        x = x.reshape(BV * P, 1, L)  # (BV*P, 1, L)
        x = F.gelu(self.input_bn(self.input_proj(x)))  # (BV*P, d_model, L)
        x = self.res_blocks(x)  # (BV*P, d_model, L)
        x = self.pool(x).squeeze(-1)  # (BV*P, d_model)
        return x.reshape(BV, P, self.d_model)


# ============================================================================
# 2. Transformer Encoder (Channel-Independent)
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer."""
    def __init__(self, d_model=256, n_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with pre-norm
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2, x2, x2)[0])
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


class ProjectionHead(nn.Module):
    """2-layer MLP projection head: d_model -> 64 -> 32 -> D."""
    def __init__(self, d_model=256, hidden=64, mid=32, d_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, mid),
            nn.GELU(),
            nn.Linear(mid, d_out),
        )

    def forward(self, x):
        return self.net(x)


class MTSJEPAEncoder(nn.Module):
    """
    Channel-independent encoder.

    1. CNN tokenizer: (L, 1) patches -> d_model tokens
    2. 6-layer Transformer (8 heads, dropout 0.1) over P patch tokens
    3. Projection head: d_model -> 64 -> 32 -> D=256

    Input: (B, P, L, V) or (B, 1, L, V) multi-scale view
    Output: (B, V, P, D) patch-level representations
    """
    def __init__(self, d_model=256, d_out=256, n_layers=6, n_heads=8,
                 patch_length=20, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.patch_length = patch_length

        self.tokenizer = CNNTokenizer(patch_length, d_model)

        # Learnable positional embeddings for patches
        # Max 5 patches for fine view, 1 for coarse — use 5 max
        self.pos_embed = nn.Parameter(torch.randn(1, 5, d_model) * 0.02)

        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.projection = ProjectionHead(d_model, 64, 32, d_out)

    def forward(self, x):
        """
        x: (B, P, L, V) multi-scale view
        Returns: (B, V, P, D)
        """
        B, P, L, V = x.shape

        # Channel-independent: reshape to process each variable separately
        # (B, P, L, V) -> (B*V, P, L)
        x = x.permute(0, 3, 1, 2).reshape(B * V, P, L)

        # Tokenize patches
        tokens = self.tokenizer(x)  # (B*V, P, d_model)

        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, :P, :]

        # Transformer over patch tokens
        for layer in self.transformer:
            tokens = layer(tokens)
        tokens = self.norm(tokens)  # (B*V, P, d_model)

        # Project to output dimension
        h = self.projection(tokens)  # (B*V, P, D)

        # Reshape back: (B*V, P, D) -> (B, V, P, D)
        h = h.reshape(B, V, P, self.d_out)
        return h


# ============================================================================
# 3. Soft Codebook
# ============================================================================

class SoftCodebook(nn.Module):
    """
    K=128 learnable prototypes in R^D.

    For each patch representation h in R^D:
    1. L2-normalize h and all prototypes c_k
    2. Compute cosine similarities: sim_k = <h_bar, c_bar_k>
    3. Apply temperature scaling: p_k = softmax(sim / tau)
    4. Soft-quantized embedding: z = sum_k p_k * c_k

    Returns: (p, z) where p in Delta^{K-1} and z in R^D
    """
    def __init__(self, K=128, D=256, tau=0.1):
        super().__init__()
        self.K = K
        self.D = D
        self.tau = tau

        # Learnable prototypes, initialized uniformly on unit sphere
        self.prototypes = nn.Parameter(torch.randn(K, D))
        nn.init.uniform_(self.prototypes, -1, 1)
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

    def forward(self, h):
        """
        h: (..., D) — patch representations
        Returns: (p, z) where p: (..., K) probabilities, z: (..., D) soft-quantized
        """
        # L2-normalize
        h_bar = F.normalize(h, dim=-1)  # (..., D)
        c_bar = F.normalize(self.prototypes, dim=-1)  # (K, D)

        # Cosine similarity
        sim = torch.matmul(h_bar, c_bar.T)  # (..., K)

        # Temperature-scaled softmax
        p = F.softmax(sim / self.tau, dim=-1)  # (..., K)

        # Soft-quantized embedding
        z = torch.matmul(p, self.prototypes)  # (..., D) — use raw prototypes, not normalized

        return p, z


# ============================================================================
# 4. Fine Predictor
# ============================================================================

class FinePredictor(nn.Module):
    """
    2-layer Transformer (4 heads, hidden 128) for fine-grained prediction.

    Input: Pi_t in R^{P x K} (code distributions from codebook)
    Output: Pi_hat^fine_{t+1} in R^{P x K} (predicted fine code distributions)
    """
    def __init__(self, K=128, n_patches=5, n_layers=2, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.K = K
        # Input projection: K -> d_ff
        self.input_proj = nn.Linear(K, d_ff)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_ff) * 0.02)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_ff, n_heads, d_ff * 2, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_ff)
        # Output projection: d_ff -> K
        self.output_proj = nn.Linear(d_ff, K)

    def forward(self, pi):
        """
        pi: (B, P, K) code distributions from context
        Returns: (B, P, K) predicted code distributions for target
        """
        B, P, K = pi.shape
        x = self.input_proj(pi)  # (B, P, d_ff)
        x = x + self.pos_embed[:, :P, :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output_proj(x)  # (B, P, K)
        return F.softmax(logits, dim=-1)


# ============================================================================
# 5. Coarse Predictor
# ============================================================================

class CoarsePredictor(nn.Module):
    """
    Learnable query + cross-attention over fine codes, then 2-layer Transformer.

    Input: Pi_t in R^{P x K} (fine code distributions from context)
    Output: Pi_hat^coarse_{t+1} in R^{1 x K} (single global prediction)
    """
    def __init__(self, K=128, n_patches=5, n_layers=2, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.K = K

        # Learnable query token
        self.query = nn.Parameter(torch.randn(1, 1, d_ff) * 0.02)

        # Input projection for keys/values
        self.kv_proj = nn.Linear(K, d_ff)

        # Cross-attention: query attends to fine codes
        self.cross_attn = nn.MultiheadAttention(d_ff, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_ff)

        # Self-attention layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_ff, n_heads, d_ff * 2, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_ff)
        self.output_proj = nn.Linear(d_ff, K)

    def forward(self, pi):
        """
        pi: (B, P, K) fine code distributions from context
        Returns: (B, 1, K) predicted coarse code distribution
        """
        B, P, K = pi.shape

        # Project fine codes to keys/values
        kv = self.kv_proj(pi)  # (B, P, d_ff)

        # Cross-attention: query attends to fine codes
        query = self.query.expand(B, -1, -1)  # (B, 1, d_ff)
        x = self.cross_norm(query + self.cross_attn(query, kv, kv)[0])  # (B, 1, d_ff)

        # Self-attention refinement
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        logits = self.output_proj(x)  # (B, 1, K)
        return F.softmax(logits, dim=-1)


# ============================================================================
# 6. Reconstruction Decoder
# ============================================================================

class ReconstructionDecoder(nn.Module):
    """
    Reconstructs input patches from soft-quantized embeddings.

    Input: z_t in R^{P x D} (soft-quantized context embeddings)
    Output: X_hat_t in R^{P x L} (reconstructed univariate patches)
    """
    def __init__(self, D=256, patch_length=20):
        super().__init__()
        self.patch_length = patch_length
        self.net = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, patch_length),
        )

    def forward(self, z):
        """
        z: (..., D) soft-quantized embeddings
        Returns: (..., patch_length) reconstructed patches
        """
        return self.net(z)


# ============================================================================
# 7. Loss Functions
# ============================================================================

def kl_divergence(p_pred, p_target, eps=1e-8):
    """
    KL(p_target || p_pred) summed over patches.
    p_pred, p_target: (..., K) probability distributions
    """
    return (p_target * (torch.log(p_target + eps) - torch.log(p_pred + eps))).sum(dim=-1).mean()


def embedding_mse(z_pred, z_target):
    """||z_pred - z_target||^2 summed over patches."""
    return F.mse_loss(z_pred, z_target)


def codebook_alignment_loss(h, z):
    """
    Bidirectional alignment with stop-gradient.
    L_emb = ||sg(z) - h||^2
    L_com = ||z - sg(h)||^2
    """
    L_emb = F.mse_loss(h, z.detach())
    L_com = F.mse_loss(z, h.detach())
    return L_emb, L_com


def dual_entropy_loss(p_batch):
    """
    Sample entropy: E[H(p)] — minimize for sharp assignments
    Batch entropy: H(E[p]) — maximize for diverse code usage

    p_batch: (N, K) code distributions across a batch
    """
    eps = 1e-8
    # Sample entropy: average entropy of individual distributions
    sample_entropy = -(p_batch * torch.log(p_batch + eps)).sum(dim=-1).mean()

    # Batch entropy: entropy of the average distribution
    avg_p = p_batch.mean(dim=0)  # (K,)
    batch_entropy = -(avg_p * torch.log(avg_p + eps)).sum()

    return sample_entropy, batch_entropy


def reconstruction_loss(x_hat, x):
    """MSE reconstruction loss."""
    return F.mse_loss(x_hat, x)


# ============================================================================
# 8. Full MTS-JEPA Model
# ============================================================================

class MTSJEPA(nn.Module):
    """
    Full MTS-JEPA model with EMA target branch.

    Online branch: encoder + codebook + predictors + decoder
    Target branch: EMA copies of encoder + codebook (no gradients)
    """
    def __init__(self, n_vars, d_model=256, d_out=256, n_codes=128, tau=0.1,
                 patch_length=20, n_patches=5, n_encoder_layers=6,
                 n_heads=8, dropout=0.1, ema_rho=0.996):
        super().__init__()

        self.n_vars = n_vars
        self.d_model = d_model
        self.d_out = d_out
        self.n_codes = n_codes
        self.patch_length = patch_length
        self.n_patches = n_patches
        self.ema_rho = ema_rho

        # Online branch components
        self.encoder = MTSJEPAEncoder(
            d_model=d_model, d_out=d_out, n_layers=n_encoder_layers,
            n_heads=n_heads, patch_length=patch_length, dropout=dropout
        )
        self.codebook = SoftCodebook(K=n_codes, D=d_out, tau=tau)
        self.fine_predictor = FinePredictor(
            K=n_codes, n_patches=n_patches, n_layers=2, n_heads=4,
            d_ff=128, dropout=dropout
        )
        self.coarse_predictor = CoarsePredictor(
            K=n_codes, n_patches=n_patches, n_layers=2, n_heads=4,
            d_ff=128, dropout=dropout
        )
        self.decoder = ReconstructionDecoder(D=d_out, patch_length=patch_length)

        # RevIN is applied externally (needs to be per-window, shared across context/target)

        # EMA target branch — deep copies, no gradients
        self.ema_encoder = copy.deepcopy(self.encoder)
        self.ema_codebook = copy.deepcopy(self.codebook)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False
        for p in self.ema_codebook.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        """Update EMA encoder and codebook."""
        rho = self.ema_rho
        for p_online, p_ema in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            p_ema.data.mul_(rho).add_(p_online.data, alpha=1 - rho)
        for p_online, p_ema in zip(self.codebook.parameters(), self.ema_codebook.parameters()):
            p_ema.data.mul_(rho).add_(p_online.data, alpha=1 - rho)

    def online_params(self):
        """Parameters for the optimizer (excludes EMA targets)."""
        params = list(self.encoder.parameters())
        params += list(self.codebook.parameters())
        params += list(self.fine_predictor.parameters())
        params += list(self.coarse_predictor.parameters())
        params += list(self.decoder.parameters())
        return params

    def forward(self, x_context, x_target, return_details=False):
        """
        Forward pass for pre-training.

        x_context: (B, T_w, V) — context window (already RevIN-normalized)
        x_target: (B, T_w, V) — target window (already RevIN-normalized)

        Returns dict of all loss components.
        """
        from data_utils import create_views

        B, T, V = x_context.shape

        # ================================================================
        # Create multi-scale views
        # ================================================================
        # Context views
        ctx_fine, ctx_coarse = create_views(x_context, self.n_patches, self.patch_length)
        # ctx_fine: (B, P, L, V), ctx_coarse: (B, 1, L, V)

        # Target views (for EMA branch)
        tgt_fine, tgt_coarse = create_views(x_target, self.n_patches, self.patch_length)

        # ================================================================
        # Online branch: encode context
        # ================================================================
        h_ctx_fine = self.encoder(ctx_fine)  # (B, V, P, D)

        # Codebook on context fine representations
        # Reshape for codebook: (B, V, P, D) -> flatten -> apply codebook
        p_ctx, z_ctx = self.codebook(h_ctx_fine)  # p: (B, V, P, K), z: (B, V, P, D)

        # ================================================================
        # Online branch: predict target codes
        # ================================================================
        # Fine predictor operates per-variable: (B*V, P, K)
        p_ctx_flat = p_ctx.reshape(B * V, self.n_patches, self.n_codes)

        p_pred_fine = self.fine_predictor(p_ctx_flat)  # (B*V, P, K)
        p_pred_fine = p_pred_fine.reshape(B, V, self.n_patches, self.n_codes)

        # Coarse predictor also per-variable
        p_pred_coarse = self.coarse_predictor(p_ctx_flat)  # (B*V, 1, K)
        p_pred_coarse = p_pred_coarse.reshape(B, V, 1, self.n_codes)

        # ================================================================
        # Target branch (EMA, no gradients)
        # ================================================================
        with torch.no_grad():
            # Encode target at fine and coarse resolutions
            h_tgt_fine = self.ema_encoder(tgt_fine)  # (B, V, P, D)
            h_tgt_coarse = self.ema_encoder(tgt_coarse)  # (B, V, 1, D)

            # Codebook on target representations
            p_tgt_fine, z_tgt_fine = self.ema_codebook(h_tgt_fine)  # (B, V, P, K)
            p_tgt_coarse, z_tgt_coarse = self.ema_codebook(h_tgt_coarse)  # (B, V, 1, K)

        # ================================================================
        # Compute predicted embeddings for MSE loss
        # ================================================================
        # z_pred_fine = sum_k p_pred_k * c_k
        z_pred_fine = torch.matmul(p_pred_fine, self.codebook.prototypes)  # (B, V, P, D)
        z_pred_coarse = torch.matmul(p_pred_coarse, self.codebook.prototypes)  # (B, V, 1, D)

        # ================================================================
        # Reconstruction branch
        # ================================================================
        # Decode context embeddings back to patch space
        # z_ctx: (B, V, P, D) -> reconstruct -> (B, V, P, L)
        x_rec = self.decoder(z_ctx)  # (B, V, P, patch_length)

        # Reconstruction target: original context fine patches (per variable)
        # ctx_fine: (B, P, L, V) -> (B, V, P, L)
        x_ctx_target = ctx_fine.permute(0, 3, 1, 2)  # (B, V, P, L)

        # ================================================================
        # Loss computation
        # ================================================================
        losses = {}

        # --- Prediction losses ---
        # Fine KL: KL(p_tgt_fine || p_pred_fine)
        losses['kl_fine'] = kl_divergence(p_pred_fine, p_tgt_fine.detach())

        # Fine MSE: ||z_pred_fine - z_tgt_fine||^2
        losses['mse_fine'] = embedding_mse(z_pred_fine, z_tgt_fine.detach())

        # Coarse KL: KL(p_tgt_coarse || p_pred_coarse)
        losses['kl_coarse'] = kl_divergence(p_pred_coarse, p_tgt_coarse.detach())

        # --- Codebook losses ---
        # Alignment losses
        L_emb, L_com = codebook_alignment_loss(h_ctx_fine, z_ctx)
        losses['emb'] = L_emb
        losses['com'] = L_com

        # Dual entropy
        # Flatten all code distributions for entropy computation
        p_all = p_ctx.reshape(-1, self.n_codes)  # (B*V*P, K)
        sample_ent, batch_ent = dual_entropy_loss(p_all)
        losses['ent_sample'] = sample_ent
        losses['ent_batch'] = batch_ent

        # --- Reconstruction loss ---
        losses['rec'] = reconstruction_loss(x_rec, x_ctx_target)

        # --- Codebook utilization stats ---
        with torch.no_grad():
            # Which codes are used?
            assignments = p_all.argmax(dim=-1)
            unique_codes = assignments.unique().numel()
            losses['codebook_utilization'] = unique_codes / self.n_codes

            # Perplexity of average distribution
            avg_p = p_all.mean(dim=0)
            perplexity = torch.exp(-(avg_p * torch.log(avg_p + 1e-8)).sum())
            losses['codebook_perplexity'] = perplexity.item()

        if return_details:
            losses['p_ctx'] = p_ctx
            losses['z_ctx'] = z_ctx
            losses['p_pred_fine'] = p_pred_fine
            losses['p_pred_coarse'] = p_pred_coarse
            losses['h_ctx_fine'] = h_ctx_fine

        return losses

    def encode_for_downstream(self, x_context):
        """
        Encode context windows for downstream classification.

        x_context: (B, T_w, V) — normalized context window

        Returns: (B, P*K) flattened code representations after variable-wise max-pooling
        """
        from data_utils import create_views

        B, T, V = x_context.shape

        # Create fine view
        fine_view, _ = create_views(x_context, self.n_patches, self.patch_length)

        # Encode (using online encoder — frozen at downstream time)
        with torch.no_grad():
            h = self.encoder(fine_view)  # (B, V, P, D)
            p, z = self.codebook(h)  # p: (B, V, P, K)

        # Variable-wise max-pooling: max over V dimension
        # p: (B, V, P, K) -> (B, P, K)
        p_pooled = p.max(dim=1).values  # (B, P, K)

        # Flatten: (B, P*K)
        return p_pooled.reshape(B, -1)


# ============================================================================
# 9. Downstream MLP Classifier
# ============================================================================

class DownstreamClassifier(nn.Module):
    """
    MLP classifier for anomaly prediction.
    Input: (B, P*K) flattened code representations
    Output: (B,) anomaly probability
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# 10. Ablation Variants
# ============================================================================

class MTSJEPANoCodebook(MTSJEPA):
    """MTS-JEPA without codebook — operates in continuous latent space."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override codebook to be identity
        self.codebook = None
        self.ema_codebook = None

    def forward(self, x_context, x_target, return_details=False):
        from data_utils import create_views
        B, T, V = x_context.shape

        ctx_fine, ctx_coarse = create_views(x_context, self.n_patches, self.patch_length)
        tgt_fine, tgt_coarse = create_views(x_target, self.n_patches, self.patch_length)

        # Encode
        h_ctx_fine = self.encoder(ctx_fine)  # (B, V, P, D)

        with torch.no_grad():
            h_tgt_fine = self.ema_encoder(tgt_fine)
            h_tgt_coarse = self.ema_encoder(tgt_coarse)

        # Predict directly in continuous space (use D instead of K)
        # We use a simple MLP predictor instead
        losses = {
            'mse_fine': F.mse_loss(h_ctx_fine, h_tgt_fine.detach()),
            'rec': reconstruction_loss(self.decoder(h_ctx_fine), ctx_fine.permute(0, 3, 1, 2)),
            'kl_fine': torch.tensor(0.0, device=h_ctx_fine.device),
            'kl_coarse': torch.tensor(0.0, device=h_ctx_fine.device),
            'emb': torch.tensor(0.0, device=h_ctx_fine.device),
            'com': torch.tensor(0.0, device=h_ctx_fine.device),
            'ent_sample': torch.tensor(0.0, device=h_ctx_fine.device),
            'ent_batch': torch.tensor(0.0, device=h_ctx_fine.device),
            'codebook_utilization': 0.0,
            'codebook_perplexity': 0.0,
        }
        return losses
