# Technical Specifications: Three Directions

---

## Direction 1: SparseGraph-Transformer

### Problem Statement
Learn a sparse, discrete graph over sensor channels that captures true physical coupling. Use this graph for structured message passing. Show that the learned graph (a) matches known physics, (b) transfers across operating conditions, (c) resolves the channel-independence paradox.

### Formal Definition

**Input**: X ∈ R^{B × T × C} (B batches, T timesteps, C channels)
**Output**: ŷ ∈ R^{B × 1} (RUL prediction)
**Learned**: A ∈ {0,1}^{C × C} (binary adjacency matrix)

### Architecture

```python
class SparseGraphTransformer(nn.Module):
    def __init__(self, n_channels, d=32, heads=4, layers=2, sparsity_k=20):
        # Node embeddings for graph learning (GTS-style)
        self.node_embed = nn.Embedding(n_channels, d)
        # Pairwise edge predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d, 1)
        )
        # Per-channel temporal encoder (shared, like CI-Trans)
        self.temporal_enc = TransformerEncoder(d, heads, layers)
        # Graph convolution with learned adjacency
        self.graph_conv = SparseGCN(d, layers=2)
        # RUL head
        self.head = nn.Linear(d * n_channels, 1)
        self.sparsity_k = sparsity_k  # top-k edges to keep

    def learn_graph(self):
        """Learn binary adjacency via Gumbel-softmax."""
        embeds = self.node_embed.weight  # (C, d)
        # Pairwise: concatenate all pairs
        pairs = torch.cat([embeds[i].repeat(C,1), embeds], dim=1)  # (C*C, 2d)
        logits = self.edge_mlp(pairs).reshape(C, C)
        # Top-k sparsification + Gumbel-softmax
        A = gumbel_top_k(logits, k=self.sparsity_k, tau=0.5)
        return A

    def forward(self, x):
        B, T, C = x.shape
        A = self.learn_graph()
        # Temporal encoding per channel
        h = self.temporal_enc(x)  # (B, C, d)
        # Graph message passing
        h = self.graph_conv(h, A)  # (B, C, d)
        return self.head(h.reshape(B, -1))
```

### Training Procedure
1. Standard supervised training with MSE loss for RUL
2. Sparsity regularization: L1 on edge logits
3. Graph is learned jointly with the forecasting objective
4. Temperature annealing for Gumbel-softmax: τ = 1.0 → 0.1 over training

### What Makes This Novel
- **Physics validation**: Compare learned graph to known C-MAPSS component structure
- **Transfer**: Test if learned graph on FD001 is useful for FD002
- **Resolves paradox**: Sparse graph should beat both CI (too independent) and dense (too dependent)

### Hypotheses
1. Learned sparse graph recovers >70% of known physical edges
2. Transfer with learned graph beats CI-Trans (our baseline: ratio 6.16)
3. Sparse graph beats both dense attention and channel-independence

### Risk Assessment
- Medium risk. Graph learning is well-studied; main novelty is physics validation + transfer.
- May converge to trivial graph (all edges or no edges).

---

## Direction 2: Slot-Concept Transformer

### Problem Statement
Discover latent physical "concepts" (component states) from multivariate sensor data without supervision. Use concepts for interpretable, transferable prediction.

### Formal Definition

**Input**: X ∈ R^{B × T × C} (sensor windows)
**Output**: ŷ ∈ R^{B × 1} (RUL), S ∈ R^{B × K × d} (K concept slots)
**Learned**: Slot attention discovers K concepts from C channels

### Architecture

```python
class SlotConceptTransformer(nn.Module):
    def __init__(self, n_channels, n_slots=5, d=32, heads=4, layers=2):
        # Per-channel temporal encoder (shared)
        self.temporal_enc = TransformerEncoder(d_in=1, d=d, heads=heads, layers=layers)
        # Slot attention module
        self.slot_attn = SlotAttention(
            n_slots=n_slots, d=d, n_iters=3,
            hidden_dim=d*2
        )
        # Cross-slot transformer (like Role-Trans cross-component)
        self.cross_slot = TransformerEncoder(d, heads, layers=1)
        self.slot_pos = nn.Parameter(torch.randn(1, n_slots, d) * 0.02)
        # RUL head
        self.head = nn.Linear(d * n_slots, 1)

    def forward(self, x):
        B, T, C = x.shape
        # Encode each channel independently
        h = encode_per_channel(self.temporal_enc, x)  # (B, C, d)
        # Slot attention: C channel features → K concept slots
        slots = self.slot_attn(h)  # (B, K, d)
        # Cross-slot interaction
        slots = self.cross_slot(slots + self.slot_pos)
        return self.head(slots.reshape(B, -1))

    def get_assignments(self, x):
        """Return soft channel-to-slot assignments for interpretability."""
        h = encode_per_channel(self.temporal_enc, x)
        _, attn = self.slot_attn(h, return_attn=True)
        return attn  # (B, K, C) — which channels belong to which concept

class SlotAttention(nn.Module):
    def __init__(self, n_slots, d, n_iters=3, hidden_dim=64):
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, d) * 0.02)
        self.slots_sigma = nn.Parameter(torch.ones(1, n_slots, d) * 0.02)
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d))
        self.norm_slots = nn.LayerNorm(d)
        self.norm_inputs = nn.LayerNorm(d)

    def forward(self, inputs, return_attn=False):
        B, N, D = inputs.shape
        # Initialize slots
        slots = self.slots_mu + self.slots_sigma * torch.randn_like(self.slots_mu)
        slots = slots.expand(B, -1, -1)
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)  # (B, N, D)
        v = self.to_v(inputs)  # (B, N, D)

        for _ in range(self.n_iters):
            q = self.to_q(self.norm_slots(slots))  # (B, K, D)
            attn = torch.softmax(
                torch.bmm(q, k.transpose(1, 2)) / D**0.5,
                dim=1  # normalize over slots (competition)
            )  # (B, K, N)
            updates = torch.bmm(attn, v)  # (B, K, D)
            slots = self.gru(
                updates.reshape(-1, D),
                slots.reshape(-1, D)
            ).reshape(B, -1, D)
            slots = slots + self.mlp(slots)

        if return_attn:
            return slots, attn
        return slots
```

### Training Procedure
1. **Phase 1**: Train end-to-end on FD001 with MSE loss for RUL
2. **Phase 2**: Analyze slot assignments — do they correspond to physical components?
3. **Phase 3**: Transfer to FD002 — do concepts transfer better than CI or fixed grouping?

### What Makes This Novel
1. **First slot attention for industrial multivariate sensors** (SlotFM exists for accelerometers only)
2. **Unsupervised concept discovery** where ground truth is available for validation
3. **Connects to CBMs** (EPFL 2025) but without requiring expert concept labels
4. **Unifies Role-Trans and CI-Trans**: if K=C, it's CI; if K=5 with physics, it's Role-Trans; learned K adapts

### Hypotheses
1. With K=5 slots and C=14 sensors, slots will discover groupings similar to physics components
2. Slot-based model transfers as well as Role-Trans (or better) without requiring physics knowledge
3. Slot assignments are stable across seeds and interpretable

### Risk Assessment
- Medium-high risk. Slot attention hasn't been validated on this type of data.
- May collapse to trivial assignments (all channels → one slot).
- May not discover meaningful groupings without physics-informed initialization.

---

## Direction 3: Mechanical-JEPA (Fixed)

### Problem Statement
Fix our failed JEPA by addressing root causes identified in literature review. Show that properly designed JEPA pretraining improves cross-condition transfer.

### Formal Definition

**Pretraining**: Given unlabeled X from multiple conditions, learn encoder θ that produces transferable representations.
**Fine-tuning**: Use pretrained θ for RUL prediction on source, transfer to target.

### Architecture (addressing each failure mode)

```python
class MechanicalJEPA(nn.Module):
    def __init__(self, n_channels, d=64, heads=4, layers=4,
                 n_codes=64, mask_ratio=0.75, n_patches=10):
        # DEEP encoder (4 layers, not 2) — Apple NeurIPS 2024
        self.patch_embed = nn.Linear(n_channels * (SEQ_LEN // n_patches), d)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, 0.1, batch_first=True)
        self.online_enc = nn.TransformerEncoder(enc, layers)
        self.target_enc = deepcopy(self.online_enc)  # EMA target

        # CODEBOOK BOTTLENECK (from MTS-JEPA) — prevents collapse
        self.codebook = nn.Parameter(torch.randn(n_codes, d) * 0.02)
        self.code_temp = 0.1

        # Predictor (lightweight)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        pred_enc = nn.TransformerEncoderLayer(d, heads, d*4, 0.1, batch_first=True)
        self.predictor = nn.TransformerEncoder(pred_enc, 2)

        self.mask_ratio = mask_ratio
        self.n_patches = n_patches

    def quantize(self, z):
        """Soft codebook assignment (from MTS-JEPA)."""
        # z: (B, N, d), codebook: (K, d)
        sim = torch.mm(z.reshape(-1, z.shape[-1]), self.codebook.t()) / self.code_temp
        weights = F.softmax(sim, dim=-1)  # (B*N, K)
        quantized = torch.mm(weights, self.codebook)  # (B*N, d)
        return quantized.reshape(z.shape)

    def forward(self, x):
        B, T, C = x.shape
        # Patchify: (B, T, C) → (B, n_patches, patch_dim)
        patches = x.reshape(B, self.n_patches, -1)
        tokens = self.patch_embed(patches) + self.pos

        # HIGH masking (75%) — V-JEPA style
        n_mask = int(self.n_patches * self.mask_ratio)
        mask_idx = torch.stack([torch.randperm(self.n_patches)[:n_mask] for _ in range(B)])

        # Target: full sequence through EMA encoder + codebook
        with torch.no_grad():
            target = self.target_enc(tokens)
            target = self.quantize(target)  # Codebook bottleneck

        # Online: visible patches only
        visible_mask = torch.ones(B, self.n_patches, dtype=torch.bool)
        for b in range(B):
            visible_mask[b, mask_idx[b]] = False
        visible_tokens = tokens[visible_mask].reshape(B, -1, tokens.shape[-1])
        context = self.online_enc(visible_tokens)

        # Predict: context + mask tokens
        pred_input = tokens.clone()
        for b in range(B):
            pred_input[b, mask_idx[b]] = self.mask_token
        # Inject context into visible positions
        j = 0
        for b in range(B):
            for i in range(self.n_patches):
                if visible_mask[b, i]:
                    pred_input[b, i] = context[b, j]
                    j += 1
            j = 0
        pred_output = self.predictor(pred_input)
        pred_output = self.quantize(pred_output)

        # L1 loss on masked positions (TS-JEPA style)
        loss = 0
        for b in range(B):
            loss += F.l1_loss(pred_output[b, mask_idx[b]], target[b, mask_idx[b]])
        return loss / B

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for op, tp in zip(self.online_enc.parameters(), self.target_enc.parameters()):
            tp.data = momentum * tp.data + (1 - momentum) * op.data
```

### What's Different from Our Failed JEPA

| Feature | Our Failed JEPA | Mechanical-JEPA |
|---------|----------------|-----------------|
| Mask ratio | 40% (2/5 components) | 75% (7-8/10 patches) |
| Masking unit | Components | Temporal patches |
| Encoder depth | 2 layers | 4 layers |
| Collapse prevention | EMA only | Codebook + EMA |
| Loss | MSE | L1 |
| Multi-scale | No | Patch-based |

### Hypotheses
1. Codebook bottleneck prevents representation collapse (loss doesn't go to 0)
2. 75% temporal masking forces dynamics learning, not interpolation
3. Pretrained encoder transfers better than scratch (reversing the -33% result)

### Risk Assessment
- High risk. We've failed 3 times. But root causes are now identified.
- Codebook may add too much quantization noise.
- 75% masking may be too aggressive for 30-timestep sequences.

---

## Hybrid: Slot-JEPA (Recommended First Experiment)

### Concept
Combine Slot-Concept Transformer (Direction 2) with Mechanical-JEPA objective (Direction 3):
1. Slot attention discovers components from sensors
2. JEPA predicts future slot states from past slot states in latent space
3. Codebook bottleneck prevents collapse
4. Train on FD001+FD002 unlabeled, fine-tune on FD001, transfer to FD002

### Why This Should Work
- Slot attention provides the right decomposition (role-based but learned)
- JEPA in slot space predicts component dynamics, not sensor noise
- Codebook discretizes component states (operating regimes become discrete codes)
- The combination addresses all our failure modes

### Implementation Priority
1. **First**: Test Slot-Concept Transformer alone (Direction 2) — simpler, validates concept discovery
2. **If concepts work**: Add JEPA in slot space (Slot-JEPA hybrid)
3. **If JEPA still fails**: Stay with Slot-Concept supervised training (still novel)

---

## Evaluation Framework

### Datasets
- **C-MAPSS** (FD001→FD002): Primary. Known physics for validation.
- **ETTh1**: Forecasting baseline (7 channels, no transfer).

### Metrics
1. **RUL RMSE**: Primary accuracy metric
2. **Transfer Ratio**: target_RMSE / source_RMSE
3. **Concept Alignment**: Jaccard similarity between learned slots and known component groups
4. **Graph Sparsity**: % of possible edges used
5. **Stability**: std across 5+ seeds

### Baselines
1. Linear (trivial)
2. LSTM (standard sequence)
3. CI-Transformer (channel-independent)
4. Role-Transformer (our best, fixed physics groups)
5. MTGNN (learned graph baseline)
