"""Test MOMENT embedding extraction."""
import sys
print(f"Python: {sys.version}")

from momentfm import MOMENTPipeline
import torch
import numpy as np

print("Loading MOMENT-1-large...")
model = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large', model_kwargs={'task_name': 'embedding'})
model.eval()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"MOMENT params: {n_params:.1f}M")
print(f"seq_len: {model.seq_len}, patch_len: {model.patch_len}")

# Test embed - MOMENT expects (batch, n_channels, 512)
x = torch.randn(4, 14, 512)  # FD001 has 14 sensors
with torch.no_grad():
    out = model.embed(x_enc=x, reduction='mean')
print(f"Output type: {type(out)}")
if hasattr(out, 'embeddings'):
    print(f"Embeddings shape: {out.embeddings.shape}")
elif isinstance(out, torch.Tensor):
    print(f"Tensor shape: {out.shape}")
else:
    print(f"Output: {out}")

print("MOMENT test PASSED")
