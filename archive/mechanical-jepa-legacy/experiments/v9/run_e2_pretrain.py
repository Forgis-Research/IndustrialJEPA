"""
V9 E.2: Pretrain with dual-channel raw+FFT.
Separate script to avoid OOM — pretraining windows only.
"""
import os, sys, json, copy, math, numpy as np, torch
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
from torch.utils.data import DataLoader, TensorDataset, random_split
from jepa_v8 import MechanicalJEPAV8
from data_pipeline import load_pretrain_windows
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/checkpoints'
ckpt_path = os.path.join(CKPT_DIR, 'jepa_v9_dual_channel.pt')

if os.path.exists(ckpt_path):
    print(f"Checkpoint already exists: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"best_epoch={ckpt['best_epoch']}, best_val={ckpt['best_val_loss']:.4f}")
    sys.exit(0)

WINDOW_LEN = 1024

def make_dual_channel_windows(X_raw):
    N = len(X_raw)
    X_dual = np.zeros((N, 2, WINDOW_LEN), dtype=np.float32)
    for i, w in enumerate(X_raw):
        X_dual[i, 0] = w
        fft_mag = np.abs(np.fft.rfft(w))
        fft_512 = fft_mag[:512]
        fft_std = fft_512.std()
        if fft_std > 1e-8:
            fft_norm = (fft_512 - fft_512.mean()) / fft_std
        else:
            fft_norm = fft_512
        fft_padded = np.concatenate([fft_norm, fft_norm[::-1]])
        X_dual[i, 1] = fft_padded
    return X_dual

def get_cosine_lr(epoch, max_lr, epochs, warmup):
    if epoch < warmup:
        return max_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

print("Loading compatible_6 windows...")
X, sources = load_pretrain_windows(verbose=False)
compat = ['cwru', 'femto', 'xjtu_sy', 'ims', 'paderborn', 'ottawa_bearing']
mask = np.array([s in compat or s.replace('ottawa_bearing', 'ottawa') in compat for s in sources])
X_compat = X[mask]
print(f"Compatible windows: {len(X_compat)}")

print("Creating dual-channel (raw+FFT)...")
X_dual = make_dual_channel_windows(X_compat)
print(f"Dual-channel shape: {X_dual.shape}")

torch.manual_seed(42)
np.random.seed(42)
X_tensor = torch.from_numpy(X_dual).float()  # (N, 2, 1024) — no unsqueeze
full_ds = TensorDataset(X_tensor)
n_val = max(100, int(len(full_ds) * 0.1))
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

model = MechanicalJEPAV8(n_channels=2).to(DEVICE)
print(f"Dual-channel model: n_channels=2, n_patches={model.n_patches}, n_mask={model.n_mask}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
history = {'train_loss': [], 'val_loss': [], 'val_var': []}
best_val = float('inf')
best_epoch = 0
best_state = None
EPOCHS = 100

for epoch in range(EPOCHS):
    current_lr = get_cosine_lr(epoch, 1e-4, EPOCHS, 5)
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr
    model.train()
    train_losses = []
    for batch in train_loader:
        x = batch[0].to(DEVICE)
        loss, preds, _ = model(x)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_ema()
        train_losses.append(loss.item())
    model.eval()
    val_losses, val_vars = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(DEVICE)
            loss, preds, _ = model(x)
            val_losses.append(loss.item())
            val_vars.append(preds.var(dim=1).mean().item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    val_var = np.mean(val_vars)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_var'].append(val_var)
    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())
    if (epoch + 1) % 20 == 0:
        print(f"  [dual_channel] Epoch {epoch+1}/{EPOCHS}: train={train_loss:.4f}, "
              f"val={val_loss:.4f} (best={best_val:.4f} @ep{best_epoch})")

model.load_state_dict(best_state)
torch.save({'state_dict': model.state_dict(), 'history': history,
            'best_epoch': best_epoch, 'best_val_loss': best_val, 'n_channels': 2}, ckpt_path)
print(f"\nDual-channel: best_epoch={best_epoch}, best_val={best_val:.4f}")
print(f"Checkpoint saved: {ckpt_path}")
