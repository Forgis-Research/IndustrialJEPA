"""
V9 E.1: Pretrain with contiguous block masking.
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
ckpt_path = os.path.join(CKPT_DIR, 'jepa_v9_block_masking.pt')

if os.path.exists(ckpt_path):
    print(f"Checkpoint already exists: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"best_epoch={ckpt['best_epoch']}, best_val={ckpt['best_val_loss']:.4f}")
    sys.exit(0)

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

torch.manual_seed(42)
np.random.seed(42)
X_tensor = torch.from_numpy(X_compat).unsqueeze(1).float()
full_ds = TensorDataset(X_tensor)
n_val = max(100, int(len(full_ds) * 0.1))
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

model = MechanicalJEPAV8().to(DEVICE)
# Block masking override
def block_generate_mask(batch_size, device):
    n_patches = model.n_patches
    n_mask = model.n_mask
    mask_list, context_list = [], []
    for _ in range(batch_size):
        max_start = n_patches - n_mask
        start = np.random.randint(0, max(max_start, 1) + 1)
        mask_idx = list(range(start, start + n_mask))
        ctx_idx = [i for i in range(n_patches) if i not in mask_idx]
        mask_list.append(torch.tensor(mask_idx, dtype=torch.long, device=device))
        context_list.append(torch.tensor(ctx_idx, dtype=torch.long, device=device))
    return (torch.stack(mask_list), torch.stack(context_list))
model._generate_mask = block_generate_mask

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
        print(f"  [block_masking] Epoch {epoch+1}/{EPOCHS}: train={train_loss:.4f}, "
              f"val={val_loss:.4f} (best={best_val:.4f} @ep{best_epoch})")

model.load_state_dict(best_state)
torch.save({'state_dict': model.state_dict(), 'history': history,
            'best_epoch': best_epoch, 'best_val_loss': best_val}, ckpt_path)
print(f"\nBlock masking: best_epoch={best_epoch}, best_val={best_val:.4f}")
print(f"Checkpoint saved: {ckpt_path}")
