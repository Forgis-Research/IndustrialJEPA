"""
V9 E.1: Contiguous Block Masking — Memory-Efficient Version.
Loads only small parquet files (cwru, ims, femto, xjtu_sy from small shards).
Skips ottawa and paderborn to avoid loading the 462MB and 1.4GB parquet files.
Uses bearing_rul_3 sources: femto, xjtu_sy (shards 0-3), ims + cwru = ~4 sources.

This is a pragmatic adaptation to RAM constraints (15GB with only ~11GB free).
The experiment still tests block masking vs random masking on the same sources.
"""
import os, sys, json, copy, math, numpy as np, torch
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
from torch.utils.data import DataLoader, TensorDataset, random_split
from jepa_v8 import MechanicalJEPAV8
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/checkpoints'
CACHE_DIR = '/tmp/hf_cache/bearings'
TARGET_SR = 12800
WINDOW_LEN = 1024


def resample(signal, native_sr):
    from scipy.signal import resample_poly
    from math import gcd
    if native_sr == TARGET_SR:
        return signal.astype(np.float32)
    g = gcd(int(native_sr), int(TARGET_SR))
    up = TARGET_SR // g
    down = int(native_sr) // g
    return resample_poly(signal.astype(np.float32), up, down)


def instance_norm(w):
    std = w.std()
    if std < 1e-8:
        return None
    return ((w - w.mean()) / std).astype(np.float32)


def extract_windows(ch, max_windows=10):
    wins = []
    for i in range(min(max_windows, len(ch) // WINDOW_LEN)):
        w = ch[i * WINDOW_LEN: (i + 1) * WINDOW_LEN]
        n = instance_norm(w)
        if n is not None:
            wins.append(n)
    return wins


def load_compact_windows():
    """Load compatible pretraining windows without loading large parquet files."""
    all_windows, all_sources = [], []

    # 1. CWRU (small file: extra_cwru_mfpt.parquet ~136MB)
    print("  Loading cwru...")
    df = pd.read_parquet(os.path.join(CACHE_DIR, 'extra_cwru_mfpt.parquet'))
    for _, row in df[df['source_id'] == 'cwru'].iterrows():
        try:
            sig = np.array(row['signal'])
            ch = np.array(sig[0] if hasattr(sig[0], '__len__') else sig, dtype=np.float32)
        except Exception:
            continue
        if len(ch) < WINDOW_LEN:
            continue
        sr = int(row.get('sampling_rate', 12000))
        ch = resample(ch, sr)
        wins = extract_windows(ch, max_windows=15)
        all_windows.extend(wins)
        all_sources.extend(['cwru'] * len(wins))
    del df
    print(f"    cwru: {sum(1 for s in all_sources if s == 'cwru')} windows")

    # 2. IMS (extra_ims.parquet ~261MB)
    print("  Loading ims...")
    df = pd.read_parquet(os.path.join(CACHE_DIR, 'extra_ims.parquet'))
    n_ims = 0
    for _, row in df.iterrows():
        try:
            sig = np.array(row['signal'])
            ch = np.array(sig[0] if hasattr(sig[0], '__len__') else sig, dtype=np.float32)
        except Exception:
            continue
        if len(ch) < WINDOW_LEN:
            continue
        sr = int(row.get('sampling_rate', 20480))
        ch = resample(ch, sr)
        wins = extract_windows(ch, max_windows=10)
        all_windows.extend(wins)
        all_sources.extend(['ims'] * len(wins))
        n_ims += len(wins)
    del df
    print(f"    ims: {n_ims} windows")

    # 3. FEMTO + XJTU-SY from small shards (train-00000..00003, 10-11MB each)
    for shard_idx, shard_file in enumerate([
        'train-00000-of-00005.parquet',  # ~9.6MB
        'train-00001-of-00005.parquet',  # ~10MB
        'train-00002-of-00005.parquet',  # ~9.2MB
        'train-00003-of-00005.parquet',  # ~1.1GB!! — skip
    ]):
        if shard_idx == 3:
            print(f"  Skipping shard 3 (1.1GB) — too large")
            continue
        p = os.path.join(CACHE_DIR, shard_file)
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        for src in ['femto', 'xjtu_sy']:
            sub = df[df['source_id'] == src]
            n_src = 0
            for _, row in sub.iterrows():
                try:
                    sig = np.array(row['signal'])
                    ch = np.array(sig[0] if hasattr(sig[0], '__len__') else sig, dtype=np.float32)
                except Exception:
                    continue
                sr = int(row.get('sampling_rate', 25600))
                ch = resample(ch, sr)
                if len(ch) >= WINDOW_LEN:
                    wins = extract_windows(ch, max_windows=10)
                elif len(ch) >= 512:
                    padded = np.pad(ch, (0, WINDOW_LEN - len(ch)), mode='wrap')
                    n = instance_norm(padded)
                    wins = [n] if n is not None else []
                else:
                    wins = []
                all_windows.extend(wins)
                all_sources.extend([src] * len(wins))
                n_src += len(wins)
            if n_src > 0:
                print(f"    {src} (shard {shard_idx}): {n_src} windows")
        del df

    if not all_windows:
        raise RuntimeError("No windows loaded!")

    X = np.stack(all_windows, axis=0)
    print(f"  Total: {len(X)} windows, {X.nbytes / 1e6:.0f} MB")
    return X, all_sources


def get_cosine_lr(epoch, max_lr, epochs, warmup):
    if epoch < warmup:
        return max_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def pretrain_block(X, name, ckpt_path, epochs=100, seed=42):
    print(f"\nPretraining {name} ({len(X)} windows)...")
    torch.manual_seed(seed); np.random.seed(seed)

    X_tensor = torch.from_numpy(X).unsqueeze(1).float()
    full_ds = TensorDataset(X_tensor)
    n_val = max(100, int(len(full_ds) * 0.1))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = MechanicalJEPAV8().to(DEVICE)

    if 'block' in name:
        def block_mask(batch_size, device):
            n_patches, n_mask = model.n_patches, model.n_mask
            ml, cl = [], []
            for _ in range(batch_size):
                start = np.random.randint(0, max(n_patches - n_mask, 1) + 1)
                mi = list(range(start, start + n_mask))
                ci = [i for i in range(n_patches) if i not in mi]
                ml.append(torch.tensor(mi, dtype=torch.long, device=device))
                cl.append(torch.tensor(ci, dtype=torch.long, device=device))
            return torch.stack(ml), torch.stack(cl)
        model._generate_mask = block_mask

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history = {'train_loss': [], 'val_loss': []}
    best_val, best_epoch, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        current_lr = get_cosine_lr(epoch, 1e-4, epochs, 5)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
        model.train()
        tl = []
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            loss, _, _ = model(x)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); model.update_ema()
            tl.append(loss.item())
        model.eval()
        vl = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                loss, _, _ = model(x)
                vl.append(loss.item())
        train_loss, val_loss = np.mean(tl), np.mean(vl)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 20 == 0:
            print(f"  [{name}] Epoch {epoch+1}/{epochs}: train={train_loss:.4f}, "
                  f"val={val_loss:.4f} (best={best_val:.4f} @ep{best_epoch})")

    model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(), 'history': history,
                'best_epoch': best_epoch, 'best_val_loss': best_val,
                'note': f'Trained on cwru+ims+femto+xjtu_sy (small shards only)'}, ckpt_path)
    print(f"\n{name}: best_epoch={best_epoch}, best_val={best_val:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    return model, history, best_epoch, best_val


if __name__ == '__main__':
    print(f"Device: {DEVICE}")

    # Block masking
    ckpt_block = os.path.join(CKPT_DIR, 'jepa_v9_block_masking.pt')
    ckpt_dual = os.path.join(CKPT_DIR, 'jepa_v9_dual_channel.pt')

    if os.path.exists(ckpt_block):
        print(f"Block masking checkpoint already exists: {ckpt_block}")
    else:
        print("Loading compact windows for pretraining...")
        X, sources = load_compact_windows()
        pretrain_block(X, 'block_masking', ckpt_block)
        del X  # free memory before dual-channel

    if os.path.exists(ckpt_dual):
        print(f"Dual-channel checkpoint already exists: {ckpt_dual}")
    else:
        print("\nLoading compact windows for dual-channel pretraining...")
        X, sources = load_compact_windows()
        # Create dual-channel windows
        print("Creating dual-channel (raw+FFT)...")
        X_dual = np.zeros((len(X), 2, WINDOW_LEN), dtype=np.float32)
        for i, w in enumerate(X):
            X_dual[i, 0] = w
            fft_mag = np.abs(np.fft.rfft(w))[:512]
            s = fft_mag.std()
            fft_norm = (fft_mag - fft_mag.mean()) / s if s > 1e-8 else fft_mag
            X_dual[i, 1] = np.concatenate([fft_norm, fft_norm[::-1]])
        del X

        torch.manual_seed(42); np.random.seed(42)
        X_tensor = torch.from_numpy(X_dual).float()
        full_ds = TensorDataset(X_tensor)
        n_val = max(100, int(len(full_ds) * 0.1))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        model = MechanicalJEPAV8(n_channels=2).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        history = {'train_loss': [], 'val_loss': []}
        best_val, best_epoch, best_state = float('inf'), 0, None
        EPOCHS = 100

        for epoch in range(EPOCHS):
            current_lr = get_cosine_lr(epoch, 1e-4, EPOCHS, 5)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            model.train()
            tl = []
            for batch in train_loader:
                x = batch[0].to(DEVICE)
                loss, _, _ = model(x)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); model.update_ema()
                tl.append(loss.item())
            model.eval()
            vl = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(DEVICE)
                    loss, _, _ = model(x)
                    vl.append(loss.item())
            tl_m, vl_m = np.mean(tl), np.mean(vl)
            history['train_loss'].append(tl_m)
            history['val_loss'].append(vl_m)
            if vl_m < best_val:
                best_val, best_epoch = vl_m, epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 20 == 0:
                print(f"  [dual_channel] Epoch {epoch+1}/{EPOCHS}: train={tl_m:.4f}, "
                      f"val={vl_m:.4f} (best={best_val:.4f} @ep{best_epoch})")

        model.load_state_dict(best_state)
        torch.save({'state_dict': model.state_dict(), 'history': history,
                    'best_epoch': best_epoch, 'best_val_loss': best_val, 'n_channels': 2,
                    'note': 'Dual-channel (raw+FFT), trained on cwru+ims+femto+xjtu_sy (small shards)'}, ckpt_dual)
        print(f"\ndual_channel: best_epoch={best_epoch}, best_val={best_val:.4f}")
        print(f"Checkpoint: {ckpt_dual}")

    print("\nAll pretraining complete.")
