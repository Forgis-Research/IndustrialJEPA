"""
Verify environment setup before starting overnight run.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from IndustrialJEPA root (two levels up from scripts/)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)
print(f"Loading .env from: {env_path}")

def check_hf_token():
    """Verify HuggingFace token is set and valid."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[FAIL] HF_TOKEN not found in environment or .env file")
        return False

    if token.startswith("hf_xxx"):
        print("[FAIL] HF_TOKEN is still the placeholder - add your real token to .env")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"[OK] HuggingFace authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"[FAIL] HuggingFace authentication failed: {e}")
        return False


def check_dataset_access():
    """Verify we can access the target dataset."""
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_DATASET_REPO", "Forgis/Mechanical-Components")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.dataset_info(repo_id)
        print(f"[OK] Dataset accessible: {repo_id}")
        print(f"   Last modified: {info.last_modified}")
        return True
    except Exception as e:
        print(f"[FAIL] Cannot access dataset {repo_id}: {e}")
        return False


def check_disk_space():
    """Check available disk space."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"[OK] Disk space: {free_gb:.1f} GB free")
    if free_gb < 15:
        print("[WARN]  Warning: Less than 15GB free - be careful with downloads")
    return True


def check_dependencies():
    """Check required Python packages."""
    required = ["huggingface_hub", "datasets", "scipy", "pandas", "numpy", "tqdm", "requests"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[FAIL] Missing packages: {missing}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False

    print(f"[OK] All {len(required)} required packages installed")
    return True


def main():
    print("=" * 50)
    print("Mechanical Datasets - Setup Verification")
    print("=" * 50)
    print()

    checks = [
        ("Dependencies", check_dependencies),
        ("Disk Space", check_disk_space),
        ("HuggingFace Token", check_hf_token),
        ("Dataset Access", check_dataset_access),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        results.append(check_fn())

    print("\n" + "=" * 50)
    if all(results):
        print("[OK] All checks passed - ready for overnight run!")
    else:
        print("[FAIL] Some checks failed - fix issues before starting")
        sys.exit(1)


if __name__ == "__main__":
    main()
