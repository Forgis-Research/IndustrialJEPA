#!/usr/bin/env python3
"""
Autoresearch Runner

This script runs the autoresearch loop:
1. Prepare data (once)
2. Run experiment (5 min)
3. Log results
4. Repeat with Claude modifying train.py

Usage:
    # Prepare data first
    python prepare.py

    # Run single experiment
    python run.py --single

    # Run with Claude Code (overnight)
    ./start.sh "Run autoresearch loop, improve val_loss"

Manual overnight mode:
    python run.py --loop --max-iterations 50
"""

import argparse
import subprocess
import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key, value)


def run_experiment():
    """Run train.py and capture results."""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Parse results
    results = {}
    for line in result.stdout.split('\n'):
        if ':' in line and not line.startswith('='):
            try:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    results[key] = float(value)
                except ValueError:
                    results[key] = value
            except:
                pass

    results['elapsed'] = elapsed
    results['timestamp'] = datetime.now().isoformat()
    results['return_code'] = result.returncode

    return results


def log_results(results: dict, log_file: str = "experiment_log.jsonl"):
    """Append results to log file."""
    with open(log_file, 'a') as f:
        f.write(json.dumps(results) + '\n')


def get_best_result(log_file: str = "experiment_log.jsonl") -> float:
    """Get best val_loss from log."""
    best = float('inf')
    if Path(log_file).exists():
        with open(log_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if 'val_loss' in r and r['val_loss'] < best:
                        best = r['val_loss']
                except:
                    pass
    return best


def print_leaderboard(log_file: str = "experiment_log.jsonl", top_n: int = 10):
    """Print top experiments."""
    results = []
    if Path(log_file).exists():
        with open(log_file) as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except:
                    pass

    # Sort by val_loss
    results = [r for r in results if 'val_loss' in r]
    results.sort(key=lambda x: x.get('val_loss', float('inf')))

    print(f"\n{'='*60}")
    print(f"LEADERBOARD (top {top_n})")
    print(f"{'='*60}")

    for i, r in enumerate(results[:top_n], 1):
        ts = r.get('timestamp', 'unknown')[:19]
        val_loss = r.get('val_loss', float('inf'))
        elapsed = r.get('elapsed', 0)
        print(f"{i:2d}. val_loss={val_loss:.6f}  elapsed={elapsed:.0f}s  {ts}")


def main():
    parser = argparse.ArgumentParser(description="Autoresearch runner")
    parser.add_argument("--single", action="store_true",
                        help="Run single experiment")
    parser.add_argument("--loop", action="store_true",
                        help="Run loop (for manual overnight)")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Max iterations for loop mode")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Show leaderboard")
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
        return

    if args.single:
        results = run_experiment()
        log_results(results)
        print(f"\nResults: {results}")

        best = get_best_result()
        if results.get('val_loss', float('inf')) <= best:
            print(f"\n🏆 NEW BEST: {results['val_loss']:.6f}")
        else:
            print(f"\nCurrent best: {best:.6f}")

    elif args.loop:
        print(f"Running {args.max_iterations} iterations...")
        print("Note: For Claude-driven iteration, use Claude Code instead")

        for i in range(args.max_iterations):
            print(f"\n--- Iteration {i+1}/{args.max_iterations} ---")
            results = run_experiment()
            log_results(results)

            if results.get('return_code', 1) != 0:
                print("Experiment failed, stopping")
                break

            print_leaderboard(top_n=5)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
