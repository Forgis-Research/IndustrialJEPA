"""
RUL Training Loop — downstream task after JEPA pretraining.

Usage:
  python downstream/rul/train.py --encoder checkpoints/jepa_v9_compatible_6.pt

Run from: /home/sagemaker-user/IndustrialJEPA/mechanical-jepa/
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v9/results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str,
                        default='checkpoints/jepa_v9_compatible_6.pt',
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--head', type=str, default='lstm',
                        choices=['lstm', 'tcn_transformer', 'probabilistic'],
                        help='Temporal head architecture')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--include-hc', action='store_true',
                        help='Include handcrafted features')
    parser.add_argument('--include-deviation', action='store_true',
                        help='Include deviation-from-baseline features')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name for saving results')
    args = parser.parse_args()

    # Load encoder
    from jepa_v8 import MechanicalJEPAV8
    model = MechanicalJEPAV8().to(DEVICE)
    if os.path.exists(args.encoder):
        ckpt = torch.load(args.encoder, map_location=DEVICE)
        state = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state)
        print(f"Loaded encoder from {args.encoder}")
    else:
        print(f"Warning: encoder not found at {args.encoder}, using random initialization")
    model.eval()

    # Load episodes
    from experiments.v9.run_experiments import (load_rul_episodes_all,
                                                  episode_train_test_split,
                                                  run_rul_experiment,
                                                  LSTMHead, TCNTransformerHead,
                                                  ProbabilisticLSTMHead)
    from data_pipeline import episode_train_test_split as v8_split

    episodes = load_rul_episodes_all(verbose=True)
    train_eps, test_eps = episode_train_test_split(episodes, seed=42, verbose=True)

    # Build head
    z_dim = 256
    dev_extra = (256 + 1) if args.include_deviation else 0
    hc_dim = 18 if args.include_hc else 0
    input_dim = z_dim + dev_extra + hc_dim + 2  # +2 for elapsed_t + delta_t

    if args.head == 'lstm':
        head_class = LSTMHead
        head_kwargs = {'input_dim': input_dim, 'hidden_size': 256}
    elif args.head == 'tcn_transformer':
        head_class = TCNTransformerHead
        head_kwargs = {'input_dim': input_dim, 'hidden': 64}
    elif args.head == 'probabilistic':
        head_class = ProbabilisticLSTMHead
        head_kwargs = {'input_dim': input_dim, 'hidden_size': 256}
    else:
        raise ValueError(f"Unknown head: {args.head}")

    exp_name = args.name or f'{args.head}_{"hc" if args.include_hc else ""}{"dev" if args.include_deviation else ""}'

    print(f"\nRunning: {exp_name}, input_dim={input_dim}, {args.n_seeds} seeds")

    mean_rmse, std_rmse, rmses = run_rul_experiment(
        exp_name, model, episodes, train_eps, test_eps,
        head_class, head_kwargs,
        n_seeds=args.n_seeds,
        include_handcrafted=args.include_hc,
        include_deviation=args.include_deviation,
        probabilistic=(args.head == 'probabilistic'),
        epochs=args.epochs)

    print(f"\nResult: RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
    result = {
        'name': exp_name,
        'encoder': args.encoder,
        'head': args.head,
        'include_hc': args.include_hc,
        'include_deviation': args.include_deviation,
        'rmse_mean': mean_rmse,
        'rmse_std': std_rmse,
        'rmses': rmses,
        'n_seeds': args.n_seeds,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f'{exp_name}.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
