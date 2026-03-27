"""
Experiment 01: TabPFN-TS on UCI Hydraulic System
================================================

Quick assessment of TabPFN-TS forecasting capability on hydraulic pressure sensors.

Usage:
    python exp01_hydraulic.py [--synthetic] [--cycles N] [--detrend]

Options:
    --synthetic  Use synthetic data instead of real hydraulic data
    --cycles N   Number of cycles to test (default: 5)
    --detrend    Include detrended TabPFN-TS (handles linear trends better)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import sys

# Add experiments directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Optional: suppress warnings
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_hydraulic(n_timesteps=600, n_cycles=5, add_trend=False):
    """Generate synthetic hydraulic-like pressure data.

    Args:
        n_timesteps: Number of timesteps per cycle
        n_cycles: Number of cycles to generate
        add_trend: If True, add linear degradation trend (tests detrending)
    """
    all_data = []
    for i in range(n_cycles):
        t = np.linspace(0, 60, n_timesteps)
        # Pressure-like signal
        base = 100 + 20 * np.sin(2 * np.pi * 0.05 * t)
        transient = 30 * np.exp(-0.5 * t) * (t < 10)
        noise = 2 * np.random.randn(n_timesteps)

        signal = base + transient + noise

        # Add linear degradation trend if requested
        # This simulates sensor drift or system degradation
        if add_trend:
            trend = -0.5 * t  # Decreasing trend (degradation)
            signal = signal + trend

        all_data.append(signal)
    return np.array(all_data)


def load_hydraulic_data(data_dir, sensor='PS1', n_cycles=None):
    """Load real hydraulic data if available."""
    file_path = data_dir / f'{sensor}.txt'
    if not file_path.exists():
        return None

    data = np.loadtxt(file_path)
    if n_cycles is not None:
        data = data[:n_cycles]
    return data


def naive_baseline(y_train, horizon):
    """Last value naive baseline."""
    return np.full(horizon, y_train[-1])


def seasonal_naive(y_train, horizon, period=20):
    """Seasonal naive baseline."""
    pattern = y_train[-period:]
    n_repeats = horizon // period + 1
    return np.tile(pattern, n_repeats)[:horizon]


def moving_average(y_train, horizon, window=20):
    """Moving average baseline."""
    return np.full(horizon, np.mean(y_train[-window:]))


def evaluate_forecasts(y_true, predictions_dict):
    """Calculate metrics for all methods."""
    results = {}
    for name, pred in predictions_dict.items():
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mae = mean_absolute_error(y_true, pred)
        results[name] = {'RMSE': rmse, 'MAE': mae}
    return results


def run_experiment(data, subsample=10, train_frac=0.8, include_detrend=False):
    """Run forecasting experiment on a single cycle.

    Args:
        data: Time series data for one cycle
        subsample: Subsampling factor for speed
        train_frac: Fraction of data for training
        include_detrend: If True, also test detrended TabPFN-TS
    """
    from baselines import Detrender, DetrendedForecaster

    # Subsample for speed
    y = data[::subsample]
    n = len(y)

    # Train/test split
    train_size = int(train_frac * n)
    y_train = y[:train_size]
    y_test = y[train_size:]
    horizon = len(y_test)

    if horizon == 0:
        return None

    predictions = {}

    # Baselines
    predictions['Naive (last)'] = naive_baseline(y_train, horizon)
    predictions['Seasonal naive'] = seasonal_naive(y_train, horizon)
    predictions['Moving avg'] = moving_average(y_train, horizon)

    # TabPFN-TS (standard)
    try:
        from tabpfn_ts import TabPFNForecaster
        forecaster = TabPFNForecaster(horizon=horizon)
        forecaster.fit(y_train)
        predictions['TabPFN-TS'] = forecaster.predict()
    except ImportError:
        print("TabPFN-TS not installed. Run: pip install tabpfn-time-series")
        predictions['TabPFN-TS'] = np.full(horizon, np.nan)
    except Exception as e:
        print(f"TabPFN-TS error: {e}")
        predictions['TabPFN-TS'] = np.full(horizon, np.nan)

    # TabPFN-TS with detrending (handles linear trends better)
    if include_detrend:
        try:
            from tabpfn_ts import TabPFNForecaster
            base_forecaster = TabPFNForecaster(horizon=horizon)
            detrended_forecaster = DetrendedForecaster(base_forecaster, method='linear')
            detrended_forecaster.fit(y_train)
            predictions['TabPFN-TS (detrend)'] = detrended_forecaster.predict(horizon)
        except ImportError:
            predictions['TabPFN-TS (detrend)'] = np.full(horizon, np.nan)
        except Exception as e:
            print(f"TabPFN-TS detrend error: {e}")
            predictions['TabPFN-TS (detrend)'] = np.full(horizon, np.nan)

    # Evaluate
    results = evaluate_forecasts(y_test, predictions)

    return {
        'y_train': y_train,
        'y_test': y_test,
        'predictions': predictions,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='TabPFN-TS Hydraulic Experiment')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--cycles', type=int, default=5, help='Number of cycles to test')
    parser.add_argument('--detrend', action='store_true',
                        help='Include detrended TabPFN-TS (better for trends)')
    parser.add_argument('--trend', action='store_true',
                        help='Add degradation trend to synthetic data (use with --detrend)')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'datasets' / 'data' / 'hydraulic'

    # Load or generate data
    if args.synthetic or not data_dir.exists():
        print("Using synthetic hydraulic data...")
        if args.trend:
            print("  -> Adding linear degradation trend (use --detrend to handle it)")
        data = generate_synthetic_hydraulic(n_cycles=args.cycles, add_trend=args.trend)
    else:
        print(f"Loading real hydraulic data from {data_dir}...")
        data = load_hydraulic_data(data_dir, n_cycles=args.cycles)
        if data is None:
            print("Could not load data. Using synthetic...")
            data = generate_synthetic_hydraulic(n_cycles=args.cycles)

    print(f"Data shape: {data.shape}")
    print(f"Running experiment on {len(data)} cycles...\n")

    # Run experiments
    all_results = []
    for i, cycle_data in enumerate(data):
        result = run_experiment(cycle_data, include_detrend=args.detrend)
        if result is not None:
            all_results.append(result)
            print(f"Cycle {i}: ", end='')
            for name, metrics in result['results'].items():
                print(f"{name}={metrics['RMSE']:.3f} ", end='')
            print()

    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATE RESULTS (mean RMSE across cycles)")
    print("="*60)

    methods = list(all_results[0]['results'].keys())
    for method in methods:
        rmses = [r['results'][method]['RMSE'] for r in all_results]
        rmses = [x for x in rmses if not np.isnan(x)]
        if rmses:
            print(f"{method:20s}: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # Skill scores vs naive
    print("\n" + "="*60)
    print("SKILL SCORES (vs Naive)")
    print("="*60)

    naive_rmses = [r['results']['Naive (last)']['RMSE'] for r in all_results]
    for method in methods:
        if method == 'Naive (last)':
            continue
        method_rmses = [r['results'][method]['RMSE'] for r in all_results]
        skills = []
        for nr, mr in zip(naive_rmses, method_rmses):
            if not np.isnan(mr) and nr > 0:
                skills.append(1 - mr / nr)
        if skills:
            print(f"{method:20s}: {np.mean(skills):.3f} ± {np.std(skills):.3f}")

    # Plot last cycle
    if all_results:
        last = all_results[-1]
        n_train = len(last['y_train'])
        t_all = np.arange(n_train + len(last['y_test']))

        plt.figure(figsize=(12, 5))
        plt.plot(t_all[:n_train], last['y_train'], 'b-', label='Training', linewidth=0.8)
        plt.plot(t_all[n_train:], last['y_test'], 'g-', label='True', linewidth=1.5)

        for name, pred in last['predictions'].items():
            if not np.any(np.isnan(pred)):
                plt.plot(t_all[n_train:], pred, '--', label=name, linewidth=1.2)

        plt.axvline(x=n_train, color='gray', linestyle=':', alpha=0.7)
        plt.xlabel('Time (samples)')
        plt.ylabel('Pressure (bar)')
        plt.title('Hydraulic Pressure Forecasting - Last Cycle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(__file__).parent / 'exp01_results.png'
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")
        plt.show()


if __name__ == '__main__':
    main()
