"""
Baseline forecasting methods for comparison with TabPFN-TS.
"""

import numpy as np
from typing import Optional, Any


# =============================================================================
# Detrending Utilities
# =============================================================================

class Detrender:
    """
    Remove and restore linear trends from time series.

    Use this to handle TabPFN-TS's limitation with linear trend extrapolation.

    Example:
        detrender = Detrender()
        residuals = detrender.fit_transform(y_train)
        # ... forecast residuals ...
        forecast = detrender.inverse_transform(predicted_residuals, horizon)
    """

    def __init__(self, method: str = 'linear'):
        """
        Args:
            method: 'linear' (default) or 'mean' for simple mean removal
        """
        self.method = method
        self.slope = None
        self.intercept = None
        self.mean = None
        self.n_train = None

    def fit(self, y: np.ndarray) -> 'Detrender':
        """Fit the trend to training data."""
        self.n_train = len(y)
        t = np.arange(self.n_train)

        if self.method == 'linear':
            # Fit linear trend: y = slope * t + intercept
            coeffs = np.polyfit(t, y, 1)
            self.slope = coeffs[0]
            self.intercept = coeffs[1]
        elif self.method == 'mean':
            self.mean = np.mean(y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, y: np.ndarray, t_start: int = 0) -> np.ndarray:
        """Remove trend from data."""
        t = np.arange(t_start, t_start + len(y))

        if self.method == 'linear':
            trend = self.slope * t + self.intercept
        else:
            trend = self.mean

        return y - trend

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and remove trend in one step."""
        self.fit(y)
        return self.transform(y, t_start=0)

    def inverse_transform(self, residuals: np.ndarray, t_start: int = None) -> np.ndarray:
        """Add trend back to residuals (for forecasts)."""
        if t_start is None:
            t_start = self.n_train  # Default: continue from end of training

        t = np.arange(t_start, t_start + len(residuals))

        if self.method == 'linear':
            trend = self.slope * t + self.intercept
        else:
            trend = self.mean

        return residuals + trend

    def extrapolate_trend(self, horizon: int, t_start: int = None) -> np.ndarray:
        """Get the trend values for future timesteps."""
        if t_start is None:
            t_start = self.n_train

        t = np.arange(t_start, t_start + horizon)

        if self.method == 'linear':
            return self.slope * t + self.intercept
        else:
            return np.full(horizon, self.mean)


class DetrendedForecaster:
    """
    Wrapper that applies detrending before forecasting.

    This addresses TabPFN-TS's limitation with linear trend extrapolation by:
    1. Removing the trend from training data
    2. Forecasting the residuals
    3. Adding the extrapolated trend back to the forecast

    Example:
        from tabpfn_time_series import TabPFNForecaster

        base_forecaster = TabPFNForecaster(horizon=20)
        forecaster = DetrendedForecaster(base_forecaster, method='linear')
        forecaster.fit(y_train)
        predictions = forecaster.predict(horizon=20)
    """

    def __init__(self, base_forecaster: Any, method: str = 'linear'):
        """
        Args:
            base_forecaster: Any forecaster with fit(y) and predict(horizon) methods
            method: Detrending method ('linear' or 'mean')
        """
        self.base_forecaster = base_forecaster
        self.detrender = Detrender(method=method)
        self.method = method

    def fit(self, y: np.ndarray, **kwargs) -> 'DetrendedForecaster':
        """Fit on detrended data."""
        # Remove trend
        residuals = self.detrender.fit_transform(y)

        # Fit base forecaster on residuals
        self.base_forecaster.fit(residuals, **kwargs)

        return self

    def predict(self, horizon: int = None, **kwargs) -> np.ndarray:
        """Predict and add trend back."""
        # Get residual forecast from base model
        if horizon is not None:
            residual_forecast = self.base_forecaster.predict(horizon, **kwargs)
        else:
            residual_forecast = self.base_forecaster.predict(**kwargs)

        # Add extrapolated trend
        return self.detrender.inverse_transform(residual_forecast)


# =============================================================================
# Basic Forecasters
# =============================================================================

class NaiveForecaster:
    """Last-value naive baseline."""

    def __init__(self):
        self.last_value = None

    def fit(self, y):
        self.last_value = y[-1]
        return self

    def predict(self, horizon: int):
        return np.full(horizon, self.last_value)


class SeasonalNaiveForecaster:
    """Seasonal naive baseline - repeats last period."""

    def __init__(self, period: int = 20):
        self.period = period
        self.pattern = None

    def fit(self, y):
        self.pattern = y[-self.period:]
        return self

    def predict(self, horizon: int):
        n_repeats = horizon // self.period + 1
        return np.tile(self.pattern, n_repeats)[:horizon]


class MovingAverageForecaster:
    """Moving average baseline."""

    def __init__(self, window: int = 20):
        self.window = window
        self.mean_value = None

    def fit(self, y):
        self.mean_value = np.mean(y[-self.window:])
        return self

    def predict(self, horizon: int):
        return np.full(horizon, self.mean_value)


class LinearTrendForecaster:
    """Linear extrapolation baseline."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.intercept = None
        self.slope = None

    def fit(self, y):
        recent = y[-self.lookback:]
        t = np.arange(len(recent))
        self.slope = np.polyfit(t, recent, 1)[0]
        self.intercept = recent[-1]
        return self

    def predict(self, horizon: int):
        return self.intercept + self.slope * np.arange(1, horizon + 1)


class ARIMAForecaster:
    """ARIMA baseline (requires statsmodels)."""

    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None

    def fit(self, y):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(y, order=self.order).fit()
        except ImportError:
            print("statsmodels not installed. Run: pip install statsmodels")
            self.model = None
        except Exception as e:
            print(f"ARIMA fitting error: {e}")
            self.model = None
        return self

    def predict(self, horizon: int):
        if self.model is None:
            return np.full(horizon, np.nan)
        return self.model.forecast(steps=horizon)


def get_all_baselines(period: Optional[int] = 20) -> dict:
    """Get dictionary of all baseline forecasters."""
    return {
        'Naive': NaiveForecaster(),
        'Seasonal Naive': SeasonalNaiveForecaster(period=period),
        'Moving Average': MovingAverageForecaster(window=period),
        'Linear Trend': LinearTrendForecaster(lookback=period),
        'ARIMA': ARIMAForecaster(),
    }


class TabPFNForecaster:
    """
    Simple wrapper around TabPFNTSPipeline with array-based API.

    This provides a simpler interface that matches common forecasting APIs:
        forecaster = TabPFNForecaster(horizon=20)
        forecaster.fit(y_train)
        predictions = forecaster.predict()

    Internally converts numpy arrays to the DataFrame format required by TabPFNTSPipeline.

    Note: Uses TabPFNMode.CLIENT by default (cloud API, no GPU required).
    """

    def __init__(self, horizon: int = 10, use_local: bool = False):
        """
        Args:
            horizon: Number of steps to forecast
            use_local: If True, use local GPU inference. If False (default), use cloud API.
        """
        self.horizon = horizon
        self.use_local = use_local
        self.pipeline = None
        self.context_df = None
        self._predictions_df = None

    def fit(self, y: np.ndarray, X: np.ndarray = None) -> 'TabPFNForecaster':
        """
        Fit the forecaster on training data.

        Args:
            y: Target time series (1D array)
            X: Optional covariates (2D array with shape [n_samples, n_features])
        """
        try:
            from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode
            import pandas as pd

            mode = TabPFNMode.LOCAL if self.use_local else TabPFNMode.CLIENT
            self.pipeline = TabPFNTSPipeline(tabpfn_mode=mode)

            # Create context DataFrame with required columns
            # item_id is optional for single time series
            self.context_df = pd.DataFrame({
                'item_id': ['series_0'] * len(y),
                'timestamp': pd.date_range('2024-01-01', periods=len(y), freq='s'),
                'target': y
            })

            # Add covariates if provided
            if X is not None:
                for i in range(X.shape[1]):
                    self.context_df[f'covariate_{i}'] = X[:, i]

            self._y_train = y
            self._X_train = X

        except ImportError:
            raise ImportError("TabPFN-TS not installed. Run: pip install tabpfn-time-series")

        return self

    def predict(self, horizon: int = None, X: np.ndarray = None) -> np.ndarray:
        """
        Generate forecasts.

        Args:
            horizon: Forecast horizon (uses self.horizon if not provided)
            X: Future covariates (required if covariates were used in fit)

        Returns:
            Numpy array of predictions
        """
        import pandas as pd

        if self.pipeline is None:
            raise ValueError("Must call fit() before predict()")

        h = horizon if horizon is not None else self.horizon

        if X is not None:
            # Create future DataFrame with covariates
            last_timestamp = self.context_df['timestamp'].iloc[-1]
            future_df = pd.DataFrame({
                'item_id': ['series_0'] * h,
                'timestamp': pd.date_range(last_timestamp + pd.Timedelta(seconds=1),
                                           periods=h, freq='s'),
            })
            for i in range(X.shape[1]):
                future_df[f'covariate_{i}'] = X[:, i]

            self._predictions_df = self.pipeline.predict_df(
                context_df=self.context_df, future_df=future_df
            )
        else:
            self._predictions_df = self.pipeline.predict_df(
                context_df=self.context_df, prediction_length=h
            )

        # Extract target column (point predictions)
        if 'target' in self._predictions_df.columns:
            return self._predictions_df['target'].values
        elif '0.5' in self._predictions_df.columns:
            return self._predictions_df['0.5'].values
        else:
            # Return first numeric column
            return self._predictions_df.iloc[:, 0].values

    def predict_quantiles(self, quantiles: list = None) -> np.ndarray:
        """
        Get quantile predictions from the last predict() call.

        Args:
            quantiles: List of quantiles to return (default: [0.1, 0.5, 0.9])

        Returns:
            Array of shape [horizon, n_quantiles]
        """
        if self._predictions_df is None:
            raise ValueError("Must call predict() before predict_quantiles()")

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        result = []
        for q in quantiles:
            col = str(q)
            if col in self._predictions_df.columns:
                result.append(self._predictions_df[col].values)
            elif q == 0.5 and 'target' in self._predictions_df.columns:
                result.append(self._predictions_df['target'].values)
            else:
                result.append(np.full(len(self._predictions_df), np.nan))

        return np.column_stack(result)


def get_tabpfn_forecaster(horizon: int, detrend: bool = False):
    """
    Get TabPFN-TS forecaster, optionally with detrending.

    Args:
        horizon: Forecast horizon
        detrend: If True, wrap with DetrendedForecaster to handle linear trends

    Returns:
        Forecaster instance or None if TabPFN-TS not installed
    """
    try:
        forecaster = TabPFNForecaster(horizon=horizon)

        if detrend:
            return DetrendedForecaster(forecaster, method='linear')
        return forecaster

    except ImportError:
        print("TabPFN-TS not installed. Run: pip install tabpfn-time-series")
        return None
