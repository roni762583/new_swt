"""
Pytest configuration and shared fixtures for SWT tests
"""
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV price data for testing"""
    np.random.seed(42)
    n_bars = 360  # 6 hours of minute data

    # Generate realistic GBPJPY price data
    base_price = 185.0
    prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.01)

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01 00:00', periods=n_bars, freq='1min'),
        'open': prices + np.random.randn(n_bars) * 0.001,
        'high': prices + np.abs(np.random.randn(n_bars) * 0.002),
        'low': prices - np.abs(np.random.randn(n_bars) * 0.002),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_bars)
    })

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_position_state():
    """Generate sample position state for testing"""
    return {
        'has_position': True,
        'is_long': True,
        'is_short': False,
        'entry_price': 185.50,
        'current_price': 185.75,
        'bars_since_entry': 30,
        'current_equity_pips': 25.0,
        'max_drawdown_pips': -10.0,
        'pips_from_peak': -5.0,
        'position_efficiency': 0.6
    }


@pytest.fixture
def sample_wst_window():
    """Generate sample 256-bar price window for WST"""
    np.random.seed(42)
    base_price = 185.0
    prices = base_price + np.cumsum(np.random.randn(256) * 0.01)
    return prices.astype(np.float32)


@pytest.fixture
def device():
    """Get available device (CPU/CUDA)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')