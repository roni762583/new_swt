# Feature Formulas Reference

## 🎯 Correct Position Features (6 features from micro setup)

```python
1. position_side: float
   - Current position direction
   - Formula: -1.0 (short), 0.0 (flat), 1.0 (long)

2. position_pips: float
   - Current unrealized P&L in pips
   - Formula: (current_price - entry_price) * 100  # For JPY pairs
   - Sign adjusted for position direction

3. bars_since_entry: float
   - Number of bars since position opened
   - Formula: current_bar_index - entry_bar_index
   - 0 when flat

4. pips_from_peak: float
   - Distance from maximum profit achieved
   - Formula: peak_pips - current_pips
   - Tracks drawdown from best point

5. max_drawdown_pips: float
   - Maximum drawdown experienced in position
   - Formula: max(all_drawdowns_in_position)
   - Reset on new position

6. accumulated_dd: float
   - Cumulative drawdown over time
   - Formula: sum(all_drawdowns)
   - Running total, not reset
```

## 📊 Market Features (7 optimal features discovered)

### 1. React Ratio
```python
# Normalized momentum indicator
reactive = close - SMA(close, 200)
less_reactive = SMA(close, 20) - SMA(close, 200)
react_ratio = reactive / (less_reactive + 0.0001)
react_ratio = clip(react_ratio, -5, 5)
```

### 2. H1 Trend
```python
# Higher timeframe trend direction
h1_sma20 = SMA(h1_close, 20)
if h1_close > h1_sma20 * 1.001:
    h1_trend = 1.0
elif h1_close < h1_sma20 * 0.999:
    h1_trend = -1.0
else:
    h1_trend = 0.0
```

### 3. H1 Momentum
```python
# Higher timeframe rate of change
h1_momentum = (h1_close[t] - h1_close[t-5]) / h1_close[t-5]
# This is 5-bar ROC on H1 timeframe
```

### 4. Efficiency Ratio (Kaufman's)
```python
# Measures trend efficiency (signal vs noise)
direction = abs(close[t] - close[t-10])
volatility = sum(abs(close[i] - close[i-1]) for i in range(t-9, t+1))
efficiency_ratio = direction / (volatility + 0.0001)
# Range: [0, 1] where higher = stronger trend
```

### 5. Bollinger Band Position
```python
# Position within Bollinger Bands
bb_middle = SMA(close, 20)
bb_std = STD(close, 20)
bb_upper = bb_middle + 2 * bb_std
bb_lower = bb_middle - 2 * bb_std
bb_position = (close - bb_middle) / (2 * bb_std + 0.0001)
bb_position = clip(bb_position, -1, 1)
# -1 = at lower band, 0 = at middle, 1 = at upper band
```

### 6. RSI Extreme
```python
# RSI distance from neutral (50)
delta = close.diff()
gain = where(delta > 0, delta, 0)
loss = where(delta < 0, -delta, 0)
avg_gain = EMA(gain, 14)
avg_loss = EMA(loss, 14)
rs = avg_gain / (avg_loss + 0.0001)
rsi = 100 - (100 / (1 + rs))
rsi_extreme = (rsi - 50) / 50
# Range: [-1, 1] where -1 = oversold, 1 = overbought
```

### 7. ATR Ratio (Use Mean Reversion Flag)
```python
# Volatility regime indicator
true_range = max(high - low,
                 abs(high - prev_close),
                 abs(low - prev_close))
atr = SMA(true_range, 14)
atr_ratio = atr / (SMA(close, 20) + 0.0001)

# Convert to regime flag
if efficiency_ratio < 0.3:
    use_mean_reversion = 1.0  # Range-bound market
else:
    use_mean_reversion = 0.0  # Trending market
```

## 📐 Alternative Technical Features (from micro_feature_builder.py)

These are the original micro features if you want to use them instead:

```python
1. position_in_range_60:
   range_60 = max(close[-60:]) - min(close[-60:])
   position = (close - min(close[-60:])) / range_60
   feature = tanh(position - 0.5)

2. min_max_scaled_momentum_60:
   momentum = close - close[-60]
   feature = tanh(momentum / 10.0)

3. min_max_scaled_rolling_range:
   range_60 = max(close[-60:]) - min(close[-60:])
   feature = tanh(range_60 / 10.0)

4. min_max_scaled_momentum_5:
   momentum = close - close[-5]
   feature = tanh(momentum / 10.0)

5. price_change_pips:
   change = (close - prev_close) * 100  # JPY pairs
   feature = tanh(change / 10.0)
```

## 🔄 Normalization Notes

All features should be normalized to similar ranges:
- Binary features: 0.0 or 1.0
- Directional features: -1.0 to 1.0
- Continuous features: Use tanh() or clip() to bound

## 💡 Usage in RL State

```python
def get_state():
    # Market features (7)
    market = np.array([
        react_ratio,         # -5 to 5
        h1_trend,           # -1, 0, 1
        h1_momentum,        # ~-0.05 to 0.05
        efficiency_ratio,   # 0 to 1
        bb_position,        # -1 to 1
        rsi_extreme,        # -1 to 1
        use_mean_reversion  # 0 or 1
    ])

    # Position features (6)
    position = np.array([
        position_side,      # -1, 0, 1
        position_pips / 100.0,  # Normalize
        bars_since_entry / 100.0,  # Normalize
        pips_from_peak / 100.0,  # Normalize
        max_drawdown_pips / 100.0,  # Normalize
        accumulated_dd / 1000.0  # Normalize
    ])

    return np.concatenate([market, position])  # Shape: (13,)
```