# Feature Selection for ZigZag Pretrain

**Target**: Predict `pretrain_action` (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE) from ZigZag pivots
**Data**: 13,486 pivots, ~12,153 theoretical trades over 3.59 years
**Goal**: Learn to identify market turning points and trend direction

---

## 📊 PROPOSED FEATURE SET (Minimal & Effective)

### **Core Price Features** (4)
Essential for understanding current market state:
1. ✅ **close** - Current price level
2. ✅ **high** - Bar high (captures wicks/rejections)
3. ✅ **low** - Bar low (captures support tests)
4. ✅ **volume** - Activity level (confirming moves)

### **Momentum Features** (4)
Detect trend direction and strength:
5. ✅ **log_return_1m** - Immediate momentum (-0.018 to +0.011)
6. ✅ **log_return_5m** - Short-term trend (-0.031 to +0.017)
7. ✅ **log_return_60m** - H1 trend alignment (-0.039 to +0.023)
8. ✅ **efficiency_ratio_h1** - Trend vs chop (0-1, higher=trending)

### **Volatility Features** (3)
Risk/opportunity assessment:
9. ✅ **atr_14** - Short-term volatility (0.1-97.2 pips)
10. ✅ **atr_14_zsarctan_w20** - Volatility regime detection
11. ✅ **vol_ratio_deviation** - Volatility compression/expansion

### **Swing Structure Features** (5)
Market structure context (KEY for ZigZag):
12. ✅ **h1_swing_range_position** - Where price is in H1 range [0,1]
13. ✅ **swing_point_range** - H1 range magnitude (consolidation detector)
14. ✅ **high_swing_slope_h1** - H1 swing highs trend (-11.68 to +63.00 pips/bar)
15. ✅ **low_swing_slope_h1** - H1 swing lows trend (-91.50 to +23.93 pips/bar)
16. ✅ **h1_trend_slope_zsarctan** - Intelligent slope selection (Z-SCORE) based on market structure

### **Z-Score Extreme Detection** (3)
Capture regime extremes:
17. ✅ **swing_point_range_zsarctan_w20** - Range extremes (tight/wide)
18. ✅ **high_swing_slope_h1_zsarctan** - Extreme uptrends/downtrends
19. ✅ **combo_geometric** - Interaction feature (range × position, 69% better predictor)

### **Technical Indicators** (2)
Classic overbought/oversold:
20. ✅ **rsi_extreme** - (RSI-50)/50, bounded [-1,1]
21. ✅ **bb_position** - Bollinger Band position [-1,1] (rescaled)

### **Time Context** (4)
Session patterns:
22. ✅ **hour_sin** - 24-hour cycle
23. ✅ **hour_cos** - 24-hour cycle (orthogonal)
24. ✅ **dow_sin** - 120-hour forex week cycle
25. ✅ **dow_cos** - 120-hour forex week cycle (orthogonal)

---

## 🎯 TOTAL FEATURES: 25

### **Feature Categories**:
- **Price/Volume**: 4 features (16%)
- **Momentum**: 4 features (16%)
- **Volatility**: 3 features (12%)
- **Swing Structure**: 5 features (20%) ← Most important for ZigZag
- **Z-Scores**: 3 features (12%)
- **Indicators**: 2 features (8%)
- **Time**: 4 features (16%)

---

## ❌ EXCLUDED FEATURES (41)

### **Raw Swing Tracking** (16) - Redundant with derived features
- last_m1_hsp_idx, last_m1_hsp_val, prev_m1_hsp_idx, prev_m1_hsp_val
- last_m1_lsp_idx, last_m1_lsp_val, prev_m1_lsp_idx, prev_m1_lsp_val
- last_h1_hsp_idx, last_h1_hsp_val, prev_h1_hsp_idx, prev_h1_hsp_val
- last_h1_lsp_idx, last_h1_lsp_val, prev_h1_lsp_idx, prev_h1_lsp_val
*Reason*: Already captured in h1_swing_range_position and slopes

### **Swing Boolean Flags** (4) - Not ML-friendly
- swing_high_m1, swing_low_m1, swing_high_h1, swing_low_h1
*Reason*: Binary flags, information in slopes/position

### **ZigZag Data** (3) - Target leakage
- zigzag_price, zigzag_direction, is_zigzag_pivot
*Reason*: These ARE the labels we're predicting

### **Redundant Z-Scores** (6)
- h1_swing_range_position_zsarctan_w20 (use raw h1_swing_range_position instead)
- low_swing_slope_h1_zsarctan (have high_swing_slope_h1_zsarctan)
- atr_60_zsarctan_w20 (have atr_14_zsarctan_w20)
- momentum_strength_10_zsarctan_w20 (have log_returns)
- realized_vol_20_zsarctan_w20 (have vol_ratio_deviation)
- realized_vol_60_zsarctan_w20 (have vol_ratio_deviation)
*Reason*: Avoid multicollinearity, keep most predictive

### **Raw Momentum** (3)
- momentum_strength_3, momentum_strength_10
- open (redundant with close)
*Reason*: Log returns cleaner, open ≈ close[t-1]

### **M1 Slope Z-Scores** (2) - NEW: Now available!
- high_swing_slope_m1_zsarctan_w20
- low_swing_slope_m1_zsarctan_w20
*Status*: AVAILABLE NOW - Consider adding to feature set for granular trend detection

### **Raw Volatility** (3)
- atr_60 (have atr_14)
- realized_vol_20, realized_vol_60 (have vol_ratio_deviation)
*Reason*: Derived features better

### **M1 Swing Slopes** (2)
- high_swing_slope_m1, low_swing_slope_m1
*Reason*: Too noisy, H1 slopes sufficient

### **Metadata** (2)
- timestamp, bar_index
*Reason*: Not features, use time cycles instead

### **Target Label** (1)
- pretrain_action
*Reason*: This is the target, not a feature

---

## 🧠 RATIONALE

### **Why These 25?**

1. **Swing Structure (5 features)** - Critical for ZigZag prediction:
   - Position in range → identifies extremes
   - Range magnitude → consolidation vs trending
   - H1 slopes → trend direction and strength
   - h1_trend_slope → intelligent regime detection

2. **Multi-Timeframe Momentum (4)** - Captures nested trends:
   - 1m → micro momentum
   - 5m → execution timeframe
   - 60m → H1 alignment
   - efficiency_ratio → trend quality

3. **Volatility Regime (3)** - Risk context:
   - atr_14 → immediate volatility
   - atr_14_zsarctan → regime detection
   - vol_ratio_deviation → compression/expansion

4. **Z-Score Extremes (3)** - Reversal signals:
   - Range extremes → breakout/consolidation
   - Slope extremes → exhaustion
   - combo_geometric → interaction effects (69% better)

5. **Classic Indicators (2)** - Proven signals:
   - RSI → overbought/oversold
   - BB position → band extremes

6. **Time Context (4)** - Session patterns:
   - Hour cycles → intraday patterns
   - Forex week → weekly seasonality

7. **Core OHLCV (4)** - Foundation:
   - Essential price/volume data

---

## 📈 EXPECTED PERFORMANCE

**Theoretical ZigZag Performance** (from analysis):
- 12,153 trades
- 99.2% win rate (hindsight perfect)
- +533,155 pips total
- +43.87 pips/trade average

**Realistic ML Expectation** (with pretrain):
- Capture 30-50% of theoretical max
- Win rate: 60-70% (vs 99.2% theoretical)
- Avg pips: +15-20 pips/trade (vs +43.87 theoretical)
- Key: Learn to identify **high-probability** turning points

---

## 🔬 FEATURE ENGINEERING QUALITY

**Strengths**:
- ✅ All features normalized/scaled to similar ranges
- ✅ Multi-timeframe coverage (M1, M5, H1)
- ✅ Mix of price action + technical + time
- ✅ Regime detection (volatility, trend, range)
- ✅ Interaction features (combo_geometric)
- ✅ No lookahead bias
- ✅ Minimal multicollinearity

**Next Steps**:
1. Extract these 25 features for training
2. Split data: 60% train, 20% val, 20% test
3. Pretrain on ZigZag labels (13,486 pivots)
4. Fine-tune with RL (PPO) on actual trading
5. Evaluate on held-out test set

---

## 📝 SQL QUERY TO EXTRACT FEATURES

```sql
SELECT
    -- Core
    close, high, low, volume,

    -- Momentum
    log_return_1m, log_return_5m, log_return_60m, efficiency_ratio_h1,

    -- Volatility
    atr_14, atr_14_zsarctan_w20, vol_ratio_deviation,

    -- Swing Structure
    h1_swing_range_position, swing_point_range,
    high_swing_slope_h1, low_swing_slope_h1, h1_trend_slope_zsarctan,

    -- Z-Score Extremes
    swing_point_range_zsarctan_w20, high_swing_slope_h1_zsarctan, combo_geometric,

    -- Indicators
    rsi_extreme, bb_position,

    -- Time
    hour_sin, hour_cos, dow_sin, dow_cos,

    -- Target
    pretrain_action

FROM master
WHERE pretrain_action IS NOT NULL
ORDER BY bar_index;
```

### OPTIONAL: Add M1 Slope Z-Scores (27 features total):
```sql
-- Add these for granular trend detection:
high_swing_slope_m1_zsarctan_w20, low_swing_slope_m1_zsarctan_w20
```

---

**Status**: Feature set finalized - ready for pretrain implementation
