# 📊 SWT Validation Framework

## Institutional-Grade Trading System Validation

A comprehensive validation framework implementing Dr. Howard Bandy's quantitative trading methodologies for robust performance assessment.

## 📋 Latest Validation Report
**[VALIDATION_REPORT.md](./VALIDATION_REPORT.md)** - Comprehensive validation results for Episodes 10 & 775 with performance comparisons and deployment recommendations.

---

## 🎯 **Overview**

The SWT Validation Framework provides production-grade tools for evaluating trading system performance using multiple validation techniques to ensure robust, reliable deployment decisions.

### **Core Philosophy**
- **No overfitting**: Walk-forward analysis to detect curve-fitting
- **Conservative estimates**: CAR25 (25th percentile) for realistic expectations
- **Multi-factor assessment**: Beyond simple expectancy metrics
- **Statistical confidence**: Monte Carlo methods for robust validation

---

## 🔧 **Components**

### **1. Composite Scorer** (`composite_scorer.py`)
Multi-factor scoring system that balances multiple performance dimensions:

#### **Scoring Weights**
- **Expectancy**: 30% - Expected profit per trade
- **Risk-Adjusted Returns**: 30% - Sharpe/Sortino ratios
- **Consistency**: 20% - Win rate and profit factor stability
- **Drawdown Control**: 20% - Maximum drawdown management

#### **Features**
- Letter grades (A+ to F) for quick assessment
- Deployment recommendations (DEPLOY, TEST, IMPROVE, REJECT)
- Strength/weakness identification
- Checkpoint comparison and ranking

#### **Usage**
```python
from swt_validation.composite_scorer import CompositeScorer, CheckpointMetrics

# Create metrics
metrics = CheckpointMetrics(
    expectancy=0.45,
    win_rate=0.55,
    sharpe_ratio=1.8,
    max_drawdown_pct=0.15,
    # ... other metrics
)

# Calculate score
scorer = CompositeScorer()
score = scorer.calculate_composite_score(metrics)

print(f"Score: {score.total_score}/100")
print(f"Grade: {score.grade}")
print(f"Recommendation: {score.recommendation}")
```

---

### **2. Automated Validator** (`automated_validator.py`)
Intelligent validation system with automatic triggering based on performance improvements.

#### **Validation Triggers**
- **Expectancy improvement**: ≥10% improvement
- **Episode interval**: Every 100 episodes
- **Time interval**: Every 6 hours
- **Score improvement**: ≥5 point improvement
- **Best checkpoint**: Always validate new best

#### **Validation Levels**
| Level | Duration | Description |
|-------|----------|-------------|
| QUICK | ~30s | Basic metrics and scoring |
| STANDARD | ~2min | Quick backtest + composite scoring |
| FULL | ~10-30min | Monte Carlo CAR25 validation |
| COMPREHENSIVE | ~1-2hr | Full MC + Walk-forward analysis |

#### **Integration with Training**
```python
from swt_validation.automated_validator import AutomatedValidator, create_validation_callback

# Setup validator
validator = AutomatedValidator(
    data_path="data/test_data.csv",
    output_dir="validation_results"
)

# Create callback for training
callback = create_validation_callback(validator)

# Use in training loop
await callback(checkpoint_path, metrics_dict)
```

---

### **3. Monte Carlo CAR25 Validator** (`monte_carlo_car25.py`)
Implementation of Dr. Howard Bandy's CAR25 methodology for conservative performance estimation.

#### **Key Concepts**
- **CAR25**: Compound Annual Return at 25th percentile
- **Bootstrap sampling**: Random resampling for robustness
- **Monte Carlo simulation**: 1000+ runs for statistical confidence

#### **Thresholds (Dr. Bandy's Recommendations)**
- Minimum CAR25: 15% annual return
- Maximum drawdown: 25%
- Minimum profit factor: 1.5
- Minimum win rate: 40%

#### **Metrics Calculated**
- CAR25, CAR50, CAR75 (percentile returns)
- Average win rate across simulations
- Drawdown distribution
- Quality score (0-100)
- Pass/fail assessment

#### **Usage**
```python
from swt_validation.monte_carlo_car25 import MonteCarloCAR25Validator, CAR25Config

# Configure validation
config = CAR25Config(
    monte_carlo_runs=1000,
    bootstrap_sample_size=252,  # Trading days per year
    confidence_level=0.25  # 25th percentile
)

# Create validator
validator = MonteCarloCAR25Validator(config)
validator.load_checkpoint("checkpoints/best.pth")
validator.load_test_data("data/test_data.csv")

# Run validation
report = validator.run_monte_carlo_validation()
print(f"CAR25: {report['car25_metrics']['car25']:.2%}")
```

---

### **4. Walk-Forward Analyzer** (`walk_forward_analysis.py`)
Detects overfitting through systematic in-sample/out-of-sample testing.

#### **Walk-Forward Modes**
- **Rolling**: Fixed window size moving through time
- **Anchored**: In-sample always starts from beginning

#### **Key Metrics**
- **Efficiency Ratio**: Out-sample return / In-sample return
- **Robustness Score**: Overall system robustness (0-100%)
- **Consistency Score**: Performance stability across periods
- **Degradation Test**: Acceptable performance drop out-of-sample

#### **Configuration**
```python
from swt_validation.walk_forward_analysis import WalkForwardAnalyzer, WalkForwardConfig

# Configure analysis
config = WalkForwardConfig(
    total_periods=12,
    in_sample_months=6,
    out_sample_months=2,
    anchored_mode=False
)

# Run analysis
analyzer = WalkForwardAnalyzer(config)
analyzer.load_data("data/test_data.csv")
analyzer.set_checkpoint("checkpoints/best.pth")
report = analyzer.run_walk_forward_analysis()
```

---

### **5. Episode 13475 Baseline Validator** (`validate_episode_13475.py`)
Comprehensive validation specifically for Episode 13475 checkpoint to establish baseline metrics.

#### **Validation Phases**
1. **Quick Metrics**: Basic performance metrics
2. **Composite Scoring**: Multi-factor assessment
3. **Monte Carlo CAR25**: 1000 run validation
4. **Walk-Forward**: Robustness testing
5. **Baseline Report**: Comprehensive analysis

---

### **6. Robust Checkpoint Loader** (`robust_checkpoint_loader.py`)
Memory-efficient checkpoint loader with aggressive optimization for large model files.

#### **Features**
- **Memory Management**: Configurable memory limits with monitoring
- **Checkpoint Cleaning**: Removes training-only data for validation
- **Aggressive Optimization**: Reduces checkpoint size by up to 97%
- **Automatic Config Extraction**: Handles embedded architecture configs

#### **Usage**
```bash
# Standard optimization (removes basic training data)
python swt_validation/robust_checkpoint_loader.py \
  --optimize checkpoints/large.pth \
  --output checkpoints/optimized.pth

# Aggressive optimization (97% size reduction)
python swt_validation/robust_checkpoint_loader.py \
  --optimize checkpoints/episode_775.pth \
  --output checkpoints/episode_775_minimal.pth \
  --aggressive

# Load with memory limit
python swt_validation/robust_checkpoint_loader.py \
  --load checkpoints/large.pth \
  --memory-limit 2.0  # 2GB limit
```

#### **Size Reduction Results**
- Episode 775: 358MB → 9.5MB (97% reduction)
- Episode 800: 369MB → ~10MB (expected)
- Removes: optimizer states, schedulers, replay buffers, training history
- Keeps: network weights, minimal config, episode metadata

---

## 📈 **Performance Thresholds**

### **Deployment Criteria**
A checkpoint is considered deployment-ready when it meets ALL of the following:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Composite Score | ≥70/100 | Overall quality assessment |
| CAR25 | ≥15% | Conservative annual return estimate |
| Max Drawdown | ≤25% | Maximum acceptable loss |
| Expectancy | >0 | Average profit per trade (must be positive) |
| Profit Factor | ≥1.3 | Gross profit / Gross loss ratio |
| Sharpe Ratio | ≥1.0 | Risk-adjusted return measure |
| Robustness Score | ≥50% | Walk-forward robustness |
| Sample Size | ≥100 trades | Minimum statistical significance |

---

## 🚀 **Quick Start**

### **1. Validate a Single Checkpoint**
```bash
# Quick validation
python swt_validation/composite_scorer.py --checkpoint checkpoints/episode_13475.pth

# Full Monte Carlo validation
python swt_validation/monte_carlo_car25.py \
  --checkpoint checkpoints/episode_13475.pth \
  --data data/GBPJPY_M1_202201-202508.csv \
  --runs 1000
```

### **2. Automated Training Validation**
```bash
# Training with automatic validation
python training_main.py \
  --enable-validation \
  --validation-data data/GBPJPY_M1_202201-202508.csv
```

### **3. Walk-Forward Analysis**
```bash
python swt_validation/walk_forward_analysis.py \
  --checkpoint checkpoints/best.pth \
  --data data/test_data.csv \
  --periods 12 \
  --in-sample-months 6 \
  --out-sample-months 2
```

### **4. Episode 13475 Baseline**
```bash
python validate_episode_13475.py \
  --checkpoint checkpoints/episode_13475.pth \
  --data data/GBPJPY_M1_202201-202508.csv
```

---

## 📊 **Output Examples**

### **Composite Score Report**
```
COMPOSITE SCORE ANALYSIS
========================
Checkpoint: episode_13475.pth
Total Score: 78.5/100
Grade: B+

COMPONENTS:
  Expectancy: 82.0/100
  Risk-Adjusted: 75.0/100
  Consistency: 80.0/100
  Drawdown Control: 77.0/100

RECOMMENDATION: TEST - Good balanced performance, validate in paper trading
CONFIDENCE: MEDIUM

STRENGTHS:
  ✅ Excellent expectancy (0.620)
  ✅ High consistency (Win rate: 61.1%)

WEAKNESSES:
  ⚠️ Moderate drawdown risk (Max DD: 15.4%)
```

### **CAR25 Validation Report**
```
CAR25 VALIDATION SUMMARY
========================
CAR25 (Conservative): 18.5%
CAR50 (Median): 22.3%
CAR75 (Optimistic): 26.8%
Quality Score: 72.5/100
Validation: ✅ PASSED

Recommendation: RECOMMENDED - System meets all thresholds with good performance
```

---

## 🔍 **Best Practices**

### **1. Validation Frequency**
- **Quick validation**: Every checkpoint
- **Standard validation**: Every 100 episodes or 10% improvement
- **Full validation**: Before production deployment
- **Walk-forward**: Weekly or after major changes

### **2. Interpreting Results**
- **CAR25 < 10%**: System needs improvement
- **CAR25 10-15%**: Marginal, continue development
- **CAR25 15-25%**: Good, ready for paper trading
- **CAR25 > 25%**: Excellent, but verify for realism

### **3. Red Flags**
- Robustness score < 40% (likely overfitting)
- Large in-sample/out-sample performance gap
- Win rate > 70% with low profit factor
- Extremely high CAR25 (>50%) - check for errors

### **4. Understanding Risk:Reward**
- **Win rate is NOT everything**: A strategy with 40% win rate and 3:1 risk:reward is highly profitable
- **Expectancy matters most**: (Win% × Avg Win) - (Loss% × Avg Loss) must be positive
- **Example**: 45% win rate with 2:1 R:R = 0.45×2 - 0.55×1 = 0.35 positive expectancy

---

## 📚 **References**

- Bandy, H. (2011). *Quantitative Trading Systems*
- Bandy, H. (2013). *Mean Reversion Trading Systems*
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*

---

## 🛠️ **Development**

### **Adding New Validators**
1. Inherit from base validator class
2. Implement validation logic
3. Add to automated validation pipeline
4. Update thresholds as needed

### **Custom Scoring Weights**
```python
from swt_validation.composite_scorer import ScoringWeights

custom_weights = ScoringWeights(
    expectancy=0.40,  # Increase expectancy weight
    risk_adjusted_return=0.30,
    consistency=0.15,
    drawdown_control=0.15
)

scorer = CompositeScorer(weights=custom_weights)
```

---

## ⚠️ **Important Notes**

1. **Past performance does not guarantee future results**
2. **Always validate on out-of-sample data**
3. **Consider market regime changes**
4. **Paper trade before live deployment**
5. **Monitor live performance against validation metrics**

---

## 📧 **Support**

For questions or issues with the validation framework:
- Review the individual module docstrings
- Check the test files for usage examples
- Consult Dr. Bandy's books for methodology details