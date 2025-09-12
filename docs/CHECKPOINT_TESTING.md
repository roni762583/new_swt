# 🧪 SWT Checkpoint Testing Framework

## Overview

The SWT system includes a comprehensive checkpoint testing framework that validates any checkpoint (Episode 13475 or future models) against unseen test data. This ensures model performance and Episode 13475 compatibility.

## 🎯 **Key Features**

- **Universal Testing**: Works with any checkpoint format (PyTorch, Pickle, JSON)
- **Episode 13475 Validation**: Verifies exact parameter compatibility
- **Performance Benchmarking**: Measures inference speed and accuracy
- **Trading Simulation**: Tests real trading performance
- **Memory Efficient**: Handles large CSV datasets incrementally
- **Docker Support**: Isolated testing environment
- **Comprehensive Reports**: Detailed analysis with recommendations

## 📁 **File Structure**

```
new_swt/
├── test_checkpoint_performance.py    # Main testing framework
├── run_checkpoint_tests.sh          # Automated testing script
├── scripts/
│   └── download_oanda_data.py        # Data downloader
├── data/
│   ├── GBPJPY_M1_202201-202508.csv  # Test data (3+ years)
│   └── training_data/               # Training data
├── checkpoints/
│   └── episode_13475.pt             # Episode 13475 checkpoint
└── test_results/                    # Test outputs
```

## 🚀 **Quick Start**

### 1. Download Test Data
```bash
# Download GBPJPY M1 data from 2022-2025
python scripts/download_oanda_data.py
```

### 2. Run Tests
```bash
# Quick validation test
./run_checkpoint_tests.sh quick

# Full performance test
./run_checkpoint_tests.sh test

# Docker-based testing
./run_checkpoint_tests.sh docker
```

### 3. View Results
```bash
# Results are saved in test_results/
ls test_results/episode_13475_*/
```

## 📊 **Test Types**

### **1. Episode 13475 Compatibility Test**
Verifies exact parameter matching:
- MCTS: 15 simulations, C_PUCT=1.25
- Features: 9D position + 128D market = 137D observation
- WST: J=2, Q=6 parameters
- Normalization: duration/720.0, pnl/100.0, etc.

### **2. Performance Benchmark**
Measures system performance:
- **Inference Speed**: Average, P95, P99 latency
- **Feature Processing**: Time for observation creation
- **Success Rate**: Percentage of successful inferences
- **Memory Usage**: Resource consumption

### **3. Trading Simulation**
Real trading performance test:
- **Virtual Trading**: Simulated trades on historical data
- **P&L Tracking**: Profit/loss calculation
- **Risk Management**: Position limits and safety checks
- **Win Rate**: Percentage of profitable trades

### **4. Action Analysis**
Decision-making evaluation:
- **Action Distribution**: Hold/Buy/Sell frequency
- **Confidence Scores**: Model confidence levels
- **Decision Quality**: Trading signal analysis

## 🔧 **Usage Examples**

### **Test Specific Checkpoint**
```bash
python test_checkpoint_performance.py \
  --checkpoint checkpoints/my_model.pt \
  --data data/GBPJPY_M1_202201-202508.csv \
  --output test_results/my_model \
  --sample-size 5000
```

### **Test with Different Data**
```bash
CHECKPOINT_PATH=checkpoints/episode_20000.pt \
TEST_DATA_PATH=data/EURUSD_M1_2024.csv \
./run_checkpoint_tests.sh test
```

### **Quick Performance Check**
```bash
# Test with 100 samples for fast validation
./run_checkpoint_tests.sh quick
```

### **Download and Test**
```bash
# Download data and run full test
./run_checkpoint_tests.sh download
./run_checkpoint_tests.sh test
```

## 📋 **Test Report Example**

```markdown
# SWT Checkpoint Performance Test Report

## Test Summary
- **Test Date**: 2025-01-08T10:30:00
- **Episode 13475 Compatible**: ✅ Yes
- **Performance Grade**: A+ (Excellent)

## Performance Metrics
- **Success Rate**: 99.8%
- **Average Inference Time**: 145.2ms
- **P95 Inference Time**: 289.4ms

## Action Distribution
- **Hold**: 1,245 (62.3%)
- **Buy**: 398 (19.9%)
- **Sell**: 357 (17.8%)

## Trading Simulation Results
- **Total Return**: +12.4%
- **Win Rate**: 68.2%
- **Total Trades**: 89
- **Max Drawdown**: -$45.20

## Performance Assessment
✅ Excellent - Average inference time: 145.2ms
✅ Good Diversity - Hold rate: 62.3%
✅ Profitable - Return: +12.4%
```

## 🎯 **Performance Grading**

The framework assigns performance grades based on:

### **Speed (30 points)**
- A: ≤100ms average inference
- B: ≤200ms average inference  
- C: ≤300ms average inference
- D: ≤500ms average inference
- F: >500ms average inference

### **Success Rate (40 points)**
- A: ≥95% successful inferences
- B: ≥90% successful inferences
- C: ≥85% successful inferences
- D: ≥80% successful inferences
- F: <80% successful inferences

### **Decision Making (20 points)**
- A: 50-70% hold rate (balanced)
- B: 70-80% hold rate (conservative)
- C: 80-90% hold rate (very conservative)
- D: 90-95% hold rate (too conservative)
- F: >95% hold rate (no decisions)

### **Confidence (10 points)**
- A: Mean >0.3, StdDev >0.1 (varied confidence)
- B: Mean >0.3 (good confidence)
- C: StdDev >0.1 (some variation)
- D: Basic confidence distribution
- F: Poor confidence distribution

## 🐋 **Docker Testing**

### **Advantages**
- **Isolated Environment**: No dependency conflicts
- **Reproducible Results**: Consistent testing environment
- **Resource Limits**: Memory/CPU constraints
- **Production Simulation**: Container-like deployment

### **Usage**
```bash
# Build and run tests in Docker
./run_checkpoint_tests.sh docker

# Monitor resource usage
docker stats swt-checkpoint-test
```

## 📈 **Data Requirements**

### **CSV Format**
Required columns:
```csv
timestamp,open,high,low,close,volume
2022-01-01 00:00:00,1.20000,1.20005,1.19995,1.20002,150
```

### **Minimum Data**
- **Time Range**: At least 30 days of data
- **Data Points**: Minimum 10,000 candles
- **Quality**: Complete candles only (no gaps)

### **Recommended Data**
- **Time Range**: 1+ years for comprehensive testing
- **Data Points**: 100,000+ candles
- **Currency Pairs**: GBPJPY, EURUSD, USDJPY

## 🔍 **Troubleshooting**

### **Common Issues**

#### **"Checkpoint file not found"**
```bash
# Check available checkpoints
ls checkpoints/
# Verify path
echo $CHECKPOINT_PATH
```

#### **"Test data file not found"**
```bash
# Download data
python scripts/download_oanda_data.py
# Or specify existing data
export TEST_DATA_PATH="path/to/your/data.csv"
```

#### **"Memory errors during testing"**
```bash
# Limit data size
python test_checkpoint_performance.py --max-rows 10000 --sample-size 500
```

#### **"Import errors"**
```bash
# Check Python path
export PYTHONPATH=$PWD:$PYTHONPATH
# Install missing dependencies
pip install -r requirements.txt
```

### **Performance Issues**

#### **Slow inference**
- Check MCTS simulation count (should be 15 for Episode 13475)
- Verify GPU/CPU usage
- Consider model optimization

#### **Poor trading results**
- Verify Episode 13475 compatibility
- Check feature normalization parameters
- Review confidence thresholds

## 🔄 **Integration with CI/CD**

### **Automated Testing**
```yaml
# .github/workflows/checkpoint-test.yml
name: Checkpoint Testing
on:
  push:
    paths:
      - 'checkpoints/**'
jobs:
  test-checkpoints:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test Checkpoint
        run: ./run_checkpoint_tests.sh quick
```

### **Performance Monitoring**
```bash
# Run tests on model updates
./run_checkpoint_tests.sh test > performance_log.txt
# Check for regressions
grep "Performance Grade" performance_log.txt
```

## 📚 **Advanced Usage**

### **Custom Test Configuration**
```python
# test_config.py
CUSTOM_CONFIG = {
    'max_test_samples': 10000,
    'confidence_threshold': 0.4,
    'position_size': 0.01,
    'risk_limits': {
        'daily_loss': -100.0,
        'max_position': 0.05
    }
}
```

### **Batch Testing Multiple Checkpoints**
```bash
#!/bin/bash
for checkpoint in checkpoints/*.pt; do
    echo "Testing $checkpoint"
    CHECKPOINT_PATH=$checkpoint ./run_checkpoint_tests.sh test
done
```

### **Performance Comparison**
```python
# compare_checkpoints.py
import json
import pandas as pd

# Load multiple test results
results = []
for result_file in glob.glob("test_results/*/test_results.json"):
    with open(result_file) as f:
        results.append(json.load(f))

# Create comparison DataFrame
df = pd.DataFrame([{
    'checkpoint': r['checkpoint_info']['episode'],
    'performance_grade': r['performance_metrics']['performance_grade'],
    'inference_time': r['performance_metrics']['performance']['avg_inference_time_ms'],
    'trading_return': r['trading_simulation']['total_return_pct']
} for r in results])

print(df.sort_values('trading_return', ascending=False))
```

## 🎯 **Best Practices**

### **Testing Strategy**
1. **Regular Testing**: Test every new checkpoint
2. **Multiple Data Sets**: Use different currency pairs and time periods
3. **Performance Monitoring**: Track inference speed over time
4. **A/B Testing**: Compare checkpoints side-by-side

### **Data Management**
1. **Clean Data**: Ensure no missing values or gaps
2. **Sufficient Volume**: Use at least 100k candles for comprehensive testing
3. **Unseen Data**: Never use training data for testing
4. **Regular Updates**: Refresh test data periodically

### **Result Interpretation**
1. **Episode 13475 Compatibility**: Must be ✅ for production use
2. **Speed Requirements**: Target <200ms for live trading
3. **Trading Performance**: Positive returns over multiple test periods
4. **Decision Quality**: Balanced action distribution (not >90% hold)

---

This framework ensures robust validation of any SWT checkpoint against unseen data, maintaining the high standards established by Episode 13475 while enabling continuous improvement and testing of new models.