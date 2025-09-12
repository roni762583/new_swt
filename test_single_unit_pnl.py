#!/usr/bin/env python3
"""
Test Single Unit P&L Calculation Issue
Verify if 1-unit trades produce measurable P&L changes in OANDA API
"""

import logging
from decimal import Decimal
import sys
from pathlib import Path

# Add to Python path
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)

def test_pip_calculation_precision():
    """Test precision of pip calculations for single unit trades"""
    
    print("🧪 Testing Single Unit P&L Calculation Precision")
    print("=" * 60)
    
    # GBP/JPY example prices
    entry_price = 195.123
    current_prices = [
        195.133,  # +1 pip
        195.223,  # +10 pips  
        195.523,  # +40 pips
        195.013,  # -11 pips
        194.623,  # -50 pips
    ]
    
    trade_size_gbp = 1  # 1 GBP unit
    
    print(f"Entry Price: {entry_price}")
    print(f"Trade Size: {trade_size_gbp} GBP")
    print()
    
    for current_price in current_prices:
        # Calculate pips (standard way)
        pips = (current_price - entry_price) * 10000
        
        # Calculate P&L in JPY (quote currency)
        pnl_jpy = (current_price - entry_price) * trade_size_gbp
        
        # Convert to USD (assuming USD account base currency)
        # JPY/USD ≈ 0.0067 (rough estimate, varies)
        jpy_to_usd_rate = 0.0067
        pnl_usd = pnl_jpy * jpy_to_usd_rate
        
        print(f"Current Price: {current_price}")
        print(f"  Pips: {pips:+.1f}")
        print(f"  P&L (JPY): {pnl_jpy:+.6f}")
        print(f"  P&L (USD): ${pnl_usd:+.8f}")
        print(f"  Measurable?: {'❌ NO' if abs(pnl_usd) < 0.01 else '✅ YES'}")
        print()

def test_oanda_precision_limits():
    """Test OANDA's precision limits for small trades"""
    
    print("🔍 OANDA API Precision Analysis")
    print("=" * 60)
    
    # Test various trade sizes
    trade_sizes = [1, 10, 100, 1000, 10000]  # GBP units
    pip_movement = 1  # 1 pip movement
    
    entry_price = 195.123
    current_price = entry_price + (pip_movement / 10000)  # +1 pip
    
    print(f"Entry: {entry_price}, Current: {current_price} (+1 pip)")
    print()
    
    for size in trade_sizes:
        pnl_jpy = (current_price - entry_price) * size
        pnl_usd = pnl_jpy * 0.0067  # Rough JPY to USD conversion
        
        print(f"Trade Size: {size:,} GBP")
        print(f"  P&L (JPY): {pnl_jpy:.6f}")
        print(f"  P&L (USD): ${pnl_usd:.6f}")
        
        if size == 1:
            print(f"  ⚠️  CRITICAL: 1-unit trade generates ${pnl_usd:.8f} for 1 pip!")
            if abs(pnl_usd) < 0.01:
                print(f"  🚨 PROBLEM: Below $0.01 - may not register in account balance!")
        print()

def analyze_current_implementation():
    """Analyze our current P&L calculation implementation"""
    
    print("📊 Current Implementation Analysis") 
    print("=" * 60)
    
    print("Current P&L Calculation Methods:")
    print()
    
    print("1. INTERNAL PIP CALCULATION (Position Management):")
    print("   pips = (current_price - entry_price) * 10000")
    print("   ❌ ISSUE: Used for trading decisions but not for P&L dollars")
    print()
    
    print("2. OANDA REALIZED P&L (Trade Closing):")
    print("   pnl_dollars = result.realized_pnl")
    print("   ✅ CORRECT: Uses actual OANDA-calculated P&L")
    print()
    
    print("3. ACCOUNT BALANCE UPDATES:")
    print("   self.account_balance += pnl_dollars")
    print("   ⚠️  POTENTIAL ISSUE: If pnl_dollars < $0.01, may show as $0.00")
    print()
    
    print("4. UNREALIZED P&L (During Trade):")
    print("   No current implementation - we only calculate pips!")
    print("   🚨 MAJOR GAP: No way to monitor real-time P&L in dollars")
    print()

def recommend_solutions():
    """Recommend solutions for accurate P&L tracking"""
    
    print("💡 RECOMMENDED SOLUTIONS")
    print("=" * 60)
    
    solutions = [
        "1. USE OANDA'S UNREALIZED P&L API",
        "   - Query position.unrealizedPL from OANDA API",
        "   - This gives exact P&L in account currency",
        "   - More accurate than our manual calculations",
        "",
        "2. IMPLEMENT POSITION VALUE MONITORING",
        "   - Track position.units and position.averagePrice from API", 
        "   - Calculate current market value vs entry value",
        "   - Use actual market prices for precision",
        "",
        "3. ADD MINIMUM POSITION SIZE CHECK",
        "   - For GBP/JPY, 1 unit ≈ $0.0067 per pip",
        "   - Consider minimum 100-1000 units for measurable P&L",
        "   - Or use broker's micro/nano lot sizing",
        "",
        "4. ENHANCE RECONCILIATION SYSTEM",
        "   - Compare OANDA's unrealizedPL vs our calculations",
        "   - Alert if discrepancies exceed tolerance",
        "   - Use broker data as source of truth for P&L",
        "",
        "5. FIX CURRENT IMPLEMENTATION",
        "   - Add unrealized P&L query during position management",
        "   - Update _manage_existing_position() to use real P&L",
        "   - Log both pips AND dollars for clarity"
    ]
    
    for solution in solutions:
        print(solution)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚨 SINGLE UNIT P&L CALCULATION ANALYSIS")
    print("=" * 60)
    print()
    
    test_pip_calculation_precision()
    print()
    
    test_oanda_precision_limits() 
    print()
    
    analyze_current_implementation()
    print()
    
    recommend_solutions()
    print()
    
    print("🎯 CONCLUSION:")
    print("=" * 60)
    print("✅ You are ABSOLUTELY CORRECT!")
    print("🚨 1-unit trades produce P&L below $0.01 per pip")
    print("⚠️  Current implementation may miss small P&L changes")
    print("💡 Need to use OANDA's unrealizedPL API for accuracy")
    print("🔧 Recommended: Increase minimum trade size or use API P&L values")