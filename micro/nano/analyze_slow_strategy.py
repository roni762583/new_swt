#!/usr/bin/env python3
"""Analyze Slow strategy expectancy and frequency."""

import pandas as pd

# Load the results
df = pd.read_csv('/workspace/breakout_touch_results.csv')

# Filter for Slow strategy only
slow_trades = df[df['params'] == 'Slow (4H zones, 8 pips)']

print('🔬 SLOW STRATEGY (4H zones, 8 pips) ANALYSIS')
print('=' * 60)

# Basic stats
print(f'Total trades: {len(slow_trades)}')
print(f'Win rate: {(slow_trades["pnl_pips"] > 0).mean() * 100:.1f}%')
print(f'Average P&L (net): {slow_trades["pnl_pips"].mean():.1f} pips')
print()

# Calculate R (average loss)
losses = slow_trades[slow_trades['pnl_pips'] < 0]['pnl_pips'].values
wins = slow_trades[slow_trades['pnl_pips'] > 0]['pnl_pips'].values

R = abs(losses.mean()) if len(losses) > 0 else 0
avg_win = wins.mean() if len(wins) > 0 else 0

print('📊 TRADE STATISTICS:')
print(f'Winning trades: {len(wins)} ({len(wins)/len(slow_trades)*100:.1f}%)')
print(f'Losing trades: {len(losses)} ({len(losses)/len(slow_trades)*100:.1f}%)')
print(f'Average win: {avg_win:.1f} pips')
print(f'Average loss: {losses.mean():.1f} pips')
print(f'R (avg loss): {R:.1f} pips')
print()

# Calculate Peoples Fintech Expectancy
avg_trade = slow_trades['pnl_pips'].mean()
expectancy = avg_trade / R if R > 0 else 0

print('💰 PEOPLES FINTECH EXPECTANCY:')
print(f'Formula: Expectancy = Avg Trade / R')
print(f'Calculation: {avg_trade:.1f} / {R:.1f}')
print(f'**Expectancy = {expectancy:.3f} R**')
print()

# Analyze trade frequency
total_bars_analyzed = 100000  # From the test
trades_generated = len(slow_trades)
bars_per_trade = total_bars_analyzed / trades_generated

print('⏰ TRADE FREQUENCY:')
print(f'Data analyzed: {total_bars_analyzed:,} bars (minutes)')
print(f'Trades generated: {trades_generated}')
print(f'**Frequency: 1 trade every {bars_per_trade:.0f} minutes**')
print(f'  = 1 trade every {bars_per_trade/60:.1f} hours')
print(f'  = 1 trade every {bars_per_trade/1440:.2f} days')
print()

# Daily/Monthly projections (24-hour forex market)
trades_per_day = 1440 / bars_per_trade  # 1440 minutes in a day
trades_per_month = trades_per_day * 22  # 22 trading days

print('📅 PROJECTED TRADING VOLUME:')
print(f'Trades per day: {trades_per_day:.2f}')
print(f'Trades per week: {trades_per_day * 5:.1f}')
print(f'Trades per month: {trades_per_month:.0f}')
print()

# Expected returns
print('📈 EXPECTED RETURNS:')
monthly_pips = trades_per_month * avg_trade
print(f'Expected monthly P&L: {monthly_pips:.0f} pips')
print(f'With $1000 per pip: ${monthly_pips * 1000:,.0f}/month')
print(f'With $100 per pip: ${monthly_pips * 100:,.0f}/month')
print(f'With $10 per pip: ${monthly_pips * 10:,.0f}/month')
print()

# Risk analysis
print('⚠️ RISK METRICS:')
print(f'Max win: {slow_trades["pnl_pips"].max():.1f} pips')
print(f'Max loss: {slow_trades["pnl_pips"].min():.1f} pips')
print(f'Std deviation: {slow_trades["pnl_pips"].std():.1f} pips')

# Calculate max drawdown potential
consecutive_losses = 5  # Assume 5 losses in a row
max_drawdown = R * consecutive_losses
print(f'Potential drawdown (5 losses): {max_drawdown:.0f} pips')
print()

# Summary
print('📊 SUMMARY:')
if expectancy > 0:
    print(f'✅ Positive expectancy: {expectancy:.3f}R per trade')
    print(f'   For every 1R risked, expect {expectancy:.3f}R return')
else:
    print(f'❌ Negative expectancy: {expectancy:.3f}R')

print(f'   Trade frequency: ~{trades_per_day:.1f} trades/day')
print(f'   Monthly expectation: {monthly_pips:.0f} pips')