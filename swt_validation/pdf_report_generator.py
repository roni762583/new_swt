#!/usr/bin/env python3
"""
PDF Report Generator for SWT Validation
Generates comprehensive validation reports with charts and metrics
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

# Import matplotlib for chart generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for containers
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("Matplotlib not available - PDF reports disabled")

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generate comprehensive PDF validation reports"""

    def __init__(self, output_dir: str = "validation_results"):
        """
        Initialize PDF report generator

        Args:
            output_dir: Directory to save PDF reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_MATPLOTLIB:
            logger.warning("PDF generation disabled - matplotlib not available")

    def generate_validation_report(
        self,
        checkpoint_name: str,
        validation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate comprehensive PDF validation report

        Args:
            checkpoint_name: Name of the checkpoint
            validation_results: Validation results dictionary
            save_path: Optional path to save PDF

        Returns:
            Path to saved PDF or None if generation failed
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Cannot generate PDF - matplotlib not installed")
            return None

        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if save_path is None:
                save_path = self.output_dir / f"validation_{checkpoint_name}_{timestamp}.pdf"

            # Create PDF
            with PdfPages(save_path) as pdf:
                # Page 1: Summary
                self._create_summary_page(pdf, checkpoint_name, validation_results)

                # Page 2: Performance Metrics
                self._create_metrics_page(pdf, validation_results)

                # Page 3: Risk Analysis
                self._create_risk_page(pdf, validation_results)

                # Page 4: Trading Statistics
                self._create_trading_stats_page(pdf, validation_results)

                # Add metadata
                d = pdf.infodict()
                d['Title'] = f'SWT Validation Report - {checkpoint_name}'
                d['Author'] = 'SWT Validation System'
                d['Subject'] = 'Trading Model Validation'
                d['Keywords'] = 'SWT, MuZero, Trading, Validation'
                d['CreationDate'] = datetime.now()

            logger.info(f"✅ PDF report saved: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return None

    def _create_summary_page(self, pdf: PdfPages, checkpoint_name: str, results: Dict):
        """Create summary page"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f'SWT Validation Report\n{checkpoint_name}', fontsize=16, fontweight='bold')

        # Summary text
        summary_text = f"""
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        KEY METRICS:
        • Episode: {results.get('episode', 'N/A')}
        • Quality Score: {results.get('quality_score', 0):.2f}
        • Win Rate: {results.get('win_rate', 0)*100:.1f}%
        • Expectancy: {results.get('expectancy', 0):.4f}
        • CAR25: {results.get('car25', 0)*100:.1f}%
        • Max Drawdown: {results.get('max_drawdown', 0)*100:.1f}%
        • Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
        • Profit Factor: {results.get('profit_factor', 0):.2f}

        VALIDATION STATUS: {'✅ PASSED' if results.get('passed', False) else '❌ FAILED'}
        Grade: {results.get('grade', 'N/A')}
        """

        plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        plt.axis('off')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_metrics_page(self, pdf: PdfPages, results: Dict):
        """Create performance metrics page with charts"""
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Performance Metrics', fontsize=14, fontweight='bold')

        # Equity curve
        if 'equity_curve' in results:
            axes[0, 0].plot(results['equity_curve'])
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_xlabel('Trades')
            axes[0, 0].set_ylabel('Cumulative PnL')
            axes[0, 0].grid(True, alpha=0.3)

        # Win/Loss distribution
        if 'trade_returns' in results:
            returns = results['trade_returns']
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r <= 0]
            axes[0, 1].hist([wins, losses], label=['Wins', 'Losses'], bins=20, alpha=0.7)
            axes[0, 1].set_title('Trade Distribution')
            axes[0, 1].set_xlabel('Return (pips)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Monthly returns heatmap
        if 'monthly_returns' in results:
            monthly = results['monthly_returns']
            im = axes[1, 0].imshow(monthly.reshape(-1, 12), cmap='RdYlGn', aspect='auto')
            axes[1, 0].set_title('Monthly Returns Heatmap')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Year')
            plt.colorbar(im, ax=axes[1, 0])

        # Drawdown chart
        if 'drawdown_series' in results:
            axes[1, 1].fill_between(range(len(results['drawdown_series'])),
                                   results['drawdown_series'], 0,
                                   color='red', alpha=0.3)
            axes[1, 1].set_title('Drawdown')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Drawdown %')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_risk_page(self, pdf: PdfPages, results: Dict):
        """Create risk analysis page"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Risk Analysis', fontsize=14, fontweight='bold')

        risk_text = f"""
        RISK METRICS:

        • Value at Risk (95%): {results.get('var_95', 0):.2f} pips
        • Conditional VaR (95%): {results.get('cvar_95', 0):.2f} pips
        • Maximum Drawdown: {results.get('max_drawdown', 0)*100:.1f}%
        • Average Drawdown: {results.get('avg_drawdown', 0)*100:.1f}%
        • Recovery Factor: {results.get('recovery_factor', 0):.2f}
        • Ulcer Index: {results.get('ulcer_index', 0):.2f}

        MONTE CARLO ANALYSIS:
        • Probability of Profit: {results.get('prob_profit', 0)*100:.1f}%
        • 95% Confidence Lower: {results.get('conf_lower', 0):.2f}
        • 95% Confidence Upper: {results.get('conf_upper', 0):.2f}
        • CAR25 (Conservative): {results.get('car25', 0)*100:.1f}%

        STRESS TEST RESULTS:
        • Robustness Score: {results.get('robustness_score', 0)}/100
        • Worst Case Scenario: {results.get('worst_case', 0)*100:.1f}%
        • Best Case Scenario: {results.get('best_case', 0)*100:.1f}%
        """

        plt.text(0.1, 0.5, risk_text, fontsize=11, verticalalignment='center')
        plt.axis('off')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_trading_stats_page(self, pdf: PdfPages, results: Dict):
        """Create trading statistics page"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Trading Statistics', fontsize=14, fontweight='bold')

        stats_text = f"""
        TRADING PERFORMANCE:

        • Total Trades: {results.get('total_trades', 0)}
        • Winning Trades: {results.get('winning_trades', 0)}
        • Losing Trades: {results.get('losing_trades', 0)}
        • Win Rate: {results.get('win_rate', 0)*100:.1f}%

        • Average Win: {results.get('avg_win', 0):.2f} pips
        • Average Loss: {results.get('avg_loss', 0):.2f} pips
        • Largest Win: {results.get('largest_win', 0):.2f} pips
        • Largest Loss: {results.get('largest_loss', 0):.2f} pips

        • Profit Factor: {results.get('profit_factor', 0):.2f}
        • Expectancy: {results.get('expectancy', 0):.4f}
        • Payoff Ratio: {results.get('payoff_ratio', 0):.2f}

        • Average Trade Duration: {results.get('avg_duration', 0):.1f} minutes
        • Maximum Consecutive Wins: {results.get('max_consec_wins', 0)}
        • Maximum Consecutive Losses: {results.get('max_consec_losses', 0)}

        POSITION DISTRIBUTION:
        • Long Trades: {results.get('long_trades', 0)} ({results.get('long_win_rate', 0)*100:.1f}% win rate)
        • Short Trades: {results.get('short_trades', 0)} ({results.get('short_win_rate', 0)*100:.1f}% win rate)
        • Hold Periods: {results.get('hold_periods', 0)}
        """

        plt.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center')
        plt.axis('off')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def generate_quick_report(checkpoint_path: str, validation_results: Dict) -> Optional[str]:
    """
    Quick helper function to generate a validation report

    Args:
        checkpoint_path: Path to checkpoint file
        validation_results: Validation results dictionary

    Returns:
        Path to generated PDF or None
    """
    generator = PDFReportGenerator()
    checkpoint_name = Path(checkpoint_path).stem
    return generator.generate_validation_report(checkpoint_name, validation_results)