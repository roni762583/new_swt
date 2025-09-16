"""
System Quality Number (SQN) Calculator for Trading Performance Evaluation

SQN provides a normalized performance metric that accounts for:
- Expectancy (average trade outcome)
- Variability of results (consistency)
- Sample size (statistical significance)

This makes it more reliable than expectancy alone for evaluating trading systems.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SQNResult:
    """Result of SQN calculation with detailed metrics"""
    sqn: float
    expectancy: float
    std_dev: float
    num_trades: int
    r_multiples: List[float]
    classification: str
    confidence_level: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'sqn': self.sqn,
            'expectancy': self.expectancy,
            'std_dev': self.std_dev,
            'num_trades': self.num_trades,
            'classification': self.classification,
            'confidence_level': self.confidence_level,
            'avg_r_multiple': self.expectancy,
            'max_r_multiple': max(self.r_multiples) if self.r_multiples else 0,
            'min_r_multiple': min(self.r_multiples) if self.r_multiples else 0
        }


class SQNCalculator:
    """
    Calculate System Quality Number (SQN) for trading performance evaluation.

    SQN = (Expectancy / StdDev) * sqrt(N)

    Where:
    - Expectancy = Average R-multiple
    - StdDev = Standard deviation of R-multiples
    - N = Number of trades

    Classification thresholds (Van Tharp):
    - < 1.6: Poor system
    - 1.6-1.9: Below average
    - 2.0-2.4: Average
    - 2.5-2.9: Good
    - 3.0-4.9: Excellent
    - 5.0-6.9: Superb
    - >= 7.0: Holy Grail
    """

    # SQN Classification thresholds
    CLASSIFICATIONS = [
        (7.0, "Holy Grail System"),
        (5.0, "Superb System"),
        (3.0, "Excellent System"),
        (2.5, "Good System"),
        (2.0, "Average System"),
        (1.6, "Below Average System"),
        (0.0, "Poor System"),
        (-float('inf'), "Losing System")
    ]

    # Sample size confidence levels
    CONFIDENCE_LEVELS = [
        (100, "High Confidence"),
        (30, "Moderate Confidence"),
        (20, "Low Confidence"),
        (0, "Insufficient Data")
    ]

    def __init__(self, risk_per_trade: Optional[float] = None):
        """
        Initialize SQN Calculator.

        Args:
            risk_per_trade: Fixed risk amount per trade (optional).
                           If None, will use actual loss amounts as risk.
        """
        self.risk_per_trade = risk_per_trade

    def calculate_r_multiples(
        self,
        pnl_values: List[float],
        risk_values: Optional[List[float]] = None
    ) -> List[float]:
        """
        Convert P&L values to R-multiples.

        Args:
            pnl_values: List of profit/loss values
            risk_values: List of risk amounts per trade (optional)

        Returns:
            List of R-multiples
        """
        if not pnl_values:
            return []

        r_multiples = []

        for i, pnl in enumerate(pnl_values):
            # Determine risk for this trade
            if risk_values and i < len(risk_values):
                risk = risk_values[i]
            elif self.risk_per_trade:
                risk = self.risk_per_trade
            else:
                # Use absolute value of loss as risk proxy
                risk = abs(pnl) if pnl < 0 else abs(pnl) * 0.5

            # Avoid division by zero
            if risk == 0:
                risk = 1.0

            r_multiple = pnl / risk
            r_multiples.append(r_multiple)

        return r_multiples

    def calculate_sqn(
        self,
        pnl_values: List[float],
        risk_values: Optional[List[float]] = None
    ) -> SQNResult:
        """
        Calculate System Quality Number from trade P&L values.

        Args:
            pnl_values: List of profit/loss values
            risk_values: Optional list of risk amounts per trade

        Returns:
            SQNResult with detailed metrics
        """
        # Handle empty or insufficient data
        if not pnl_values:
            return SQNResult(
                sqn=0.0,
                expectancy=0.0,
                std_dev=0.0,
                num_trades=0,
                r_multiples=[],
                classification="No Trades",
                confidence_level="No Data"
            )

        # Convert to R-multiples
        r_multiples = self.calculate_r_multiples(pnl_values, risk_values)
        num_trades = len(r_multiples)

        # Calculate expectancy (mean of R-multiples)
        expectancy = np.mean(r_multiples)

        # Calculate standard deviation
        if num_trades > 1:
            std_dev = np.std(r_multiples, ddof=1)  # Sample standard deviation
        else:
            std_dev = 0.0

        # Calculate SQN
        if std_dev > 0:
            sqn = (expectancy / std_dev) * np.sqrt(num_trades)
        elif expectancy > 0:
            sqn = float('inf')  # Perfect consistency with positive expectancy
        else:
            sqn = 0.0

        # Classify the system
        classification = self._classify_sqn(sqn)
        confidence_level = self._get_confidence_level(num_trades)

        return SQNResult(
            sqn=sqn,
            expectancy=expectancy,
            std_dev=std_dev,
            num_trades=num_trades,
            r_multiples=r_multiples,
            classification=classification,
            confidence_level=confidence_level
        )

    def _classify_sqn(self, sqn: float) -> str:
        """Classify system based on SQN value"""
        for threshold, classification in self.CLASSIFICATIONS:
            if sqn >= threshold:
                return classification
        return "Losing System"

    def _get_confidence_level(self, num_trades: int) -> str:
        """Determine confidence level based on sample size"""
        for threshold, level in self.CONFIDENCE_LEVELS:
            if num_trades >= threshold:
                return level
        return "Insufficient Data"

    def calculate_rolling_sqn(
        self,
        pnl_values: List[float],
        window_size: int = 30,
        risk_values: Optional[List[float]] = None
    ) -> List[float]:
        """
        Calculate rolling SQN over a moving window.

        Args:
            pnl_values: List of profit/loss values
            window_size: Size of rolling window
            risk_values: Optional list of risk amounts

        Returns:
            List of rolling SQN values
        """
        if len(pnl_values) < window_size:
            return []

        rolling_sqn = []
        for i in range(window_size, len(pnl_values) + 1):
            window_pnl = pnl_values[i-window_size:i]
            window_risk = risk_values[i-window_size:i] if risk_values else None

            result = self.calculate_sqn(window_pnl, window_risk)
            rolling_sqn.append(result.sqn)

        return rolling_sqn

    def analyze_session(
        self,
        trades: List[Dict],
        use_amddp1: bool = True
    ) -> SQNResult:
        """
        Analyze a trading session and calculate SQN.

        Args:
            trades: List of trade dictionaries with 'pnl' and optional 'risk' keys
            use_amddp1: Whether to use AMDDP1 rewards if available

        Returns:
            SQNResult with session analysis
        """
        if not trades:
            return self.calculate_sqn([])

        # Extract P&L values
        pnl_values = []
        risk_values = []

        for trade in trades:
            # Use AMDDP1 reward if available and requested
            if use_amddp1 and 'amddp1_reward' in trade:
                pnl = trade['amddp1_reward']
            elif 'pnl' in trade:
                pnl = trade['pnl']
            else:
                continue

            pnl_values.append(pnl)

            # Extract risk if available
            if 'risk' in trade:
                risk_values.append(trade['risk'])

        # Calculate SQN
        result = self.calculate_sqn(
            pnl_values,
            risk_values if risk_values else None
        )

        # Log analysis
        logger.info(f"📊 SQN Analysis: {result.sqn:.2f} ({result.classification})")
        logger.info(f"   Trades: {result.num_trades}, Expectancy: {result.expectancy:.3f}")
        logger.info(f"   StdDev: {result.std_dev:.3f}, Confidence: {result.confidence_level}")

        return result


def calculate_session_sqn(
    trades: List[Dict],
    risk_per_trade: Optional[float] = None
) -> float:
    """
    Convenience function to calculate SQN for a session.

    Args:
        trades: List of trade dictionaries
        risk_per_trade: Optional fixed risk per trade

    Returns:
        SQN value
    """
    calculator = SQNCalculator(risk_per_trade)
    result = calculator.analyze_session(trades)
    return result.sqn