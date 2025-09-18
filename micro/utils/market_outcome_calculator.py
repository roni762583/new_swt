#!/usr/bin/env python3
"""
Market outcome calculator based on rolling standard deviation.

Classifies price movements into discrete outcomes:
- UP: price change > 0.5 * rolling_stdev
- NEUTRAL: price change within ±0.5 * rolling_stdev
- DOWN: price change < -0.5 * rolling_stdev
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarketOutcomeCalculator:
    """
    Calculate market outcomes based on rolling statistics.
    """

    def __init__(
        self,
        window_size: int = 20,
        threshold_multiplier: float = 0.5,
        min_threshold: float = 0.0001  # Minimum threshold to avoid zero division
    ):
        """
        Initialize calculator.

        Args:
            window_size: Window for rolling statistics
            threshold_multiplier: Multiplier for stdev threshold (default 0.5)
            min_threshold: Minimum threshold to prevent zero division
        """
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.min_threshold = min_threshold
        self.price_history = []

    def add_price(self, price: float):
        """Add new price to history."""
        self.price_history.append(price)
        # Keep only necessary history
        if len(self.price_history) > self.window_size * 2:
            self.price_history.pop(0)

    def get_rolling_stdev(self) -> Optional[float]:
        """
        Calculate current rolling standard deviation.

        Returns:
            Rolling stdev or None if insufficient data
        """
        if len(self.price_history) < self.window_size:
            return None

        recent_prices = self.price_history[-self.window_size:]
        stdev = np.std(recent_prices)
        return max(stdev, self.min_threshold)

    def calculate_outcome(
        self,
        current_price: float,
        next_price: float,
        custom_stdev: Optional[float] = None
    ) -> int:
        """
        Calculate market outcome for price change.

        Args:
            current_price: Current price
            next_price: Next price
            custom_stdev: Optional custom stdev (otherwise uses rolling)

        Returns:
            0: UP (price increased significantly)
            1: NEUTRAL (price stayed within threshold)
            2: DOWN (price decreased significantly)
        """
        price_change = next_price - current_price

        # Use custom or rolling stdev
        if custom_stdev is not None:
            stdev = custom_stdev
        else:
            stdev = self.get_rolling_stdev()
            if stdev is None:
                # Not enough history - use price-relative threshold
                stdev = abs(current_price) * 0.001

        # Calculate threshold
        threshold = self.threshold_multiplier * stdev

        # Classify outcome
        if price_change > threshold:
            return 0  # UP
        elif price_change < -threshold:
            return 2  # DOWN
        else:
            return 1  # NEUTRAL

    def calculate_outcome_from_returns(
        self,
        return_pct: float,
        stdev_returns: Optional[float] = None
    ) -> int:
        """
        Calculate outcome from percentage returns.

        Args:
            return_pct: Percentage return
            stdev_returns: Standard deviation of returns

        Returns:
            Market outcome (0: UP, 1: NEUTRAL, 2: DOWN)
        """
        if stdev_returns is None:
            # Calculate from price history if available
            if len(self.price_history) >= 2:
                returns = []
                for i in range(1, len(self.price_history)):
                    r = (self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]
                    returns.append(r)

                if len(returns) >= self.window_size:
                    stdev_returns = np.std(returns[-self.window_size:])
                else:
                    stdev_returns = 0.001  # Default 0.1% stdev
            else:
                stdev_returns = 0.001

        threshold = self.threshold_multiplier * stdev_returns

        if return_pct > threshold:
            return 0  # UP
        elif return_pct < -threshold:
            return 2  # DOWN
        else:
            return 1  # NEUTRAL

    def get_outcome_probabilities_empirical(
        self,
        lookback: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate empirical outcome probabilities from historical data.

        Args:
            lookback: Number of periods to look back (None = all history)

        Returns:
            Array of [P(UP), P(NEUTRAL), P(DOWN)]
        """
        if len(self.price_history) < 2:
            # No history - return uniform
            return np.array([1/3, 1/3, 1/3])

        # Calculate outcomes for historical data
        outcomes = []
        start_idx = 1 if lookback is None else max(1, len(self.price_history) - lookback)

        for i in range(start_idx, len(self.price_history)):
            if i >= self.window_size:  # Need enough history for stdev
                outcome = self.calculate_outcome(
                    self.price_history[i-1],
                    self.price_history[i]
                )
                outcomes.append(outcome)

        if not outcomes:
            return np.array([1/3, 1/3, 1/3])

        # Count frequencies
        counts = np.bincount(outcomes, minlength=3)
        probabilities = counts / counts.sum()

        return probabilities

    def reset(self):
        """Reset calculator state."""
        self.price_history = []


def calculate_outcome_from_window(
    prices: np.ndarray,
    window_size: int = 20,
    threshold_multiplier: float = 0.5
) -> Tuple[int, float]:
    """
    Calculate outcome from price window (convenience function).

    Args:
        prices: Array of prices (oldest to newest)
        window_size: Window for stdev calculation
        threshold_multiplier: Threshold multiplier

    Returns:
        Tuple of (outcome, stdev)
    """
    if len(prices) < 2:
        return 1, 0.0  # NEUTRAL if no data

    # Calculate rolling stdev
    if len(prices) >= window_size:
        stdev = np.std(prices[-window_size:])
    else:
        stdev = np.std(prices)

    # Prevent zero stdev
    stdev = max(stdev, 1e-6)

    # Calculate price change
    price_change = prices[-1] - prices[-2]

    # Classify
    threshold = threshold_multiplier * stdev
    if price_change > threshold:
        outcome = 0  # UP
    elif price_change < -threshold:
        outcome = 2  # DOWN
    else:
        outcome = 1  # NEUTRAL

    return outcome, stdev