#!/usr/bin/env python3
"""
Composite Scoring System for SWT Checkpoint Evaluation
Implements balanced scoring considering expectancy, risk, consistency, and drawdown
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Configurable weights for composite scoring components"""
    expectancy: float = 0.30  # 30% weight
    risk_adjusted_return: float = 0.30  # 30% weight (Sharpe/Sortino)
    consistency: float = 0.20  # 20% weight (Win rate stability)
    drawdown_control: float = 0.20  # 20% weight
    
    def validate(self) -> None:
        """Ensure weights sum to 1.0"""
        total = self.expectancy + self.risk_adjusted_return + self.consistency + self.drawdown_control
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class PerformanceThresholds:
    """Performance thresholds for scoring"""
    # Expectancy thresholds (pips)
    expectancy_excellent: float = 0.5
    expectancy_good: float = 0.3
    expectancy_acceptable: float = 0.1
    
    # Sharpe ratio thresholds
    sharpe_excellent: float = 2.0
    sharpe_good: float = 1.5
    sharpe_acceptable: float = 1.0
    
    # Sortino ratio thresholds (higher than Sharpe since only downside volatility)
    sortino_excellent: float = 2.5
    sortino_good: float = 2.0
    sortino_acceptable: float = 1.5
    
    # Win rate thresholds
    win_rate_excellent: float = 0.60
    win_rate_good: float = 0.50
    win_rate_acceptable: float = 0.40
    
    # Max drawdown thresholds (lower is better)
    max_dd_excellent: float = 0.10  # 10%
    max_dd_good: float = 0.15  # 15%
    max_dd_acceptable: float = 0.20  # 20%
    
    # Profit factor thresholds
    profit_factor_excellent: float = 2.0
    profit_factor_good: float = 1.5
    profit_factor_acceptable: float = 1.2
    
    # Recovery factor thresholds
    recovery_factor_excellent: float = 3.0
    recovery_factor_good: float = 2.0
    recovery_factor_acceptable: float = 1.0


@dataclass
class CheckpointMetrics:
    """Complete metrics for a checkpoint"""
    checkpoint_path: str
    episode: int
    timestamp: datetime
    
    # Core metrics
    expectancy: float  # Expected value per trade
    win_rate: float
    profit_factor: float
    total_trades: int
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_pips: float
    
    # Additional metrics
    avg_win_pips: float
    avg_loss_pips: float
    win_loss_ratio: float
    recovery_factor: float
    
    # Optional advanced metrics
    kelly_criterion: Optional[float] = None
    var_95: Optional[float] = None  # Value at Risk
    cvar_95: Optional[float] = None  # Conditional VaR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'checkpoint_path': self.checkpoint_path,
            'episode': self.episode,
            'timestamp': self.timestamp.isoformat(),
            'expectancy': self.expectancy,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_drawdown_pips': self.max_drawdown_pips,
            'avg_win_pips': self.avg_win_pips,
            'avg_loss_pips': self.avg_loss_pips,
            'win_loss_ratio': self.win_loss_ratio,
            'recovery_factor': self.recovery_factor,
            'kelly_criterion': self.kelly_criterion,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95
        }


@dataclass
class CompositeScore:
    """Composite score with component breakdown"""
    total_score: float  # 0-100
    expectancy_score: float  # Component scores
    risk_score: float
    consistency_score: float
    drawdown_score: float
    
    grade: str  # A+, A, B+, etc.
    recommendation: str  # DEPLOY, TEST, IMPROVE, REJECT
    confidence_level: str  # HIGH, MEDIUM, LOW
    
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'total_score': self.total_score,
            'components': {
                'expectancy': self.expectancy_score,
                'risk_adjusted': self.risk_score,
                'consistency': self.consistency_score,
                'drawdown_control': self.drawdown_score
            },
            'grade': self.grade,
            'recommendation': self.recommendation,
            'confidence_level': self.confidence_level,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses
        }


class CompositeScorer:
    """
    Composite scoring system for checkpoint evaluation
    Balances multiple performance factors for robust assessment
    """
    
    def __init__(self, 
                 weights: Optional[ScoringWeights] = None,
                 thresholds: Optional[PerformanceThresholds] = None):
        """
        Initialize scorer with configurable weights and thresholds
        
        Args:
            weights: Component weights for scoring
            thresholds: Performance thresholds for evaluation
        """
        self.weights = weights or ScoringWeights()
        self.weights.validate()
        self.thresholds = thresholds or PerformanceThresholds()
        
    def calculate_composite_score(self, metrics: CheckpointMetrics) -> CompositeScore:
        """
        Calculate comprehensive composite score for checkpoint
        
        Args:
            metrics: Checkpoint performance metrics
            
        Returns:
            CompositeScore with detailed breakdown
        """
        # Calculate component scores
        expectancy_score = self._score_expectancy(metrics)
        risk_score = self._score_risk_adjusted_return(metrics)
        consistency_score = self._score_consistency(metrics)
        drawdown_score = self._score_drawdown_control(metrics)
        
        # Calculate weighted total
        total_score = (
            expectancy_score * self.weights.expectancy +
            risk_score * self.weights.risk_adjusted_return +
            consistency_score * self.weights.consistency +
            drawdown_score * self.weights.drawdown_control
        )
        
        # Determine grade
        grade = self._calculate_grade(total_score)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            total_score, metrics, expectancy_score, risk_score, 
            consistency_score, drawdown_score
        )
        
        # Determine confidence level
        confidence_level = self._assess_confidence_level(metrics)
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._analyze_performance(
            metrics, expectancy_score, risk_score, 
            consistency_score, drawdown_score
        )
        
        return CompositeScore(
            total_score=total_score,
            expectancy_score=expectancy_score,
            risk_score=risk_score,
            consistency_score=consistency_score,
            drawdown_score=drawdown_score,
            grade=grade,
            recommendation=recommendation,
            confidence_level=confidence_level,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _score_expectancy(self, metrics: CheckpointMetrics) -> float:
        """Score expectancy component (0-100)"""
        expectancy = metrics.expectancy
        
        if expectancy >= self.thresholds.expectancy_excellent:
            return 100
        elif expectancy >= self.thresholds.expectancy_good:
            # Linear interpolation between good and excellent
            ratio = (expectancy - self.thresholds.expectancy_good) / (
                self.thresholds.expectancy_excellent - self.thresholds.expectancy_good
            )
            return 70 + ratio * 30
        elif expectancy >= self.thresholds.expectancy_acceptable:
            ratio = (expectancy - self.thresholds.expectancy_acceptable) / (
                self.thresholds.expectancy_good - self.thresholds.expectancy_acceptable
            )
            return 40 + ratio * 30
        elif expectancy > 0:
            ratio = expectancy / self.thresholds.expectancy_acceptable
            return ratio * 40
        else:
            return 0
    
    def _score_risk_adjusted_return(self, metrics: CheckpointMetrics) -> float:
        """Score risk-adjusted return using Sharpe and Sortino ratios (0-100)"""
        # Use Sortino if available (better for downside risk), otherwise Sharpe
        ratio = metrics.sortino_ratio if metrics.sortino_ratio > 0 else metrics.sharpe_ratio
        
        if metrics.sortino_ratio > 0:
            # Using Sortino ratio thresholds
            if ratio >= self.thresholds.sortino_excellent:
                return 100
            elif ratio >= self.thresholds.sortino_good:
                interp = (ratio - self.thresholds.sortino_good) / (
                    self.thresholds.sortino_excellent - self.thresholds.sortino_good
                )
                return 70 + interp * 30
            elif ratio >= self.thresholds.sortino_acceptable:
                interp = (ratio - self.thresholds.sortino_acceptable) / (
                    self.thresholds.sortino_good - self.thresholds.sortino_acceptable
                )
                return 40 + interp * 30
            elif ratio > 0:
                return (ratio / self.thresholds.sortino_acceptable) * 40
            else:
                return 0
        else:
            # Using Sharpe ratio thresholds
            if ratio >= self.thresholds.sharpe_excellent:
                return 100
            elif ratio >= self.thresholds.sharpe_good:
                interp = (ratio - self.thresholds.sharpe_good) / (
                    self.thresholds.sharpe_excellent - self.thresholds.sharpe_good
                )
                return 70 + interp * 30
            elif ratio >= self.thresholds.sharpe_acceptable:
                interp = (ratio - self.thresholds.sharpe_acceptable) / (
                    self.thresholds.sharpe_good - self.thresholds.sharpe_acceptable
                )
                return 40 + interp * 30
            elif ratio > 0:
                return (ratio / self.thresholds.sharpe_acceptable) * 40
            else:
                return 0
    
    def _score_consistency(self, metrics: CheckpointMetrics) -> float:
        """Score consistency based on win rate and profit factor (0-100)"""
        # Combine win rate and profit factor for consistency score
        win_rate_score = 0
        profit_factor_score = 0
        
        # Score win rate
        if metrics.win_rate >= self.thresholds.win_rate_excellent:
            win_rate_score = 100
        elif metrics.win_rate >= self.thresholds.win_rate_good:
            ratio = (metrics.win_rate - self.thresholds.win_rate_good) / (
                self.thresholds.win_rate_excellent - self.thresholds.win_rate_good
            )
            win_rate_score = 70 + ratio * 30
        elif metrics.win_rate >= self.thresholds.win_rate_acceptable:
            ratio = (metrics.win_rate - self.thresholds.win_rate_acceptable) / (
                self.thresholds.win_rate_good - self.thresholds.win_rate_acceptable
            )
            win_rate_score = 40 + ratio * 30
        else:
            ratio = metrics.win_rate / self.thresholds.win_rate_acceptable
            win_rate_score = ratio * 40
        
        # Score profit factor
        if metrics.profit_factor >= self.thresholds.profit_factor_excellent:
            profit_factor_score = 100
        elif metrics.profit_factor >= self.thresholds.profit_factor_good:
            ratio = (metrics.profit_factor - self.thresholds.profit_factor_good) / (
                self.thresholds.profit_factor_excellent - self.thresholds.profit_factor_good
            )
            profit_factor_score = 70 + ratio * 30
        elif metrics.profit_factor >= self.thresholds.profit_factor_acceptable:
            ratio = (metrics.profit_factor - self.thresholds.profit_factor_acceptable) / (
                self.thresholds.profit_factor_good - self.thresholds.profit_factor_acceptable
            )
            profit_factor_score = 40 + ratio * 30
        elif metrics.profit_factor > 1:
            ratio = (metrics.profit_factor - 1) / (self.thresholds.profit_factor_acceptable - 1)
            profit_factor_score = ratio * 40
        else:
            profit_factor_score = 0
        
        # Weight: 60% win rate, 40% profit factor
        return win_rate_score * 0.6 + profit_factor_score * 0.4
    
    def _score_drawdown_control(self, metrics: CheckpointMetrics) -> float:
        """Score drawdown control (0-100, lower drawdown = higher score)"""
        max_dd = abs(metrics.max_drawdown_pct)
        
        if max_dd <= self.thresholds.max_dd_excellent:
            return 100
        elif max_dd <= self.thresholds.max_dd_good:
            # Inverted interpolation (lower is better)
            ratio = (self.thresholds.max_dd_good - max_dd) / (
                self.thresholds.max_dd_good - self.thresholds.max_dd_excellent
            )
            return 70 + ratio * 30
        elif max_dd <= self.thresholds.max_dd_acceptable:
            ratio = (self.thresholds.max_dd_acceptable - max_dd) / (
                self.thresholds.max_dd_acceptable - self.thresholds.max_dd_good
            )
            return 40 + ratio * 30
        elif max_dd <= 0.30:  # Up to 30% drawdown
            ratio = (0.30 - max_dd) / (0.30 - self.thresholds.max_dd_acceptable)
            return ratio * 40
        else:
            return 0
    
    def _calculate_grade(self, total_score: float) -> str:
        """Convert numeric score to letter grade"""
        if total_score >= 95:
            return "A+"
        elif total_score >= 90:
            return "A"
        elif total_score >= 85:
            return "A-"
        elif total_score >= 80:
            return "B+"
        elif total_score >= 75:
            return "B"
        elif total_score >= 70:
            return "B-"
        elif total_score >= 65:
            return "C+"
        elif total_score >= 60:
            return "C"
        elif total_score >= 55:
            return "C-"
        elif total_score >= 50:
            return "D"
        else:
            return "F"
    
    def _generate_recommendation(self, total_score: float, metrics: CheckpointMetrics,
                                expectancy_score: float, risk_score: float,
                                consistency_score: float, drawdown_score: float) -> str:
        """Generate deployment recommendation"""
        
        # Check for critical failures
        if metrics.expectancy <= 0:
            return "REJECT: Negative expectancy"
        if metrics.max_drawdown_pct > 0.30:
            return "REJECT: Excessive drawdown risk"
        if metrics.total_trades < 30:
            return "REJECT: Insufficient sample size"
        
        # Generate recommendation based on score
        if total_score >= 80:
            if metrics.total_trades >= 100:
                return "DEPLOY: Excellent performance with sufficient validation"
            else:
                return "DEPLOY-CAUTIOUS: Strong performance, monitor initial trades"
        elif total_score >= 70:
            if min(expectancy_score, risk_score, consistency_score, drawdown_score) >= 50:
                return "TEST: Good balanced performance, validate in paper trading"
            else:
                weak_areas = []
                if expectancy_score < 50:
                    weak_areas.append("expectancy")
                if risk_score < 50:
                    weak_areas.append("risk-adjusted returns")
                if consistency_score < 50:
                    weak_areas.append("consistency")
                if drawdown_score < 50:
                    weak_areas.append("drawdown control")
                return f"IMPROVE: Address weak areas - {', '.join(weak_areas)}"
        elif total_score >= 60:
            return "IMPROVE: Performance below deployment threshold"
        else:
            return "REJECT: Insufficient performance for production"
    
    def _assess_confidence_level(self, metrics: CheckpointMetrics) -> str:
        """Assess confidence level based on sample size and consistency"""
        if metrics.total_trades >= 200:
            if metrics.win_rate >= 0.50 and metrics.profit_factor >= 1.5:
                return "HIGH"
            else:
                return "MEDIUM"
        elif metrics.total_trades >= 100:
            if metrics.win_rate >= 0.55 and metrics.profit_factor >= 1.8:
                return "MEDIUM"
            else:
                return "LOW"
        else:
            return "LOW"
    
    def _analyze_performance(self, metrics: CheckpointMetrics,
                            expectancy_score: float, risk_score: float,
                            consistency_score: float, drawdown_score: float) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Analyze expectancy
        if expectancy_score >= 80:
            strengths.append(f"Excellent expectancy ({metrics.expectancy:.3f})")
        elif expectancy_score < 50:
            weaknesses.append(f"Low expectancy ({metrics.expectancy:.3f})")
        
        # Analyze risk-adjusted returns
        if risk_score >= 80:
            strengths.append(f"Strong risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
        elif risk_score < 50:
            weaknesses.append(f"Poor risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
        
        # Analyze consistency
        if consistency_score >= 80:
            strengths.append(f"High consistency (Win rate: {metrics.win_rate:.1%})")
        elif consistency_score < 50:
            weaknesses.append(f"Inconsistent performance (Win rate: {metrics.win_rate:.1%})")
        
        # Analyze drawdown
        if drawdown_score >= 80:
            strengths.append(f"Excellent risk control (Max DD: {metrics.max_drawdown_pct:.1%})")
        elif drawdown_score < 50:
            weaknesses.append(f"High drawdown risk (Max DD: {metrics.max_drawdown_pct:.1%})")
        
        # Additional specific checks
        if metrics.recovery_factor >= 3.0:
            strengths.append(f"Strong recovery factor ({metrics.recovery_factor:.1f})")
        elif metrics.recovery_factor < 1.0:
            weaknesses.append(f"Poor recovery from drawdowns ({metrics.recovery_factor:.1f})")
        
        if metrics.profit_factor >= 2.0:
            strengths.append(f"Excellent profit factor ({metrics.profit_factor:.2f})")
        elif metrics.profit_factor < 1.2:
            weaknesses.append(f"Low profit factor ({metrics.profit_factor:.2f})")
        
        return strengths, weaknesses
    
    def compare_checkpoints(self, checkpoints: List[CheckpointMetrics]) -> Dict[str, Any]:
        """
        Compare multiple checkpoints and rank them
        
        Args:
            checkpoints: List of checkpoint metrics to compare
            
        Returns:
            Comparison report with rankings and recommendations
        """
        if not checkpoints:
            return {"error": "No checkpoints to compare"}
        
        # Score all checkpoints
        scored_checkpoints = []
        for checkpoint in checkpoints:
            score = self.calculate_composite_score(checkpoint)
            scored_checkpoints.append({
                'checkpoint': checkpoint,
                'score': score
            })
        
        # Sort by total score
        scored_checkpoints.sort(key=lambda x: x['score'].total_score, reverse=True)
        
        # Generate comparison report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_checkpoints': len(checkpoints),
            'best_checkpoint': {
                'path': scored_checkpoints[0]['checkpoint'].checkpoint_path,
                'episode': scored_checkpoints[0]['checkpoint'].episode,
                'score': scored_checkpoints[0]['score'].total_score,
                'grade': scored_checkpoints[0]['score'].grade,
                'recommendation': scored_checkpoints[0]['score'].recommendation
            },
            'rankings': []
        }
        
        for i, item in enumerate(scored_checkpoints, 1):
            checkpoint = item['checkpoint']
            score = item['score']
            
            report['rankings'].append({
                'rank': i,
                'checkpoint': checkpoint.checkpoint_path,
                'episode': checkpoint.episode,
                'total_score': score.total_score,
                'grade': score.grade,
                'expectancy': checkpoint.expectancy,
                'sharpe_ratio': checkpoint.sharpe_ratio,
                'win_rate': checkpoint.win_rate,
                'max_drawdown': checkpoint.max_drawdown_pct,
                'recommendation': score.recommendation
            })
        
        # Identify improvement trend
        if len(checkpoints) >= 2:
            recent_scores = [s['score'].total_score for s in scored_checkpoints[-5:]]
            if len(recent_scores) >= 2:
                trend = "IMPROVING" if recent_scores[-1] > recent_scores[0] else "DECLINING"
                report['trend'] = trend
        
        return report


def create_metrics_from_dict(data: Dict[str, Any]) -> CheckpointMetrics:
    """
    Helper function to create CheckpointMetrics from dictionary
    
    Args:
        data: Dictionary with metric values
        
    Returns:
        CheckpointMetrics instance
    """
    return CheckpointMetrics(
        checkpoint_path=data.get('checkpoint_path', ''),
        episode=data.get('episode', 0),
        timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data.get('timestamp'), str) else data.get('timestamp', datetime.now()),
        expectancy=data.get('expectancy', 0),
        win_rate=data.get('win_rate', 0),
        profit_factor=data.get('profit_factor', 0),
        total_trades=data.get('total_trades', 0),
        sharpe_ratio=data.get('sharpe_ratio', 0),
        sortino_ratio=data.get('sortino_ratio', 0),
        calmar_ratio=data.get('calmar_ratio', 0),
        max_drawdown_pct=data.get('max_drawdown_pct', 0),
        max_drawdown_pips=data.get('max_drawdown_pips', 0),
        avg_win_pips=data.get('avg_win_pips', 0),
        avg_loss_pips=data.get('avg_loss_pips', 0),
        win_loss_ratio=data.get('win_loss_ratio', 0),
        recovery_factor=data.get('recovery_factor', 0),
        kelly_criterion=data.get('kelly_criterion'),
        var_95=data.get('var_95'),
        cvar_95=data.get('cvar_95')
    )


def main():
    """Example usage and testing"""
    # Create example metrics
    example_metrics = CheckpointMetrics(
        checkpoint_path="checkpoints/episode_13475.pth",
        episode=13475,
        timestamp=datetime.now(),
        expectancy=0.620,
        win_rate=0.611,
        profit_factor=1.85,
        total_trades=72,
        sharpe_ratio=1.75,
        sortino_ratio=2.1,
        calmar_ratio=1.5,
        max_drawdown_pct=0.154,
        max_drawdown_pips=154,
        avg_win_pips=5.3,
        avg_loss_pips=3.2,
        win_loss_ratio=1.66,
        recovery_factor=2.5
    )
    
    # Create scorer
    scorer = CompositeScorer()
    
    # Calculate score
    score = scorer.calculate_composite_score(example_metrics)
    
    # Print results
    print("\n" + "="*60)
    print("COMPOSITE SCORE ANALYSIS")
    print("="*60)
    print(f"Checkpoint: {example_metrics.checkpoint_path}")
    print(f"Episode: {example_metrics.episode}")
    print("\nSCORE BREAKDOWN:")
    print(f"  Total Score: {score.total_score:.1f}/100")
    print(f"  Grade: {score.grade}")
    print(f"\nCOMPONENTS:")
    print(f"  Expectancy: {score.expectancy_score:.1f}/100")
    print(f"  Risk-Adjusted: {score.risk_score:.1f}/100")
    print(f"  Consistency: {score.consistency_score:.1f}/100")
    print(f"  Drawdown Control: {score.drawdown_score:.1f}/100")
    print(f"\nRECOMMENDATION: {score.recommendation}")
    print(f"CONFIDENCE: {score.confidence_level}")
    
    if score.strengths:
        print("\nSTRENGTHS:")
        for strength in score.strengths:
            print(f"  ✅ {strength}")
    
    if score.weaknesses:
        print("\nWEAKNESSES:")
        for weakness in score.weaknesses:
            print(f"  ⚠️ {weakness}")
    
    print("="*60)


if __name__ == "__main__":
    main()