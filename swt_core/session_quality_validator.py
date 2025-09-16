#!/usr/bin/env python3
"""
Session Quality Validator for SWT Trading System
Pre-validates trading sessions for data quality issues
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class QualityCheckResult:
    """Result of a quality check"""
    passed: bool
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    details: Optional[Dict] = None

class SessionQualityReport(NamedTuple):
    """Complete quality report for a session"""
    session_id: str
    passed: bool
    checks: List[QualityCheckResult]
    score: float  # 0-100 quality score

class SessionQualityValidator:
    """
    Validates session data quality before training/trading begins

    Quality checks:
    - Gap detection (>10 min gaps)
    - Weekend data detection
    - Price outlier detection (>20 pips/min)
    - Volume anomaly detection
    - Data completeness
    - Timestamp consistency
    """

    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> dict:
        """Default validation configuration"""
        return {
            'max_gap_minutes': 10,
            'max_pip_per_minute': 20,  # Reject sessions with >20 pips/min bars
            'min_volume_threshold': 1,  # Minimum volume per bar
            'outlier_percentile': 99.5,  # Flag top 0.5% as outliers
            'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            'quality_score_threshold': 80.0  # Minimum score to pass
        }

    def validate_session(self,
                        session_data: pd.DataFrame,
                        session_id: str) -> SessionQualityReport:
        """
        Comprehensive session quality validation

        Args:
            session_data: DataFrame with OHLCV data
            session_id: Unique session identifier

        Returns:
            SessionQualityReport with pass/fail and detailed checks
        """
        checks = []

        # 1. Data completeness check
        checks.append(self._check_data_completeness(session_data))

        # 2. Gap detection
        checks.append(self._check_time_gaps(session_data))

        # 3. Weekend data detection
        checks.append(self._check_weekend_data(session_data))

        # 4. Price outlier detection
        checks.append(self._check_price_outliers(session_data))

        # 5. Volume anomaly detection
        checks.append(self._check_volume_anomalies(session_data))

        # 6. Timestamp consistency
        checks.append(self._check_timestamp_consistency(session_data))

        # Calculate quality score and overall pass/fail
        score = self._calculate_quality_score(checks)
        overall_passed = score >= self.config['quality_score_threshold']

        # Fail immediately on critical issues
        critical_failures = [c for c in checks if c.severity == 'critical' and not c.passed]
        if critical_failures:
            overall_passed = False

        return SessionQualityReport(
            session_id=session_id,
            passed=overall_passed,
            checks=checks,
            score=score
        )

    def _check_data_completeness(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check if all required columns are present and have no missing values"""
        missing_cols = []
        for col in self.config['required_columns']:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            return QualityCheckResult(
                passed=False,
                severity='critical',
                message=f"Missing required columns: {missing_cols}",
                details={'missing_columns': missing_cols}
            )

        # Check for NaN values
        null_counts = df[self.config['required_columns']].isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            return QualityCheckResult(
                passed=False,
                severity='error',
                message=f"Found {total_nulls} null values in data",
                details={'null_counts': null_counts.to_dict()}
            )

        return QualityCheckResult(
            passed=True,
            severity='info',
            message=f"Data completeness OK: {len(df)} bars, all columns present"
        )

    def _check_time_gaps(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for time gaps larger than max_gap_minutes"""
        if 'timestamp' not in df.columns:
            return QualityCheckResult(
                passed=False,
                severity='critical',
                message="No timestamp column for gap detection"
            )

        timestamps = pd.to_datetime(df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds() / 60  # Minutes
        large_gaps = time_diffs > self.config['max_gap_minutes']

        if large_gaps.any():
            gap_count = large_gaps.sum()
            max_gap = time_diffs.max()
            return QualityCheckResult(
                passed=False,
                severity='critical',
                message=f"Found {gap_count} gaps > {self.config['max_gap_minutes']}min (max: {max_gap:.1f}min)",
                details={'gap_count': gap_count, 'max_gap_minutes': max_gap}
            )

        return QualityCheckResult(
            passed=True,
            severity='info',
            message="No significant time gaps detected"
        )

    def _check_weekend_data(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for weekend data (Friday 21:00 GMT to Sunday 21:00 GMT)"""
        if 'timestamp' not in df.columns:
            return QualityCheckResult(
                passed=False,
                severity='error',
                message="No timestamp column for weekend detection"
            )

        weekend_count = 0
        timestamps = pd.to_datetime(df['timestamp'])

        for timestamp in timestamps:
            weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
            hour = timestamp.hour
            # Friday after 21:00 or Saturday or Sunday before 21:00
            if (weekday == 4 and hour >= 21) or weekday == 5 or (weekday == 6 and hour < 21):
                weekend_count += 1

        if weekend_count > 0:
            weekend_pct = (weekend_count / len(df)) * 100
            return QualityCheckResult(
                passed=False,
                severity='critical',
                message=f"Found {weekend_count} weekend bars ({weekend_pct:.1f}%)",
                details={'weekend_bars': weekend_count, 'weekend_percentage': weekend_pct}
            )

        return QualityCheckResult(
            passed=True,
            severity='info',
            message="No weekend data detected"
        )

    def _check_price_outliers(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for extreme price movements that indicate bad data"""
        if not all(col in df.columns for col in ['high', 'low']):
            return QualityCheckResult(
                passed=False,
                severity='error',
                message="Missing high/low columns for outlier detection"
            )

        # Calculate pip ranges (assuming GBPJPY - 0.01 = 1 pip)
        pip_ranges = (df['high'] - df['low']) * 100  # Convert to pips

        # Check for bars exceeding max_pip_per_minute
        outliers = pip_ranges > self.config['max_pip_per_minute']
        outlier_count = outliers.sum()

        if outlier_count > 0:
            max_range = pip_ranges.max()
            outlier_pct = (outlier_count / len(df)) * 100

            # Critical if >1% of bars are outliers
            severity = 'critical' if outlier_pct > 1.0 else 'warning'
            passed = outlier_pct <= 1.0

            return QualityCheckResult(
                passed=passed,
                severity=severity,
                message=f"Found {outlier_count} outlier bars ({outlier_pct:.1f}%) - max range: {max_range:.1f} pips",
                details={'outlier_count': outlier_count, 'max_pip_range': max_range, 'outlier_percentage': outlier_pct}
            )

        return QualityCheckResult(
            passed=True,
            severity='info',
            message=f"Price outliers OK - max range: {pip_ranges.max():.1f} pips"
        )

    def _check_volume_anomalies(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for volume anomalies"""
        if 'volume' not in df.columns:
            return QualityCheckResult(
                passed=True,  # Volume not critical for forex
                severity='warning',
                message="No volume column - skipping volume checks"
            )

        # Check for zero/negative volume
        invalid_volume = (df['volume'] <= 0).sum()
        if invalid_volume > 0:
            return QualityCheckResult(
                passed=False,
                severity='warning',
                message=f"Found {invalid_volume} bars with invalid volume",
                details={'invalid_volume_count': invalid_volume}
            )

        return QualityCheckResult(
            passed=True,
            severity='info',
            message="Volume data OK"
        )

    def _check_timestamp_consistency(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check timestamp consistency and ordering"""
        if 'timestamp' not in df.columns:
            return QualityCheckResult(
                passed=False,
                severity='critical',
                message="No timestamp column"
            )

        timestamps = pd.to_datetime(df['timestamp'])

        # Check for duplicates
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            return QualityCheckResult(
                passed=False,
                severity='error',
                message=f"Found {duplicates} duplicate timestamps",
                details={'duplicate_count': duplicates}
            )

        # Check chronological ordering
        if not timestamps.is_monotonic_increasing:
            return QualityCheckResult(
                passed=False,
                severity='error',
                message="Timestamps not in chronological order"
            )

        return QualityCheckResult(
            passed=True,
            severity='info',
            message="Timestamp consistency OK"
        )

    def _calculate_quality_score(self, checks: List[QualityCheckResult]) -> float:
        """Calculate overall quality score (0-100)"""
        if not checks:
            return 0.0

        # Weight checks by severity
        weights = {'info': 1.0, 'warning': 0.8, 'error': 0.5, 'critical': 0.0}

        total_weight = 0.0
        weighted_sum = 0.0

        for check in checks:
            weight = weights.get(check.severity, 0.5)
            total_weight += 1.0
            if check.passed:
                weighted_sum += 1.0
            else:
                weighted_sum += weight

        return (weighted_sum / total_weight) * 100.0 if total_weight > 0 else 0.0

def validate_session_file(csv_path: str,
                         start_idx: int,
                         length: int,
                         session_id: str = None) -> SessionQualityReport:
    """
    Validate a session extracted from CSV file

    Args:
        csv_path: Path to CSV file
        start_idx: Starting index for session
        length: Number of bars in session
        session_id: Session identifier

    Returns:
        SessionQualityReport
    """
    if not Path(csv_path).exists():
        return SessionQualityReport(
            session_id=session_id or "unknown",
            passed=False,
            checks=[QualityCheckResult(False, 'critical', f"Data file not found: {csv_path}")],
            score=0.0
        )

    try:
        # Load session data
        df = pd.read_csv(csv_path)
        end_idx = min(start_idx + length, len(df))
        session_data = df.iloc[start_idx:end_idx]

        if len(session_data) < length * 0.9:  # Allow 10% tolerance
            return SessionQualityReport(
                session_id=session_id or f"session_{start_idx}",
                passed=False,
                checks=[QualityCheckResult(False, 'critical', f"Insufficient data: got {len(session_data)}, expected {length}")],
                score=0.0
            )

        # Validate session
        validator = SessionQualityValidator()
        return validator.validate_session(session_data, session_id or f"session_{start_idx}")

    except Exception as e:
        return SessionQualityReport(
            session_id=session_id or "unknown",
            passed=False,
            checks=[QualityCheckResult(False, 'critical', f"Validation error: {e}")],
            score=0.0
        )

# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test session quality validation")
    parser.add_argument("--data", default="data/GBPJPY_M1_REAL_2022-2025.csv", help="CSV data file")
    parser.add_argument("--start", type=int, default=10000, help="Session start index")
    parser.add_argument("--length", type=int, default=360, help="Session length (6 hours)")
    args = parser.parse_args()

    # Test validation
    report = validate_session_file(args.data, args.start, args.length, f"test_session_{args.start}")

    print(f"\n🔍 Session Quality Report: {report.session_id}")
    print(f"📊 Overall Score: {report.score:.1f}/100")
    print(f"✅ Passed: {report.passed}")

    print(f"\n📋 Quality Checks:")
    for i, check in enumerate(report.checks, 1):
        status = "✅" if check.passed else "❌"
        print(f"  {i}. {status} [{check.severity.upper()}] {check.message}")
        if check.details:
            for key, value in check.details.items():
                print(f"     • {key}: {value}")