"""
SWT Configuration Manager
CRITICAL: Ensures identical feature processing between training and live trading

Loads and validates YAML configurations with:
- Episode 13475 parameter verification
- Environment variable overrides  
- Configuration validation and merging
- Process control integration
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from copy import deepcopy

from .types import (
    ProcessConfig, ProcessLimits, ProcessMonitoring,
    FeatureProcessingConfig, PositionFeatures, WSTransformConfig,
    TradingConfig, MCTSParameters, NormalizationParams
)

logger = logging.getLogger(__name__)


@dataclass
class SWTConfig:
    """
    Master configuration class for SWT trading system
    CRITICAL: Ensures Episode 13475 parameter compatibility
    """
    # System identification
    system_name: str = "SWT-Episode13475-Compatible"
    version: str = "1.0.0"
    episode_reference: str = "13475"  # Reference checkpoint
    
    # Core component configurations
    process_config: ProcessConfig = field(default_factory=ProcessConfig)
    feature_config: FeatureProcessingConfig = field(default_factory=FeatureProcessingConfig)
    trading_config: TradingConfig = field(default_factory=TradingConfig)
    mcts_config: MCTSParameters = field(default_factory=MCTSParameters)
    
    # Episode 13475 compatibility flags
    episode_13475_verified: bool = False
    parameter_validation_strict: bool = True
    
    # Runtime configuration
    runtime_config: Dict[str, Any] = field(default_factory=lambda: {
        "device": "auto",
        "log_level": "INFO", 
        "data_path": "data/",
        "checkpoint_path": "checkpoints/",
        "log_path": "logs/"
    })
    
    def verify_episode_13475_compatibility(self) -> bool:
        """
        CRITICAL: Verify configuration matches Episode 13475 exactly
        
        Returns:
            True if configuration is Episode 13475 compatible
        """
        try:
            # Verify MCTS parameters
            if self.mcts_config.num_simulations != 15:
                logger.error(f"❌ MCTS simulations mismatch: expected 15, got {self.mcts_config.num_simulations}")
                return False
                
            if abs(self.mcts_config.c_puct - 1.25) > 0.001:
                logger.error(f"❌ MCTS C_PUCT mismatch: expected 1.25, got {self.mcts_config.c_puct}")
                return False
                
            if abs(self.mcts_config.temperature - 1.0) > 0.001:
                logger.error(f"❌ MCTS temperature mismatch: expected 1.0, got {self.mcts_config.temperature}")
                return False
            
            # Verify position features (CRITICAL)
            if self.feature_config.position_features.dimension != 9:
                logger.error(f"❌ Position feature dimension mismatch: expected 9, got {self.feature_config.position_features.dimension}")
                return False
            
            # Verify trading parameters
            if abs(self.trading_config.min_confidence - 0.35) > 0.001:
                logger.error(f"❌ Confidence threshold mismatch: expected 0.35, got {self.trading_config.min_confidence}")
                return False
                
            # Verify WST parameters
            wst = self.feature_config.wst_config
            if wst.J != 2 or wst.Q != 6:
                logger.error(f"❌ WST parameter mismatch: expected J=2, Q=6, got J={wst.J}, Q={wst.Q}")
                return False
            
            logger.info("✅ Episode 13475 compatibility verified")
            self.episode_13475_verified = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Episode 13475 compatibility check failed: {e}")
            return False
    
    def validate(self) -> None:
        """Validate entire configuration for Episode 13475 compatibility"""
        try:
            # Validate individual components
            self.process_config.validate()
            self.feature_config.validate()
            self.trading_config.validate()
            self.mcts_config.validate()
            
            # CRITICAL: Verify Episode 13475 compatibility
            if self.parameter_validation_strict and not self.verify_episode_13475_compatibility():
                raise ValueError("Configuration is not compatible with Episode 13475")
            
            # Validate paths exist
            self._validate_paths()
            
            logger.info("✅ Configuration validation passed - Episode 13475 compatible")
            
        except Exception as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _validate_process_limits(self) -> None:
        """Validate process control limits are properly configured"""
        limits = self.process_config.limits
        
        if limits.max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive, got {limits.max_episodes}")
            
        if limits.max_runtime_hours <= 0:
            raise ValueError(f"max_runtime_hours must be positive, got {limits.max_runtime_hours}")
            
        if limits.max_episodes > 20000:
            logger.warning(f"⚠️ max_episodes ({limits.max_episodes}) exceeds recommended 20,000")
    
    def _validate_paths(self) -> None:
        """Validate and create necessary directories"""
        paths_to_check = [
            self.runtime_config["data_path"],
            self.runtime_config["checkpoint_path"],
            self.runtime_config["log_path"]
        ]
        
        for path_str in paths_to_check:
            path = Path(path_str)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Created directory: {path}")
    
    def _validate_feature_compatibility(self) -> None:
        """Validate feature processing compatibility with Episode 13475"""
        pos_features = self.feature_config.position_features
        
        # Check exact normalization parameters
        norm = pos_features.normalization_params
        expected_values = {
            'duration_max_bars': 720.0,
            'pnl_scale_factor': 100.0,
            'price_change_scale_factor': 50.0,
            'drawdown_scale_factor': 50.0,
            'accumulated_dd_scale': 100.0,
            'bars_since_dd_scale': 60.0
        }
        
        for param, expected in expected_values.items():
            actual = getattr(norm, param)
            if abs(actual - expected) > 0.001:
                raise ValueError(f"Feature normalization mismatch: {param} expected {expected}, got {actual}")
    
    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get effective configuration for the SWT system
        Ensures Episode 13475 compatibility
        """
        return {
            "system": {
                "name": self.system_name,
                "version": self.version,
                "episode_reference": self.episode_reference,
                "verified": self.episode_13475_verified
            },
            "process": self.process_config,
            "features": self.feature_config,
            "trading": self.trading_config,
            "mcts": self.mcts_config,
            "runtime": self.runtime_config
        }
    
    def force_episode_13475_mode(self) -> None:
        """
        Force configuration to match Episode 13475 exactly
        CRITICAL: Use this to ensure parameter compatibility
        """
        logger.info("🔧 Forcing Episode 13475 compatibility mode")
        
        # Set MCTS parameters exactly
        self.mcts_config.num_simulations = 15
        self.mcts_config.c_puct = 1.25
        self.mcts_config.temperature = 1.0
        self.mcts_config.discount_factor = 0.997
        
        # Set trading parameters exactly  
        self.trading_config.trade_volume = 1  # EXACT: 1-unit positions
        self.trading_config.min_confidence = 0.35  # EXACT: Episode 13475 max was 38.5%
        self.trading_config.max_spread_pips = 8.0
        
        # Ensure position features are 9D
        self.feature_config.position_features.dimension = 9
        
        # Set exact normalization parameters
        norm = self.feature_config.position_features.normalization_params
        norm.duration_max_bars = 720.0
        norm.pnl_scale_factor = 100.0
        norm.price_change_scale_factor = 50.0
        norm.drawdown_scale_factor = 50.0
        norm.accumulated_dd_scale = 100.0
        norm.bars_since_dd_scale = 60.0
        
        # Set WST parameters exactly
        wst = self.feature_config.wst_config
        wst.J = 2
        wst.Q = 6
        wst.backend = "fallback"
        wst.max_order = 2
        wst.output_dim = 128
        
        # Re-validate after forcing parameters
        self.validate()
        logger.info("✅ Episode 13475 compatibility mode activated")


class ConfigManager:
    """
    Configuration manager for loading and validating SWT configurations
    CRITICAL: Ensures Episode 13475 parameter compatibility
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.loaded_configs: Dict[str, Dict[str, Any]] = {}
        self.merged_config: Optional[SWTConfig] = None
        
        # Verify config directory exists
        if not self.config_dir.exists():
            raise ValueError(f"Configuration directory does not exist: {self.config_dir}")
            
        logger.info(f"📁 ConfigManager initialized with config_dir: {self.config_dir}")
    
    def load_config(self, strict_validation: bool = True) -> SWTConfig:
        """
        Load and validate Episode 13475 compatible configuration
        
        Args:
            strict_validation: Whether to enforce strict Episode 13475 validation
            
        Returns:
            Validated SWTConfig instance
        """
        try:
            # Load all required config files
            config_files = [
                "features.yaml",
                "trading.yaml", 
                "process_limits.yaml",
                "model.yaml"
            ]
            
            # Load and merge all config files
            merged_data = {}
            for config_file in config_files:
                config_data = self._load_yaml_config(config_file)
                merged_data = self._deep_merge(merged_data, config_data)
                logger.info(f"📄 Loaded config file: {config_file}")
            
            # Apply environment variable overrides
            merged_data = self._apply_env_overrides(merged_data)
            
            # Create SWTConfig instance
            config = self._dict_to_config(merged_data)
            
            # Set validation strictness
            config.parameter_validation_strict = strict_validation
            
            # Validate configuration
            config.validate()
            
            self.merged_config = config
            logger.info("✅ Configuration loaded and validated - Episode 13475 compatible")
            
            return config
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def save_config(self, config: SWTConfig, filename: str) -> None:
        """
        Save configuration to YAML file with Episode 13475 verification
        
        Args:
            config: Configuration to save
            filename: Output filename
        """
        try:
            # Verify compatibility before saving
            if not config.verify_episode_13475_compatibility():
                logger.warning("⚠️ Saving configuration that is not Episode 13475 compatible")
            
            output_path = self.config_dir / filename
            config_dict = self._config_to_dict(config)
            
            with open(output_path, 'w') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"💾 Configuration saved to: {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def create_episode_13475_config(self) -> SWTConfig:
        """
        Create a configuration guaranteed to be Episode 13475 compatible
        
        Returns:
            SWTConfig with Episode 13475 exact parameters
        """
        logger.info("🔧 Creating Episode 13475 compatible configuration")
        
        # Create base configuration
        config = SWTConfig()
        
        # Force Episode 13475 compatibility
        config.force_episode_13475_mode()
        
        # Validate
        config.validate()
        
        logger.info("✅ Episode 13475 compatible configuration created")
        return config
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load single YAML configuration file"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            logger.warning(f"⚠️ Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
                self.loaded_configs[filename] = config_data
                return config_data
                
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in {filename}: {str(e)}",
                context={"file": str(config_path)}
            )
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_overrides = {}
        
        # Look for SWT_* environment variables
        for key, value in os.environ.items():
            if key.startswith('SWT_'):
                # Convert SWT_AGENT_SYSTEM to nested dict path
                config_path = key[4:].lower().split('_')  # Remove 'SWT_' prefix
                self._set_nested_value(env_overrides, config_path, self._parse_env_value(value))
        
        if env_overrides:
            logger.info(f"🌍 Applied {len(env_overrides)} environment variable overrides")
            config_data = self._deep_merge(config_data, env_overrides)
        
        return config_data
    
    def _set_nested_value(self, d: Dict[str, Any], path: List[str], value: Any) -> None:
        """Set nested dictionary value using path list"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SWTConfig:
        """Convert dictionary to SWTConfig instance"""
        try:
            # Create base configuration
            config = SWTConfig()
            
            # Load process configuration
            if 'limits' in config_dict:
                config.process_config = self._dict_to_process_config(config_dict)
            
            # Load feature configuration
            if any(key in config_dict for key in ['wst', 'position_features']):
                config.feature_config = self._dict_to_feature_config(config_dict)
            
            # Load trading configuration
            if 'trading' in config_dict:
                config.trading_config = self._dict_to_trading_config(config_dict['trading'])
            
            # Load MCTS configuration
            if 'mcts' in config_dict:
                config.mcts_config = self._dict_to_mcts_config(config_dict['mcts'])
            
            # Load runtime configuration
            if 'runtime' in config_dict:
                config.runtime_config = config_dict['runtime']
                
            return config
            
        except Exception as e:
            error_msg = f"Failed to convert dictionary to SWTConfig: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _dict_to_process_config(self, config_dict: Dict[str, Any]) -> ProcessConfig:
        """Convert dictionary to ProcessConfig"""
        return ProcessConfig(
            limits=ProcessLimits(**config_dict.get('limits', {})),
            monitoring=ProcessMonitoring(**config_dict.get('monitoring', {}))
        )
    
    def _dict_to_mcts_config(self, mcts_dict: Dict[str, Any]) -> MCTSParameters:
        """Convert dictionary to MCTSParameters"""
        return MCTSParameters(
            num_simulations=mcts_dict.get('num_simulations', 15),
            c_puct=mcts_dict.get('c_puct', 1.25),
            temperature=mcts_dict.get('temperature', 1.0),
            discount_factor=mcts_dict.get('discount_factor', 0.997),
            max_search_time_ms=mcts_dict.get('max_search_time_ms', 1000)
        )
    
    def _dict_to_feature_config(self, config_dict: Dict[str, Any]) -> FeatureProcessingConfig:
        """Convert dictionary to FeatureProcessingConfig"""
        wst_config = WSTransformConfig(**config_dict.get('wst', {}))
        
        position_data = config_dict.get('position_features', {})
        norm_data = position_data.get('normalization', {})
        norm_params = NormalizationParams(**norm_data) if norm_data else NormalizationParams()
        
        position_config = PositionFeatures(
            dimension=position_data.get('dimension', 9),
            normalization_params=norm_params
        )
        
        return FeatureProcessingConfig(
            wst_config=wst_config,
            position_features=position_config
        )
    
    def _dict_to_trading_config(self, trading_dict: Dict[str, Any]) -> TradingConfig:
        """Convert dictionary to TradingConfig"""
        return TradingConfig(
            trade_volume=trading_dict.get('trade_volume', 1),
            min_confidence=trading_dict.get('min_confidence', 0.35),
            max_spread_pips=trading_dict.get('max_spread_pips', 8.0)
        )
    
    def _config_to_dict(self, config: SWTConfig) -> Dict[str, Any]:
        """Convert SWTConfig to dictionary for serialization"""
        return {
            'system': {
                'name': config.system_name,
                'version': config.version,
                'episode_reference': config.episode_reference,
                'verified': config.episode_13475_verified
            },
            'process': config.process_config.__dict__,
            'features': {
                'wst': config.feature_config.wst_config.__dict__,
                'position_features': config.feature_config.position_features.__dict__
            },
            'trading': config.trading_config.__dict__,
            'mcts': config.mcts_config.__dict__,
            'runtime': config.runtime_config
        }
    
# Configuration loading utility functions
def load_episode_13475_config(config_dir: str = "config") -> SWTConfig:
    """
    Load Episode 13475 compatible configuration
    
    Args:
        config_dir: Configuration directory path
        
    Returns:
        Validated SWTConfig instance
    """
    manager = ConfigManager(config_dir)
    return manager.load_config(strict_validation=True)

def create_default_config() -> SWTConfig:
    """
    Create default Episode 13475 compatible configuration
    
    Returns:
        SWTConfig with default Episode 13475 parameters
    """
    manager = ConfigManager()
    return manager.create_episode_13475_config()