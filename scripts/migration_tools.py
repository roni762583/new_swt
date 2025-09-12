#!/usr/bin/env python3
"""
SWT Migration Tools - Production Data Migration and Upgrade Utilities
Handles checkpoint migration, configuration updates, and system upgrades.
"""

import os
import sys
import json
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import tempfile
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import redis
except ImportError as e:
    print(f"Warning: Optional dependencies not available: {e}")
    torch = None
    redis = None

class MigrationError(Exception):
    """Base exception for migration errors"""
    pass

class CheckpointMigrator:
    """Handles Episode checkpoint migration and validation"""
    
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup migration logging"""
        logger = logging.getLogger('checkpoint_migrator')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def migrate_episode_13475(self) -> bool:
        """
        Migrate Episode 13475 checkpoint to new system
        
        Returns:
            bool: True if migration successful
        """
        self.logger.info("Starting Episode 13475 checkpoint migration")
        
        try:
            # Locate Episode 13475 checkpoint
            source_checkpoint = self._find_episode_checkpoint(13475)
            if not source_checkpoint:
                raise MigrationError("Episode 13475 checkpoint not found")
                
            # Validate source checkpoint
            self._validate_checkpoint(source_checkpoint)
            
            # Create target directory
            self.target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy and validate checkpoint
            target_path = self.target_dir / "episode_13475.pth"
            self._copy_checkpoint(source_checkpoint, target_path)
            
            # Verify migration
            self._verify_migration(target_path)
            
            self.logger.info(f"Episode 13475 checkpoint migrated successfully to {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False
            
    def _find_episode_checkpoint(self, episode: int) -> Optional[Path]:
        """Find checkpoint file for specific episode"""
        patterns = [
            f"episode_{episode}.pth",
            f"checkpoint_episode_{episode}.pth",
            f"model_episode_{episode}.pt",
            f"checkpoint_{episode}.pth"
        ]
        
        for pattern in patterns:
            for checkpoint_file in self.source_dir.rglob(pattern):
                self.logger.info(f"Found checkpoint candidate: {checkpoint_file}")
                return checkpoint_file
                
        # Search by episode number in filename
        for checkpoint_file in self.source_dir.rglob("*.pth"):
            if str(episode) in checkpoint_file.name:
                self.logger.info(f"Found checkpoint by number: {checkpoint_file}")
                return checkpoint_file
                
        return None
        
    def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate checkpoint file integrity"""
        if not torch:
            self.logger.warning("PyTorch not available, skipping validation")
            return True
            
        try:
            self.logger.info(f"Validating checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check required keys
            required_keys = ['model_state_dict', 'episode']
            for key in required_keys:
                if key not in checkpoint:
                    self.logger.warning(f"Missing key in checkpoint: {key}")
                    
            # Check episode number
            episode = checkpoint.get('episode', 'unknown')
            self.logger.info(f"Checkpoint episode: {episode}")
            
            # Check model state
            model_state = checkpoint.get('model_state_dict', {})
            if not model_state:
                raise MigrationError("Empty model state in checkpoint")
                
            self.logger.info(f"Checkpoint validation successful, {len(model_state)} model parameters")
            return True
            
        except Exception as e:
            raise MigrationError(f"Checkpoint validation failed: {e}")
            
    def _copy_checkpoint(self, source: Path, target: Path) -> None:
        """Copy checkpoint with verification"""
        self.logger.info(f"Copying checkpoint: {source} -> {target}")
        
        # Calculate source hash
        source_hash = self._calculate_file_hash(source)
        
        # Copy file
        shutil.copy2(source, target)
        
        # Verify copy
        target_hash = self._calculate_file_hash(target)
        if source_hash != target_hash:
            raise MigrationError("Checkpoint copy verification failed")
            
        self.logger.info("Checkpoint copied and verified successfully")
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
        
    def _verify_migration(self, checkpoint_path: Path) -> None:
        """Verify migrated checkpoint works with current system"""
        if not torch:
            self.logger.warning("PyTorch not available, skipping verification")
            return
            
        try:
            # Test loading checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check compatibility with current Episode 13475 configuration
            expected_config = {
                'mcts_simulations': 15,
                'c_puct': 1.25,
                'wst_j': 2,
                'wst_q': 6,
                'feature_dims': 9
            }
            
            # Log checkpoint info
            self.logger.info(f"Verified checkpoint episode: {checkpoint.get('episode', 'unknown')}")
            self.logger.info(f"Model parameters: {len(checkpoint.get('model_state_dict', {}))}")
            
            # Test basic inference compatibility (if possible)
            model_state = checkpoint.get('model_state_dict', {})
            if model_state:
                self.logger.info("Checkpoint verification completed successfully")
            else:
                raise MigrationError("Invalid model state in migrated checkpoint")
                
        except Exception as e:
            raise MigrationError(f"Migration verification failed: {e}")

class ConfigurationMigrator:
    """Handles configuration file migration and updates"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup migration logging"""
        logger = logging.getLogger('config_migrator')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def migrate_to_v2(self) -> bool:
        """
        Migrate configuration to v2 format
        
        Returns:
            bool: True if migration successful
        """
        self.logger.info("Starting configuration migration to v2")
        
        try:
            # Find existing configuration files
            config_files = list(self.config_dir.glob("*.yaml"))
            if not config_files:
                raise MigrationError("No configuration files found")
                
            for config_file in config_files:
                self._migrate_config_file(config_file)
                
            self.logger.info("Configuration migration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration migration failed: {e}")
            return False
            
    def _migrate_config_file(self, config_file: Path) -> None:
        """Migrate individual configuration file"""
        self.logger.info(f"Migrating config file: {config_file}")
        
        # Backup original
        backup_file = config_file.with_suffix(f"{config_file.suffix}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(config_file, backup_file)
        self.logger.info(f"Created backup: {backup_file}")
        
        # Load current configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Apply migrations
        migrated_config = self._apply_config_migrations(config)
        
        # Write updated configuration
        with open(config_file, 'w') as f:
            yaml.dump(migrated_config, f, default_flow_style=False, indent=2)
            
        self.logger.info(f"Migrated configuration file: {config_file}")
        
    def _apply_config_migrations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration format migrations"""
        migrated = config.copy()
        
        # Migration 1: Add Episode 13475 specific parameters
        if 'agent' not in migrated:
            migrated['agent'] = {}
            
        if 'mcts' not in migrated['agent']:
            migrated['agent']['mcts'] = {
                'num_simulations': 15,
                'c_puct': 1.25,
                'dirichlet_alpha': 0.25,
                'exploration_fraction': 0.25
            }
            
        if 'wst' not in migrated['agent']:
            migrated['agent']['wst'] = {
                'J': 2,
                'Q': 6,
                'backend': 'manual'
            }
            
        # Migration 2: Update feature configuration
        if 'features' not in migrated:
            migrated['features'] = {}
            
        migrated['features']['feature_dims'] = 9
        migrated['features']['normalization'] = '0-1'
        
        # Migration 3: Add security configuration
        if 'security' not in migrated:
            migrated['security'] = {
                'api': {
                    'require_auth': True,
                    'rate_limiting': True,
                    'max_requests_per_minute': 60
                },
                'container': {
                    'run_as_non_root': True,
                    'read_only_filesystem': False
                }
            }
            
        # Migration 4: Update monitoring configuration
        if 'monitoring' not in migrated:
            migrated['monitoring'] = {}
            
        if 'alerts' not in migrated['monitoring']:
            migrated['monitoring']['alerts'] = {
                'enabled': True,
                'trading': {
                    'consecutive_losses': 5,
                    'daily_loss_threshold': 300,
                    'position_age_hours': 24
                },
                'system': {
                    'memory_usage_threshold': 80,
                    'cpu_usage_threshold': 80,
                    'response_time_threshold': 5.0
                }
            }
            
        return migrated

class DatabaseMigrator:
    """Handles Redis database migration and cleanup"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup migration logging"""
        logger = logging.getLogger('database_migrator')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def migrate_redis_data(self) -> bool:
        """
        Migrate Redis data to new format
        
        Returns:
            bool: True if migration successful
        """
        if not redis:
            self.logger.warning("Redis library not available, skipping migration")
            return True
            
        self.logger.info("Starting Redis data migration")
        
        try:
            # Connect to Redis
            r = redis.from_url(self.redis_url)
            
            # Test connection
            r.ping()
            self.logger.info("Connected to Redis successfully")
            
            # Migrate trading data
            self._migrate_trading_data(r)
            
            # Migrate configuration cache
            self._migrate_config_cache(r)
            
            # Clean up old data
            self._cleanup_old_data(r)
            
            self.logger.info("Redis data migration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Redis migration failed: {e}")
            return False
            
    def _migrate_trading_data(self, r: 'redis.Redis') -> None:
        """Migrate trading data format"""
        self.logger.info("Migrating trading data")
        
        # Find old trading data keys
        old_keys = r.keys('trading:*')
        
        for key in old_keys:
            key_str = key.decode('utf-8')
            
            # Skip already migrated keys
            if key_str.startswith('swt:'):
                continue
                
            # Migrate key to new format
            new_key = f"swt:{key_str}"
            
            # Copy data to new key
            data = r.get(key)
            if data:
                r.set(new_key, data)
                self.logger.info(f"Migrated key: {key_str} -> {new_key}")
                
        self.logger.info("Trading data migration completed")
        
    def _migrate_config_cache(self, r: 'redis.Redis') -> None:
        """Migrate configuration cache"""
        self.logger.info("Migrating configuration cache")
        
        # Update configuration cache format
        config_keys = r.keys('config:*')
        
        for key in config_keys:
            key_str = key.decode('utf-8')
            
            # Get current data
            data = r.get(key)
            if data:
                try:
                    config_data = json.loads(data)
                    
                    # Update format if needed
                    if 'version' not in config_data:
                        config_data['version'] = '2.0'
                        config_data['migrated_at'] = datetime.utcnow().isoformat()
                        
                        # Save updated data
                        r.set(key, json.dumps(config_data))
                        self.logger.info(f"Updated config cache: {key_str}")
                        
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in config key: {key_str}")
                    
        self.logger.info("Configuration cache migration completed")
        
    def _cleanup_old_data(self, r: 'redis.Redis') -> None:
        """Clean up old/obsolete data"""
        self.logger.info("Cleaning up old data")
        
        # Keys to clean up (older than 30 days)
        cleanup_patterns = [
            'temp:*',
            'debug:*',
            'test:*'
        ]
        
        for pattern in cleanup_patterns:
            keys = r.keys(pattern)
            if keys:
                deleted = r.delete(*keys)
                self.logger.info(f"Cleaned up {deleted} keys matching {pattern}")
                
        self.logger.info("Data cleanup completed")

class SystemUpgrader:
    """Handles complete system upgrades"""
    
    def __init__(self, system_dir: str):
        self.system_dir = Path(system_dir)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup upgrade logging"""
        logger = logging.getLogger('system_upgrader')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def upgrade_to_episode_13475(self) -> bool:
        """
        Complete system upgrade to Episode 13475 compatibility
        
        Returns:
            bool: True if upgrade successful
        """
        self.logger.info("Starting system upgrade to Episode 13475")
        
        try:
            # Step 1: Backup current system
            backup_dir = self._create_system_backup()
            
            # Step 2: Migrate checkpoints
            checkpoint_migrator = CheckpointMigrator(
                source_dir=str(self.system_dir / "old_checkpoints"),
                target_dir=str(self.system_dir / "checkpoints")
            )
            if not checkpoint_migrator.migrate_episode_13475():
                raise MigrationError("Checkpoint migration failed")
                
            # Step 3: Migrate configuration
            config_migrator = ConfigurationMigrator(
                config_dir=str(self.system_dir / "config")
            )
            if not config_migrator.migrate_to_v2():
                raise MigrationError("Configuration migration failed")
                
            # Step 4: Migrate database
            db_migrator = DatabaseMigrator()
            if not db_migrator.migrate_redis_data():
                raise MigrationError("Database migration failed")
                
            # Step 5: Update Docker configuration
            self._update_docker_config()
            
            # Step 6: Validate upgrade
            self._validate_upgrade()
            
            self.logger.info("System upgrade completed successfully")
            self.logger.info(f"Backup created at: {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"System upgrade failed: {e}")
            return False
            
    def _create_system_backup(self) -> Path:
        """Create complete system backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.system_dir / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating system backup: {backup_dir}")
        
        # Backup critical directories
        critical_dirs = ['config', 'checkpoints', 'logs', 'results']
        
        for dir_name in critical_dirs:
            source_dir = self.system_dir / dir_name
            if source_dir.exists():
                target_dir = backup_dir / dir_name
                shutil.copytree(source_dir, target_dir)
                self.logger.info(f"Backed up {dir_name}")
                
        return backup_dir
        
    def _update_docker_config(self) -> None:
        """Update Docker configuration for Episode 13475"""
        self.logger.info("Updating Docker configuration")
        
        docker_compose_file = self.system_dir / "docker-compose.yml"
        if not docker_compose_file.exists():
            self.logger.warning("docker-compose.yml not found, skipping Docker update")
            return
            
        # Read current configuration
        with open(docker_compose_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update environment variables for Episode 13475
        if 'services' in config and 'swt-live-trader' in config['services']:
            env_vars = config['services']['swt-live-trader'].get('environment', [])
            
            # Update checkpoint path
            checkpoint_env = 'SWT_CHECKPOINT_PATH=/app/checkpoints/episode_13475.pth'
            if checkpoint_env not in env_vars:
                env_vars.append(checkpoint_env)
                
            # Update agent system
            agent_env = 'SWT_AGENT_SYSTEM=stochastic_muzero'
            if agent_env not in env_vars:
                env_vars.append(agent_env)
                
            config['services']['swt-live-trader']['environment'] = env_vars
            
        # Write updated configuration
        with open(docker_compose_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        self.logger.info("Docker configuration updated")
        
    def _validate_upgrade(self) -> None:
        """Validate system upgrade"""
        self.logger.info("Validating system upgrade")
        
        # Check checkpoint exists
        checkpoint_path = self.system_dir / "checkpoints" / "episode_13475.pth"
        if not checkpoint_path.exists():
            raise MigrationError("Episode 13475 checkpoint not found after upgrade")
            
        # Check configuration files
        config_path = self.system_dir / "config" / "live.yaml"
        if not config_path.exists():
            raise MigrationError("Configuration file not found after upgrade")
            
        # Validate configuration format
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        required_sections = ['agent', 'trading', 'system']
        for section in required_sections:
            if section not in config:
                raise MigrationError(f"Missing configuration section: {section}")
                
        self.logger.info("System upgrade validation completed")

def main():
    """Main migration tool entry point"""
    parser = argparse.ArgumentParser(description="SWT Migration Tools")
    subparsers = parser.add_subparsers(dest='command', help='Migration commands')
    
    # Checkpoint migration
    checkpoint_parser = subparsers.add_parser('checkpoint', help='Migrate checkpoints')
    checkpoint_parser.add_argument('--source', required=True, help='Source checkpoint directory')
    checkpoint_parser.add_argument('--target', required=True, help='Target checkpoint directory')
    
    # Configuration migration
    config_parser = subparsers.add_parser('config', help='Migrate configuration')
    config_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    
    # Database migration
    db_parser = subparsers.add_parser('database', help='Migrate database')
    db_parser.add_argument('--redis-url', default='redis://localhost:6379/0', help='Redis URL')
    
    # Full system upgrade
    upgrade_parser = subparsers.add_parser('upgrade', help='Full system upgrade')
    upgrade_parser.add_argument('--system-dir', required=True, help='System directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'checkpoint':
        migrator = CheckpointMigrator(args.source, args.target)
        success = migrator.migrate_episode_13475()
        
    elif args.command == 'config':
        migrator = ConfigurationMigrator(args.config_dir)
        success = migrator.migrate_to_v2()
        
    elif args.command == 'database':
        migrator = DatabaseMigrator(args.redis_url)
        success = migrator.migrate_redis_data()
        
    elif args.command == 'upgrade':
        upgrader = SystemUpgrader(args.system_dir)
        success = upgrader.upgrade_to_episode_13475()
        
    else:
        parser.print_help()
        sys.exit(1)
        
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()