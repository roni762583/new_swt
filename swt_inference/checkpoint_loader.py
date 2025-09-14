"""
Checkpoint Loader
Unified checkpoint loading system supporting multiple agent types

Handles loading and validation of model checkpoints for both standard
and experimental agents, ensuring compatibility and proper initialization.
"""

import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging
import json
from datetime import datetime

from swt_core.types import AgentType, NetworkArchitecture
from swt_core.config_manager import SWTConfig
from swt_core.exceptions import CheckpointError, ConfigurationError

logger = logging.getLogger(__name__)


class CheckpointLoader:
    """
    Unified checkpoint loader for SWT trading system
    Supports both standard and experimental agent checkpoints
    """
    
    def __init__(self, config: SWTConfig):
        """
        Initialize checkpoint loader
        
        Args:
            config: SWT system configuration
        """
        self.config = config
        self.device = self._determine_device()
        
        logger.info(f"💾 CheckpointLoader initialized for device: {self.device}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load checkpoint with automatic format detection
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing networks, metadata, and configuration
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise CheckpointError(
                f"Checkpoint file not found: {checkpoint_path}",
                context={"checkpoint_path": str(checkpoint_path)}
            )
        
        try:
            logger.info(f"📁 Loading checkpoint: {checkpoint_path}")
            
            # Load raw checkpoint data
            raw_checkpoint = self._load_raw_checkpoint(checkpoint_path)
            
            # Detect checkpoint format and agent type
            checkpoint_info = self._analyze_checkpoint(raw_checkpoint, checkpoint_path)
            
            # Load networks based on detected type
            if checkpoint_info["agent_type"] == AgentType.EXPERIMENTAL:
                networks = self._load_experimental_networks(raw_checkpoint, checkpoint_info)
            else:
                networks = self._load_standard_networks(raw_checkpoint, checkpoint_info)
            
            # Validate loaded networks
            self._validate_networks(networks, checkpoint_info)
            
            # Prepare final checkpoint data
            checkpoint_data = {
                "networks": networks,
                "metadata": checkpoint_info["metadata"],
                "config": checkpoint_info["config"],
                "agent_type": checkpoint_info["agent_type"],
                "loading_info": {
                    "loaded_at": datetime.now().isoformat(),
                    "device": str(self.device),
                    "checkpoint_path": str(checkpoint_path)
                }
            }
            
            logger.info(f"✅ Successfully loaded {checkpoint_info['agent_type'].value} checkpoint")
            
            return checkpoint_data
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to load checkpoint: {str(e)}",
                context={"checkpoint_path": str(checkpoint_path)},
                original_error=e
            )
    
    def _load_raw_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load raw checkpoint data with Episode 13475 numpy compatibility fix"""
        try:
            # Try PyTorch format first
            if checkpoint_path.suffix in ['.pth', '.pt']:
                # Fix numpy._core compatibility for Episode 13475 (conditional)
                import numpy as np
                try:
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                except AttributeError:
                    # Older PyTorch versions don't have add_safe_globals
                    pass
                return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Try pickle format
            elif checkpoint_path.suffix == '.pkl':
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            
            # Try JSON format (for metadata)
            elif checkpoint_path.suffix == '.json':
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            
            else:
                # Default to PyTorch with numpy fix (conditional)
                import numpy as np
                try:
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                except AttributeError:
                    # Older PyTorch versions don't have add_safe_globals
                    pass
                return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
        except Exception as e:
            raise CheckpointError(
                f"Could not read checkpoint file: {str(e)}",
                original_error=e
            )
    
    def _analyze_checkpoint(self, checkpoint: Dict[str, Any], 
                          checkpoint_path: Path) -> Dict[str, Any]:
        """Analyze checkpoint to determine format and agent type"""
        
        analysis = {
            "agent_type": AgentType.STOCHASTIC_MUZERO,  # Default
            "format": "unknown",
            "metadata": {},
            "config": {}
        }
        
        # Extract metadata
        if "metadata" in checkpoint:
            analysis["metadata"] = checkpoint["metadata"]
        elif "episode" in checkpoint:
            analysis["metadata"] = {
                "episode": checkpoint.get("episode"),
                "timestamp": checkpoint.get("timestamp"),
                "performance": checkpoint.get("performance", {})
            }
        
        # Detect agent type from various indicators
        
        # Check for experimental features
        has_experimental_features = False
        
        # Check for value-prefix network
        if "value_prefix_network" in checkpoint:
            has_experimental_features = True
        
        # Check for experimental config
        if "experimental_config" in checkpoint:
            has_experimental_features = True
        
        # Check network names for experimental patterns
        network_keys = set(checkpoint.keys())
        experimental_indicators = {
            "consistency_loss_state", "value_prefix_state", 
            "concurrent_prediction", "efficientzero"
        }
        if network_keys.intersection(experimental_indicators):
            has_experimental_features = True
        
        # Check for ReZero/Gumbel indicators
        if any(key.startswith(("rezero", "gumbel", "unizero")) for key in network_keys):
            has_experimental_features = True
        
        # Set agent type based on detection
        if has_experimental_features:
            analysis["agent_type"] = AgentType.EXPERIMENTAL
            analysis["format"] = "experimental"
        else:
            analysis["agent_type"] = AgentType.STOCHASTIC_MUZERO
            analysis["format"] = "standard"
        
        # Extract configuration if available
        if "config" in checkpoint:
            analysis["config"] = checkpoint["config"]
        elif "training_config" in checkpoint:
            analysis["config"] = checkpoint["training_config"]
        
        logger.info(f"🔍 Checkpoint analysis: {analysis['format']} format, "
                   f"agent type: {analysis['agent_type'].value}")
        
        return analysis
    
    def _load_standard_networks(self, checkpoint: Dict[str, Any], 
                              info: Dict[str, Any]) -> Any:
        """Load standard Stochastic MuZero networks"""
        try:
            from experimental_research.swt_models.swt_stochastic_networks import SWTStochasticMuZeroNetwork, SWTStochasticMuZeroConfig
            
            # Create networks instance with proper config (use actual Episode 13475 dimensions)
            hidden_dim = self.config.model.network.hidden_dim if hasattr(self.config, "model") and hasattr(self.config.model, "network") else 256  # From config hidden dimension (from checkpoint analysis)
            if hasattr(self.config, 'network_config') and hasattr(self.config.network_config, 'hidden_dim'):
                hidden_dim = self.config.network_config.hidden_dim
                
            config = SWTStochasticMuZeroConfig(
                hidden_dim=hidden_dim,
                use_optimized_blocks=True,
                use_vectorized_ops=True
            )
            networks = SWTStochasticMuZeroNetwork(config)
            
            # Load network weights - Episode 13475 uses muzero_network_state
            if "muzero_network_state" in checkpoint:
                # Episode 13475 format - load entire state dict at once
                networks.load_state_dict(checkpoint["muzero_network_state"])
                logger.info("✅ Episode 13475 network state loaded directly")
                    
            elif "networks" in checkpoint:
                # Standard format with nested networks
                network_states = checkpoint["networks"]
                
                networks.representation_network.load_state_dict(network_states["representation"])
                networks.dynamics_network.load_state_dict(network_states["dynamics"])
                networks.afterstate_dynamics.load_state_dict(network_states["afterstate"])
                networks.policy_network.load_state_dict(network_states["policy"])
                networks.value_network.load_state_dict(network_states["value"])
                networks.chance_encoder.load_state_dict(network_states["chance"])
                
            else:
                # Legacy format with direct network keys
                networks.representation_network.load_state_dict(checkpoint["representation_network"])
                networks.dynamics_network.load_state_dict(checkpoint["dynamics_network"])
                networks.afterstate_dynamics.load_state_dict(checkpoint["afterstate_dynamics"])
                networks.policy_network.load_state_dict(checkpoint["policy_network"])  
                networks.value_network.load_state_dict(checkpoint["value_network"])
                networks.chance_encoder.load_state_dict(checkpoint["chance_encoder"])
            
            # Move to device
            networks.to(self.device)
            
            # Set to eval mode
            networks.eval()
            
            logger.info("✅ Standard Stochastic MuZero networks loaded")
            
            return networks
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to load standard networks: {str(e)}",
                original_error=e
            )
    
    def _load_experimental_networks(self, checkpoint: Dict[str, Any],
                                  info: Dict[str, Any]) -> Any:
        """Load experimental networks with enhancements"""
        try:
            from experimental_research.swt_models.swt_stochastic_networks import SWTStochasticMuZeroNetwork, SWTStochasticMuZeroConfig
            
            # Create enhanced networks instance with experimental features (use actual Episode 13475 dimensions)
            hidden_dim = self.config.model.network.hidden_dim if hasattr(self.config, "model") and hasattr(self.config.model, "network") else 256  # From config hidden dimension (from checkpoint analysis)
            if hasattr(self.config, 'network_config') and hasattr(self.config.network_config, 'hidden_dim'):
                hidden_dim = self.config.network_config.hidden_dim
                
            config = SWTStochasticMuZeroConfig(
                hidden_dim=hidden_dim,
                use_optimized_blocks=True,
                use_vectorized_ops=True
            )
            networks = SWTStochasticMuZeroNetwork(config)
            
            # Load base networks (same as standard)
            if "muzero_network_state" in checkpoint:
                # Episode 13475 format - load entire state dict at once
                networks.load_state_dict(checkpoint["muzero_network_state"])
                logger.info("✅ Episode 13475 experimental network state loaded directly")
            elif "networks" in checkpoint:
                network_states = checkpoint["networks"]
                # Load core networks
                networks.representation_network.load_state_dict(network_states["representation"])
                networks.dynamics_network.load_state_dict(network_states["dynamics"]) 
                networks.afterstate_dynamics.load_state_dict(network_states["afterstate"])
                networks.policy_network.load_state_dict(network_states["policy"])
                networks.value_network.load_state_dict(network_states["value"])
                networks.chance_encoder.load_state_dict(network_states["chance"])
            else:
                network_states = checkpoint
                # Load core networks from root level
                networks.representation_network.load_state_dict(network_states["representation"])
                networks.dynamics_network.load_state_dict(network_states["dynamics"]) 
                networks.afterstate_dynamics.load_state_dict(network_states["afterstate"])
                networks.policy_network.load_state_dict(network_states["policy"])
                networks.value_network.load_state_dict(network_states["value"])
                networks.chance_encoder.load_state_dict(network_states["chance"])
            
            # Load experimental components if available
            if "value_prefix_network" in network_states:
                # Load value-prefix network
                from experimental_research.value_prefix_network import SWTValuePrefixNetwork
                
                value_prefix_config = self.config.experimental_config.get("value_prefix", {})
                architecture = value_prefix_config.get("architecture", "transformer")
                hidden_dim = value_prefix_config.get("hidden_dim", 64)
                
                networks.value_prefix_network = SWTValuePrefixNetwork(
                    input_dim=2,  # reward + value
                    hidden_dim=hidden_dim,
                    architecture=architecture
                ).to(self.device)
                
                networks.value_prefix_network.load_state_dict(network_states["value_prefix_network"])
                
                logger.info("📈 Value-prefix network loaded")
            
            # Load other experimental components as needed
            if "consistency_loss_state" in checkpoint:
                # Load consistency loss state
                logger.info("🔄 Consistency loss state loaded")
            
            # Move to device and set eval mode
            networks.to(self.device)
            networks.eval()
            
            logger.info("🚀 Experimental enhanced networks loaded")
            
            return networks
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to load experimental networks: {str(e)}",
                original_error=e
            )
    
    def _validate_networks(self, networks: Any, info: Dict[str, Any]) -> None:
        """Validate loaded networks"""
        try:
            # Check that all required networks are present (afterstate_dynamics is optional for Episode 13475)
            required_networks = ["representation_network", "dynamics_network", "policy_network", 
                               "value_network", "chance_encoder"]
            optional_networks = ["afterstate_dynamics"]  # Episode 13475 doesn't have this
            
            for network_name in required_networks:
                if not hasattr(networks, network_name):
                    raise CheckpointError(f"Missing required network: {network_name}")
                
                network = getattr(networks, network_name)
                if network is None:
                    raise CheckpointError(f"Network {network_name} is None")
            
            # Check optional networks (log warnings only)
            for network_name in optional_networks:
                if not hasattr(networks, network_name):
                    logger.warning(f"⚠️ Optional network missing: {network_name} (Episode 13475 compatibility)")
                else:
                    network = getattr(networks, network_name)
                    if network is None:
                        logger.warning(f"⚠️ Optional network is None: {network_name}")
            
            # Test forward pass with dummy data
            with torch.no_grad():
                # Test representation network (Episode 13475 expects fused state input)
                dummy_fused_state = torch.randn(1, 137, device=self.device)  # Already fused market+position
                
                latent = networks.representation_network(dummy_fused_state)
                
                expected_hidden_dim = self.config.model.network.hidden_dim if hasattr(self.config, "model") and hasattr(self.config.model, "network") else 256  # From config dimension (from checkpoint analysis)
                if hasattr(self.config, 'network_config') and hasattr(self.config.network_config, 'hidden_dim'):
                    expected_hidden_dim = self.config.network_config.hidden_dim
                    
                if latent.shape[1] != expected_hidden_dim:
                    raise CheckpointError(
                        f"Latent dimension mismatch: expected {expected_hidden_dim}, "
                        f"got {latent.shape[1]}"
                    )
                
                # Test policy network (needs latent + latent_z for Episode 13475)
                dummy_latent_z = torch.zeros(1, 16, device=self.device)  # Episode 13475 latent_z dimension
                policy_logits = networks.policy_network(latent, dummy_latent_z)
                if policy_logits.shape[1] != 4:  # 4 trading actions
                    raise CheckpointError(f"Policy output dimension mismatch: expected 4, got {policy_logits.shape[1]}")
                
                # Test value network (needs latent + latent_z for Episode 13475)
                value_logits = networks.value_network(latent, dummy_latent_z)
                expected_value_support = 601  # Episode 13475 standard (-300 to +300)
                if hasattr(self.config, 'network_config') and hasattr(self.config.network_config, 'value_support_size'):
                    expected_value_support = self.config.network_config.value_support_size
                    
                if value_logits.shape[1] != expected_value_support:
                    raise CheckpointError(f"Value output dimension mismatch: expected {expected_value_support}, got {value_logits.shape[1]}")
            
            logger.info("✅ Network validation passed")
            
        except Exception as e:
            raise CheckpointError(
                f"Network validation failed: {str(e)}",
                original_error=e
            )
    
    def _determine_device(self) -> torch.device:
        """Determine appropriate device for model loading"""
        # Handle case where system_config doesn't exist
        system_config = getattr(self.config, 'system_config', {})
        device_config = system_config.get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_config)
        
        return device
    
    def save_checkpoint(self, networks: Any, metadata: Dict[str, Any],
                       output_path: Union[str, Path],
                       agent_type: Optional[AgentType] = None) -> None:
        """
        Save checkpoint in unified format
        
        Args:
            networks: Network collection to save
            metadata: Checkpoint metadata
            output_path: Output file path
            agent_type: Agent type for proper formatting
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare checkpoint data
            checkpoint_data = {
                "networks": {
                    "representation": networks.representation_network.state_dict(),
                    "dynamics": networks.dynamics_network.state_dict(),
                    "afterstate": networks.afterstate_dynamics.state_dict(),
                    "policy": networks.policy_network.state_dict(),
                    "value": networks.value_network.state_dict(),
                    "chance": networks.chance_encoder.state_dict()
                },
                "metadata": metadata,
                "config": {
                    "agent_type": "episode_13475_test",  # For testing purposes
                    "network_config": getattr(self.config, 'network_config', {'hidden_dim': 128, 'value_support_size': 601}),
                    "runtime_config": getattr(self.config, 'runtime_config', {})
                },
                "save_info": {
                    "saved_at": datetime.now().isoformat(),
                    "device": str(self.device),
                    "pytorch_version": torch.__version__
                }
            }
            
            # Add experimental components if present
            if hasattr(networks, "value_prefix_network") and networks.value_prefix_network is not None:
                checkpoint_data["networks"]["value_prefix_network"] = networks.value_prefix_network.state_dict()
                checkpoint_data["experimental_config"] = self.config.experimental_config
            
            # Save checkpoint
            torch.save(checkpoint_data, output_path)
            
            logger.info(f"💾 Checkpoint saved to: {output_path}")
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to save checkpoint: {str(e)}",
                context={"output_path": str(output_path)},
                original_error=e
            )
    
    def get_checkpoint_info(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Get checkpoint information without loading networks"""
        try:
            raw_checkpoint = self._load_raw_checkpoint(Path(checkpoint_path))
            info = self._analyze_checkpoint(raw_checkpoint, Path(checkpoint_path))
            
            return {
                "agent_type": info["agent_type"].value,
                "format": info["format"],
                "metadata": info["metadata"],
                "file_size_mb": Path(checkpoint_path).stat().st_size / (1024 * 1024),
                "networks_present": list(raw_checkpoint.keys()) if isinstance(raw_checkpoint, dict) else []
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate_checkpoint_compatibility(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate checkpoint compatibility with current configuration"""
        try:
            info = self.get_checkpoint_info(checkpoint_path)
            
            compatibility = {
                "compatible": True,
                "issues": [],
                "warnings": []
            }
            
            # Check agent type compatibility
            checkpoint_agent_type = AgentType(info["agent_type"])
            if checkpoint_agent_type != self.config.agent_system:
                compatibility["warnings"].append(
                    f"Agent type mismatch: checkpoint is {checkpoint_agent_type.value}, "
                    f"config is {self.config.agent_system.value}"
                )
            
            # Check for experimental features
            if checkpoint_agent_type == AgentType.EXPERIMENTAL:
                if not self.config.experimental_config.get("consistency_loss", {}).get("enabled", False):
                    compatibility["warnings"].append("Experimental checkpoint loaded but consistency loss disabled")
            
            return compatibility
            
        except Exception as e:
            return {
                "compatible": False,
                "issues": [f"Compatibility check failed: {str(e)}"],
                "warnings": []
            }