#!/usr/bin/env python3
"""
Test Episode 13475 checkpoint structure and configuration
"""

import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any

# Add to path
sys.path.append(str(Path(__file__).parent))

def analyze_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Analyze checkpoint structure without requiring data
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint analysis
    """
    print(f"\n🔍 Analyzing checkpoint: {checkpoint_path}")
    print("="*80)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    analysis = {
        'checkpoint_path': checkpoint_path,
        'file_size_mb': Path(checkpoint_path).stat().st_size / (1024 * 1024),
        'keys': list(checkpoint.keys()),
        'structure': {}
    }
    
    # Analyze structure
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            analysis['structure'][key] = {
                'type': 'tensor',
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'device': str(value.device)
            }
        elif isinstance(value, dict):
            analysis['structure'][key] = {
                'type': 'dict',
                'keys': list(value.keys()) if len(value) < 20 else f"{len(value)} keys",
                'sample': {k: type(v).__name__ for k, v in list(value.items())[:3]}
            }
        elif isinstance(value, (list, tuple)):
            analysis['structure'][key] = {
                'type': type(value).__name__,
                'length': len(value),
                'sample_types': [type(v).__name__ for v in value[:3]]
            }
        else:
            analysis['structure'][key] = {
                'type': type(value).__name__,
                'value': str(value) if not isinstance(value, (int, float, str, bool)) else value
            }
    
    # Check for episode info
    if 'episode' in checkpoint:
        analysis['episode_info'] = checkpoint['episode']
    
    # Check for configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            analysis['config_summary'] = {
                'keys': list(config.keys()),
                'feature_config': config.get('feature_processor', {}),
                'training_config': config.get('training', {}),
                'model_config': config.get('model', {})
            }
    
    # Check for model state
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        analysis['model_info'] = {
            'num_parameters': len(model_state),
            'parameter_groups': {},
            'total_params': 0
        }
        
        # Group parameters by module
        for param_name, param_tensor in model_state.items():
            module = param_name.split('.')[0]
            if module not in analysis['model_info']['parameter_groups']:
                analysis['model_info']['parameter_groups'][module] = {
                    'count': 0,
                    'total_size': 0
                }
            analysis['model_info']['parameter_groups'][module]['count'] += 1
            analysis['model_info']['parameter_groups'][module]['total_size'] += param_tensor.numel()
            analysis['model_info']['total_params'] += param_tensor.numel()
    
    # Check for training stats
    if 'training_stats' in checkpoint:
        stats = checkpoint['training_stats']
        analysis['training_summary'] = {
            'type': type(stats).__name__,
            'keys': list(stats.keys()) if isinstance(stats, dict) else 'not a dict'
        }
    
    return analysis, checkpoint

def print_analysis(analysis: Dict[str, Any]):
    """Pretty print checkpoint analysis"""
    
    print(f"\n📊 CHECKPOINT ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📁 File Info:")
    print(f"  - Path: {analysis['checkpoint_path']}")
    print(f"  - Size: {analysis['file_size_mb']:.2f} MB")
    
    print(f"\n🔑 Top-level Keys:")
    for key in analysis['keys']:
        print(f"  - {key}")
    
    if 'episode_info' in analysis:
        print(f"\n🎯 Episode Info: {analysis['episode_info']}")
    
    if 'config_summary' in analysis:
        print(f"\n⚙️ Configuration Summary:")
        config = analysis['config_summary']
        if 'feature_config' in config and config['feature_config']:
            print(f"  Feature Processor:")
            for k, v in config['feature_config'].items():
                print(f"    - {k}: {v}")
    
    if 'model_info' in analysis:
        print(f"\n🧠 Model Information:")
        print(f"  Total Parameters: {analysis['model_info']['total_params']:,}")
        print(f"  Parameter Groups:")
        for module, info in analysis['model_info']['parameter_groups'].items():
            print(f"    - {module}: {info['count']} tensors, {info['total_size']:,} params")
    
    if 'training_summary' in analysis:
        print(f"\n📈 Training Stats: {analysis['training_summary']}")
    
    print("\n" + "="*80)

def main():
    checkpoint_path = "checkpoints/episode_13475.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    try:
        analysis, checkpoint = analyze_checkpoint(checkpoint_path)
        print_analysis(analysis)
        
        # Save analysis to JSON
        output_path = "checkpoints/episode_13475_analysis.json"
        with open(output_path, 'w') as f:
            # Convert non-serializable objects
            serializable_analysis = json.loads(
                json.dumps(analysis, default=str)
            )
            json.dump(serializable_analysis, f, indent=2)
        print(f"\n✅ Analysis saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error analyzing checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())