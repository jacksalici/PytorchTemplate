import yaml
import argparse
from dataclasses import dataclass, field, fields, MISSING
from typing import Any, Dict, Optional, Self, List, Literal
from pathlib import Path
import torch
import os

def str2bool(v):
    """Convert string representation of truth to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got: {v}')


def get_default_run_name() -> str:
    """Get the name of the parent folder of the current file."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(__file__).parent.parent.name + f"_{now}"

@dataclass
class Config:
    """Configuration class for the PyTorch template."""
    
    # General settings
    seed: int = 42
    force_reproducibility: bool = True
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    
    # Training settings
    lr: float = 0.001
    batch_size: int = 32
    num_epochs: int = 20
    
    
    # Model settings
    model_name: str = "mlp"
    input_dim: int = 10
    embed_dim: int = 64
    seq_len: int = 2400
    
    # MLP specific settings
    num_layers: int = 3
    hidden_dims: Optional[List[int]] = None
    dropout_rate: float = 0.1
    activation: str = "relu"
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    output_dim: int = 2
    
    # Transformer specific settings
    num_heads: int = 8
    attention_dropout: float = 0.1
    transformer_num_layers: int = 6
    
    # Experiment and data settings
    experiment: str = ""
    dataloader: str = ""

    # Optimizer settings
    optim: str = "adam"
    
    # Logging and saving
    avoid_wandb: bool = False
    save_model: bool = True
    checkpoint_path: str = "checkpoints"
    checkpoint_name: Optional[str] = None
    
    # Config file settings
    config: List[str] = field(default_factory=lambda: ["default.yaml"])
    config_path: str = "configs"
    run_name: str = get_default_run_name()
    task: Literal["training", "inference"] = "training"

    @classmethod
    def _from_yamls(cls, yaml_paths: List[str]) -> Dict[str, Any]:
        """Load configuration from a list of YAML files and return as dict (later files override earlier ones)."""
            
        merged_config = {}
        
        for config_file in yaml_paths:            
            try:
                with open(config_file, 'r') as f:
                    yaml_data = yaml.safe_load(f) or {}
                    merged_config.update(yaml_data)
            except FileNotFoundError:
                print(f"Warning: Config file not found: {config_file}")
        
        return merged_config
    
    @classmethod
    def from_args(cls, override_args: Optional[Dict[str, Any]] = None) -> Self:
        """Create configuration with CLI argument overrides and support for multiple config files."""
        # Default argument parser
        parser = argparse.ArgumentParser(description="PyTorch Template")
        parser.add_argument("--config", type=str, nargs='+', default=["default.yaml"], help="Paths to one or more config YAML files")
        parser.add_argument("--config_path", type=str, default="configs", help="Directory containing config files")
        parser.add_argument("--run_name", type=str, default=get_default_run_name(), help="Experiment name")
        
        args, remaining_args = parser.parse_known_args()
        
        merged_config = {}
        
        # 1. Parse known args first to get config files and then load them
        if args.config:
            merged_config = cls._from_yamls([Path(args.config_path) / config_file for config_file in args.config])
        
        merged_config['config'] = args.config
        merged_config['config_path'] = args.config_path  
        merged_config['run_name'] = args.run_name
        
        # 2.  Now add CLI arguments for all known fields plus any extra fields from YAML
        all_field_names = {f.name: f.type for f in fields(cls)} | { key: type(v) for key, v in merged_config.items() }
        
        for field_name, field_type in all_field_names.items():
            if field_name in ['config', 'config_path', 'run_name']:
                continue  
            
            if field_type:
                # Use existing field type logic
                if field_type == bool:
                    parser.add_argument(f"--{field_name}", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                                      default=argparse.SUPPRESS, help=f"Override {field_name}")
                elif field_type == int:
                    parser.add_argument(f"--{field_name}", type=int, default=argparse.SUPPRESS, help=f"Override {field_name}")
                elif field_type == float:
                    parser.add_argument(f"--{field_name}", type=float, default=argparse.SUPPRESS, help=f"Override {field_name}")
                elif field_type == str or field_type == Optional[str]:
                    parser.add_argument(f"--{field_name}", type=str, default=argparse.SUPPRESS, help=f"Override {field_name}")
                elif field_type == Optional[List[int]] or field_type == List[int]:
                    parser.add_argument(f"--{field_name}", nargs='+', type=int, default=argparse.SUPPRESS, help=f"Override {field_name}")
                elif field_type == Optional[List[float]] or field_type == List[float]:
                    parser.add_argument(f"--{field_name}", nargs='+', type=float, default=argparse.SUPPRESS, help=f"Override {field_name}")
                elif field_type == Optional[list] or field_type == list:
                    parser.add_argument(f"--{field_name}", nargs='+', default=argparse.SUPPRESS, help=f"Override {field_name}")
                elif hasattr(field_type, '__origin__') and field_type.__origin__ is Literal or \
                          hasattr(field_type, '__args__') and len(field_type.__args__) > 0 and hasattr(field_type.__args__[0], '__origin__') and field_type.__args__[0].__origin__ is Literal:
                    # Handle Literal types and Optional[Literal]
                    literal_type = field_type if field_type.__origin__ is Literal else field_type.__args__[0]
                    parser.add_argument(f"--{field_name}", type=str, default=argparse.SUPPRESS, choices=list(literal_type.__args__), help=f"Override {field_name}")
            else:
                # If no type is specified, treat it as a string
                parser.add_argument(f"--{field_name}", type=str, default=argparse.SUPPRESS, help=f"Override {field_name}")
        
        args = parser.parse_args()
        
        # Override default values with merged YAML config
        known_fields = {f.name: f.default if f.default is not MISSING else None for f in fields(cls)}
        
        config_dict = known_fields.copy()
        config_dict.update(merged_config)
        
        cli_overrides = {k: v for k, v in vars(args).items() 
                        if k not in ['config', 'config_path'] and hasattr(args, k)}
        config_dict.update(cli_overrides)
        
        if override_args:
            # Apply additional override args if provided
            config_dict.update(override_args)
        
        # Create instance with known fields only and add extra fields as attributes
        known_config_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        instance = cls(**known_config_dict)
        
        for key, value in config_dict.items():
            if key not in known_fields:
                setattr(instance, key, value)
        
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {f.name: getattr(self, f.name) for f in fields(self)}
        
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
   
    def get_device(self) -> str:
        """Return the device to use for training."""
        if self.device in ["auto", "cuda"] and torch.cuda.is_available():
                return "cuda"
        
        if self.device in ["auto", "mps"] and torch.backends.mps.is_available(): 
             if torch.backends.mps.is_available():
                return "mps"
        
        return "cpu"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default. Actually a fallback when using the config as a dict."""
        return getattr(self, key, default)
    
    
    def get_checkpoint_path(self) -> str:
        """Get the path to the config directory."""
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
                
        if self.checkpoint_name is not None:
            path = os.path.join(self.checkpoint_path, self.checkpoint_name)
        else:
            path = f"{str(os.path.join(self.checkpoint_path, self.model_name))}.pth"
            
        return path
    
