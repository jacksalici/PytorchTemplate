import yaml
import argparse
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Self, List, Literal
from pathlib import Path
import torch

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
    ff_dim: int = 2048
    max_seq_len: int = 5000
    attention_dropout: float = 0.1
    transformer_num_layers: int = 6
    
    # Experiment and data settings
    experiment: str = ""
    dataloader: str = ""
    
    
    # Toy sorting specific settings
    sequence_length: int = 6
    num_digits: int = 1
    

    # Optimizer settings
    optim: str = "adam"
    
    # Logging and saving
    avoid_wandb: bool = False
    save_model: bool = True
    checkpoint_path: str = "checkpoints"
    checkpoint_name: Optional[str] = None
    
    # Config file settings
    config: str = "default.yaml"
    config_path: str = "configs"
    run_name: str = get_default_run_name()
    task: Literal["training", "inference"] = "training"

    @classmethod
    def _from_yaml(cls, yaml_path: str) -> Self:
        """Load configuration from a YAML file."""
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}
        
        # Filter only valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in yaml_data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    @classmethod
    def from_args(cls, override_args: Optional[Dict[str, Any]] = None) -> Self:
        """Create configuration with CLI argument overrides."""
        # Default argument parser
        parser = argparse.ArgumentParser(description="PyTorch Template")
        parser.add_argument("--config", type=str, default="default.yaml", help="Path to config YAML file")
        parser.add_argument("--config_path", type=str, default="configs", help="Directory containing config files")
        parser.add_argument("--run_name", type=str, default=get_default_run_name(), help="Experiment name")
        
        for f in fields(cls):
            if f.name in ['config', 'config_path', 'run_name']:
                continue  # Already added above
                
            if f.type == bool:
                parser.add_argument(f"--{f.name}", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                                  default=argparse.SUPPRESS, help=f"Override {f.name}")
            elif f.type == int:
                parser.add_argument(f"--{f.name}", type=int, default=argparse.SUPPRESS, help=f"Override {f.name}")
            elif f.type == float:
                parser.add_argument(f"--{f.name}", type=float, default=argparse.SUPPRESS, help=f"Override {f.name}")
            elif f.type == str or f.type == Optional[str]:
                parser.add_argument(f"--{f.name}", type=str, default=argparse.SUPPRESS, help=f"Override {f.name}")
            elif f.type == Optional[List[int]] or f.type == List[int]:
                parser.add_argument(f"--{f.name}", nargs='+', type=int, default=argparse.SUPPRESS, help=f"Override {f.name}")
            elif f.type == Optional[List[float]] or f.type == List[float]:
                parser.add_argument(f"--{f.name}", nargs='+', type=float, default=argparse.SUPPRESS, help=f"Override {f.name}")
            elif f.type == Optional[list] or f.type == list:
                parser.add_argument(f"--{f.name}", nargs='+', default=argparse.SUPPRESS, help=f"Override {f.name}")
            elif hasattr(f.type, '__origin__') and f.type.__origin__ is Literal or \
                      hasattr(f.type, '__args__') and len(f.type.__args__) > 0 and hasattr(f.type.__args__[0], '__origin__') and f.type.__args__[0].__origin__ is Literal:
                # Handle Literal types and Optional[Literal]
                literal_type = f.type if f.type.__origin__ is Literal else f.type.__args__[0]
                parser.add_argument(f"--{f.name}", type=str, default=argparse.SUPPRESS, choices=list(literal_type.__args__), help=f"Override {f.name}")
                 
        args = parser.parse_args()
        
        # Start with default config
        config = cls()
        
        # Load from YAML if specified
        if args.config:
            config_file = Path(args.config_path) / args.config
            config = cls._from_yaml(str(config_file))
        
        # Override with CLI arguments
        cli_overrides = {k: v for k, v in vars(args).items() 
                        if k not in ['config', 'config_path'] and hasattr(args, k)}
        
        for key, value in cli_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        if override_args:
            for key, value in override_args.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
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
        
    
