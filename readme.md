# Pytorch Template
> A custom template for scalable PyTorch projects with flexible configuration management.

## Installation

1. Clone the repository or make it from this template
	```bash
	git clone <your-repo-url>
	cd <your-repo-name>
	```
2. Install dependencies:
	```bash
    python -m venv .venv
	pip install -r requirements.txt
	```

## Usage

### Training with Configuration Files

To train a model using a configuration file:

```bash
python run.py --config example.yaml
```

### Command Line Arguments

You can override any configuration parameter via command line:

```bash
python run.py --config example.yaml --lr 0.01 --batch_size 64 --num_epochs 100
```

### Running without Config File

You can run with default settings and override specific parameters:

```bash
python run.py --model_name mlp --lr 0.001 --num_epochs 50
```

## Configuration

Configuration is handled by the `configs/conf.py` file which provides:
- Default values for all parameters
- YAML configuration file loading
- Command line argument overrides
- Type validation

### Configuration Priority (highest to lowest)
1. Command line arguments
2. YAML configuration file values  
3. Default values in `Config` class


## Example Usage:

```bash
# Use MLP config
python run.py --config example.yaml

# Override parameters
python run.py --config example.yaml --hidden_dims 256 128 64 --use_batch_norm false

# Use transformer
python run.py --config transformer.yaml --num_heads 12 --transformer_num_layers 8
```

> The Toy Sorter is a simple example inspired by the [MinGPT Demo](https://github.com/karpathy/minGPT/blob/master/demo.ipynb). From a sequence of 6 number the model should learn to sort them in ascending order.

To evaluate a model, run:

```bash
python inf.py --config configs/toySorter.yaml
``` 

## Configuration
- Place your config files in the `configs/` directory.
- Edit `requirements.txt` to add/remove dependencies as needed.


