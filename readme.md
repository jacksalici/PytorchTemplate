# Pytorch Template 🔧

![Open Issues](https://img.shields.io/github/issues/jacksalici/PytorchTemplate) ![GitHub Repo stars](https://img.shields.io/github/stars/jacksalici/PytorchTemplate?style=flat) ![Static Badge](https://img.shields.io/badge/made_with-pizza_and_coffee-lightgray) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white&style=flat) [![Licence](https://img.shields.io/github/license/jacksalici/PyTorchTemplate?style=for-the-badge&style=flat)](./LICENSE)
 
> A custom template for scalable PyTorch projects with flexible configuration management. Designed to be ready in minimal time while remaining maintainable and scalable. ♻️ 

⚠️ This template is still in _beta_ and may undergo significant changes. Feedback and contributions are welcome!

## Template Structure 📂
```bash
PytorchTemplate/
├── readme.md  👈 You are here
├── requirements.txt            
├── run.py 
├── configs/ 
│   ├── config.py             
│   ├── default.yaml
│   └── ...
├── dataloaders/ 
│   └── ...                      
├── experiments/ 
│   ├── exp_base.py              
│   └── ...
├── models/                      
│   ├── base.py                  
│   └── ...
└── utils/                       
	├── logger.py
	├── binary_metrics.py   
	├── reproducibility.py
	└── ...                 
```

- **`run.py`**: Main entry point that handles configuration loading and experiment execution.
- **`configs/`**: The _certainly_not_over_engineered_ configuration system (read below for more details).
- **`experiments/`**: Modular experiment classes that define training/inference logic. For larger projects, you may want a file for each experiment, such as forecasting, classification, etc.
- **`models/`**: PyTorch model implementations with a common base class that inherits all methods from `torch.nn.Module`.
- **`dataloaders/`**: Data loading and preprocessing modules.
- **`utils/`**: Shared utilities for logging, metrics, plotting, and reproducibility. **Wandb** comes pre-integrated in the logger.

## Installation 🧨

1. Clone the repository or create it from this template:
	```bash
	git clone <your-repo-url>
	cd <your-repo-name>
	```
2. Install dependencies:
	```bash
	python3 -m venv .venv
	pip3 install -r requirements.txt
	```

🚯 To remove all files regarding the toy problem below and start with a fresly minted template run the following command!
```bash
bash cleanup.sh
```


## Training and Inference 🚀

> The template comes with a simple experiment inspired by the [MinGPT Demo](https://github.com/karpathy/minGPT/blob/master/demo.ipynb). From a sequence of 6 numbers, the transformer model should learn to sort them in ascending order. 
> Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2

To train the model, run:

```bash
python run.py --config default.yaml toySorter.yaml --task training
```

The models will be trained in a minute or two, and the accuracy should reach 100%.
To infer from the model, run:

```bash
python run.py --config default.yaml toySorter.yaml --task inference
```

## Configuration 🔧

Configuration is handled by the `configs/config.py` file, which provides:
- Default values for all parameters.
- YAML configuration file loading.
- Command-line argument overrides.

This means that the configuration priority is (from highest to lowest): (1) command-line arguments, (2) YAML configuration file values, and (3) default values in the `Config` class.

## Notes 📝
The template is developed as my personal starting point for new PyTorch projects. There is a trade-off when developing this type of template: the goal is to write the maximum amount of [reusable code to save time](https://imgs.xkcd.com/comics/code_lifespan.png) in the future while avoiding to add complexity and knowledge overhead. I _hope_ this strikes the right balance. It should be easy to understand and extend, while also providing a solid foundation for scalable projects.

The template will be updated as I discover new features to add or encounter bugs. If you have any suggestions or issues, feel free to open an issue on the repository.

## License 📜
[MIT License](./LICENSE)

 