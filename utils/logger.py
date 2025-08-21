import json
import logging
from typing import Literal
class Logger:
    _logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    
    def __init__(self, project_name, log_level= Literal["debug", "info", "warning", "error"], avoid_wandb = True, remote_logger_run_name = None, separator = "|"):
        """
        Initializes the Logger instance.
        
        Args:
            project_name (str): Name of the project for logging.
            log_level (Literal["debug", "info", "warning", "error"], optional): Logging level. Defaults to "info".
            avoid_wandb (bool, optional): If True, avoids using Weights & Biases for logging. Defaults to True.
            remote_logger_run_name (str, optional): Name for the remote logger run. Defaults to None.
            separator (str, optional): Separator used in log messages. Defaults to "|".
        """
        
        
        self.avoid_wandb = avoid_wandb
        
        # Setup logging
        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(self._logging_levels.get(log_level, logging.INFO))
        self.separator = separator
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s [%(levelname)s] {separator} %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if not self.avoid_wandb:
            import wandb
            wandb.init(project=project_name, name=remote_logger_run_name)
    
    def print_config(self, config_dict):
        """Prints the configuration dictionary to the console and logs it where needed.
        Args:
            config_dict (dict): Configuration dictionary to print.
        """
        
        if not self.avoid_wandb:
            wandb.config.update(config_dict)
        self.logger.info(f"Configuration: {json.dumps(config_dict, indent=4)}")

    def log(self, info: dict, level: Literal["debug", "info", "warning", "error"] = "info"):
        """Send information to the logger. This can be used to log metrics, parameters, or any other information.
        It will log to the console and, if wandb is not avoided, to wandb as well.

        Args:
            info (dict): Information to log, typically a dictionary of metrics or parameters.
            level (Literal["debug", "info", "warning", "error"], optional): The logging level of the message. Defaults to "info".
            
        """
        
        if not self.avoid_wandb:
            wandb.log(info)
        
        log_message = f" {self.separator} ".join([f"{key}: {value}" for key, value in info.items()])
        
        logging_level = self._logging_levels.get(level, logging.INFO)   
            
        self.logger.log(logging_level, log_message)