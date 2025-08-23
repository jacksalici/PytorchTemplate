"""
Unified logging utility combining console output with optional remote tracking

This module provides a flexible logging interface that combines Python's
standard logging with optional Weights & Biases integration.
It enables logging text messages at different severity levels and structured
data like metrics and configurations in both local and remote contexts.
"""

import json
import logging
from typing import Any, Dict, Literal


class Logger:
    """
    Provides console logging and optional Weights & Biases integration.
    """

    _logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    _wandb = None

    def __init__(
        self,
        project_name: str,
        log_level=Literal["debug", "info", "warning", "error"],
        avoid_wandb: bool = True,
        remote_logger_run_name: str | None = None,
        separator: str = "|",
        multi_line: bool = True,
    ):
        """
        Initializes the Logger instance.

        Args:
            project_name: Name of the project for logging.
            log_level: Logging level. Defaults to "info".
            avoid_wandb: If True, avoids using Weights & Biases for logging. Defaults to True.
            remote_logger_run_name: Name for the remote logger run. Defaults to None.
            separator: Separator used in log dict messages. Defaults to "|".
            multi_line: If True, formats log dict messages in multiple lines. Defaults to True.
        """

        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(self._logging_levels.get(log_level, logging.INFO))
        self.separator = separator
        self.multi_line = multi_line

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f"%(asctime)s [%(levelname)s] {separator} %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if not avoid_wandb:
            self._init_wandb(project_name, remote_logger_run_name)

    def print_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Prints the configuration dict to the console and logs it where needed.

        Args:
            config_dict (dict): Configuration dictionary to print.
        """

        if self._wandb:
            self._wandb.config.update(config_dict)

        self.logger.info(f"Configuration: {json.dumps(config_dict, indent=4)}")

    def __call__(
        self,
        info: dict | str,
        level: Literal["debug", "info", "warning", "error"] = "info",
    ) -> None:
        """
        Send information to the logger.

        This can be used to log metrics, parameters, or any other information.
        It will log to the console and to wandb as well (if enabled).

        Args:
            info: Information to log, typically a dictionary of metrics or
                parameters or a string message.
            level The logging level of the message. Defaults to "info".
                - Available levels: "debug", "info", "warning", "error".
        """
        assert isinstance(info, (dict, str)), "Info must be a dictionary or a string."

        # Case 1: info is a dictionary
        if isinstance(info, dict):
            if self._wandb:
                self._wandb.log(info)

            log_message = (
                f" {self.separator} ".join(
                    [f"{key}: {value}" for key, value in info.items()]
                )
                if not self.multi_line
                else json.dumps(info, indent=4)
            )

        # Case 2: info is a string
        else:
            log_message = info

        logging_level = self._logging_levels.get(level, logging.INFO)

        self.logger.log(logging_level, log_message)

    def _init_wandb(self, project_name: str, remote_logger_run_name: str | None = None):
        try:
            import wandb

            wandb.init(project=project_name, name=remote_logger_run_name)
            self._wandb = wandb
        except ImportError:
            self.logger.warning(
                "Weights & Biases (wandb) is not installed. "
                "Please install it to use wandb logging."
            )
            self._wandb = None
