"""Training package for the Go code reviewer project.

Why this file exists:
- Marks `training/` as an explicit Python package.
- Provides a clean import surface for commonly used training functions.

Example:
	from training import load_training_config, run_training

	cfg = load_training_config("training/training_config.yaml")
	run_training(cfg)
"""


def load_training_config(config_path: str | None = None):
	"""Load training configuration from YAML/defaults."""
	from .fine_tune_go_reviewer import load_config

	return load_config(config_path)


def run_training(config: dict):
	"""Run fine-tuning with the provided config dictionary."""
	from .fine_tune_go_reviewer import train

	return train(config)


__all__ = [
	"load_training_config",
	"run_training",
]

