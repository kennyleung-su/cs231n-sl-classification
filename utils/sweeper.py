"""Module for sweeping a set of hyperparameters for a given experiment, and analyzing the results."""
import copy
from configs.config import ConfigObjFromDict
from enum import Enum
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from utils.metrics import AccuracySaver, LossSaver, PredictionSaver

class ValueType(Enum):
		DISCRETE = 1
		CONTINUOUS = 2
		FROM_LIST = 3  # TODO: Make FROM_LIST samples exhastive for the list, if possible.


class HyperparameterOption(object):
	def __init__(self, value_type, range, round_to=10):
		self._value_type = value_type
		self._range = range   # either of the form [min, max) or [x1, x2, x3]
		self._round_to = round_to
		if self._value_type != ValueType.FROM_LIST:
			assert len(range) == 2, "Range must be of length 2 for the given value type range."

	def sample(self):
		if self._value_type == ValueType.DISCRETE:
			# Sample a discrete value
			return np.random.randint(*self._range)
		elif self._value_type == ValueType.CONTINUOUS:
			# Sample a continuous value
			return np.around(np.random.uniform(*self._range), self._round_to)
		elif self._value_type == ValueType.FROM_LIST:
			# Sample a random value from the list
			return np.random.choice(self._range).astype(list)  # For some reason this fixes dtype issues.


class ModelConfigMetadataWrapper(ConfigObjFromDict):
	"""Wrapper around a configuration object that tracks associated statistics metadata."""

	def __init__(self, sampled_config, metrics_dir, *args, **kwargs):
		super(ModelConfigMetadataWrapper, self).__init__(*args, **kwargs)
		# These public values can be set by the main program generating statistics from
		# the model config. They are of type MetricsCsvSaver.
		self.sampled_config = sampled_config

		# TODO: Auto-initialize these. Requires the directory to the config.
		h = self.__hash__()
		self.train_acc_saver = AccuracySaver(
			os.path.join(metrics_dir, '{0}.{1}'.format(h, 'train_acc')))
		self.valid_acc_saver = AccuracySaver(
			os.path.join(metrics_dir, '{0}.{1}'.format(h, 'valid_acc')))
		self.train_loss_saver = LossSaver(
			os.path.join(metrics_dir, '{0}.{1}'.format(h, 'train_loss')))
		self.preds_saver = PredictionSaver(
			os.path.join(metrics_dir, '{0}.{1}'.format(h, 'preds')),
				num_classes=len(kwargs['gesture_labels']))

	def analyze(self):
		# TODO: Add useful analysis code.
		metric_name_to_saver = {
			'training loss': self.train_loss_saver,
			'training accuracy': self.train_acc_saver,
			'validation accuracy': self.valid_acc_saver
		}
		logging.info('\n====== Experiment {0} ======\nSampled Config: {1}'.format(
				self.__hash__(), self.sampled_config))
		for (name, saver) in metric_name_to_saver.items():
			logging.info('avg {0}: {1}'.format(name, saver.meter.avg))
			logging.info('min {0}: {1}'.format(name, saver.meter.min))
			logging.info('max {0}: {1}'.format(name, saver.meter.max))


class HyperparameterSweeper(object):

	def __init__(self, config_options, model_config, metrics_dir, plots_dir):
		"""Constructs the sweeper.

		Args:
			config_options: dict where key values correspond to attributes of model_config,
				and values are HyperparameterOptions with value ranges and other options
			model_config: MODEL_CONFIG (ConfigObjFromDict) from configs/config.py
			metrics_dir: Path to store metrics log data
			metrics_dir: Path to store plots of the metrics/analysis
		"""
		self._config_options = config_options
		self._model_config = model_config
		self._metrics_dir = metrics_dir
		self._plots_dir = plots_dir
		# Stores a reference to the sampled configuration, in sample_hyperparameters.
		self._generated_configs = []

	def get_original_sweep(self):
		model_config = copy.copy(self._model_config.to_dict())
		model_config_md_wrapper = ModelConfigMetadataWrapper(
			sampled_config={}, metrics_dir=self._metrics_dir, **model_config)
		self._generated_configs.append(model_config_md_wrapper)
		return model_config_md_wrapper

	def get_random_sweeps(self, k):
		"""Randomly sample k model from the hyperparameters. Returns a generator."""
		for _ in range(k):
			model_config = copy.copy(self._model_config.to_dict())
			sampled_config = self._sample_hyperparameters()
			for k, v in sampled_config.items():
				model_config[k] = v

			model_config_md_wrapper = ModelConfigMetadataWrapper(sampled_config=sampled_config,
				metrics_dir=self._metrics_dir, **model_config)
			self._generated_configs.append(model_config_md_wrapper)
			yield model_config_md_wrapper

	def _sample_hyperparameters(self):
		"""Randomly samples hyperparameter dicts for all configuration fields defined in
		config_options."""
		config = {}
		for attr, option in self._config_options.items():
			config[attr] = option.sample()
		return config

	def analyze_performance(self):
		for config in self._generated_configs:
			config.analyze()

	def analyze_hyperparameter(self, hyperparameter):
		"""Plots data on the same graph across all hyperparameters."""

		# TODO: Clean up this logic. Sorry for the mess!
		plt.figure()
		plt.xlabel('Iteration')
		plt.ylabel('Loss')
		plt.title('Average Training Loss Across Iterations')
		for config in self._generated_configs:
			plt.plot(*config.train_loss_saver.get_plot_points(),
				label='{0} = {1}'.format(hyperparameter, config.sampled_config[hyperparameter]))
		train_loss_output_path = os.path.join(self._plots_dir, 'train_loss.{2}.{0}.{1}.png'.format(
			hyperparameter, time.time(), self._model_config.experiment))
		logging.info('Saving {0} hyperparameter comparison plot to: {1}'.format(
			hyperparameter, train_loss_output_path))
		plt.legend()
		plt.savefig(train_loss_output_path)

		plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.title('Validation Accuracy Across Epochs')
		for config in self._generated_configs:
			plt.plot(*config.valid_acc_saver.get_plot_points(),
				label='{0} = {1}'.format(hyperparameter, config.sampled_config[hyperparameter]))
		valid_acc_output_path = os.path.join(self._plots_dir, 'valid_acc.{2}.{0}.{1}.png'.format(
			hyperparameter, time.time(), self._model_config.experiment))
		logging.info('Saving {0} hyperparameter comparison plot to: {1}'.format(
			hyperparameter, valid_acc_output_path))
		plt.legend()
		plt.savefig(valid_acc_output_path)

		plt.figure()
		plt.xlabel('Iteration')
		plt.ylabel('Loss')
		plt.title('Training Accuracy Across Epochs')
		for config in self._generated_configs:
			plt.plot(*config.train_acc_saver.get_plot_points(),
				label='{0} = {1}'.format(hyperparameter, config.sampled_config[hyperparameter]))
		train_acc_output_path = os.path.join(self._plots_dir, 'train_acc.{2}.{0}.{1}.png'.format(
			hyperparameter, time.time(), self._model_config.experiment))
		logging.info('Saving {0} hyperparameter comparison plot to: {1}'.format(
			hyperparameter, train_acc_output_path))
		plt.legend()
		plt.savefig(train_acc_output_path)

	def number_of_completed_sweeps(self):
		return len(self._generated_configs)

