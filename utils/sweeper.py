"""Module for sweeping a set of hyperparameters for a given experiment, and analyzing the results."""
import copy
from configs.config import ConfigObjFromDict
from enum import Enum
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from utils.metrics import AccuracySaver, LossSaver, PredictionSaver

class ValueType(Enum):
		DISCRETE = 1
		CONTINUOUS = 2
		PROBABILITY	= 3


class HyperparameterOption(object):
	def __init__(self, value_type, value_range=(1, 10), exp_range=None, round_to=8):
		self._value_type = value_type
		# Values are range^(exp_range) drawn uniformly using (low, high) tuples.
		self._range = value_range
		self._exp_range = exp_range
		self._round_to = round_to
		self._eps = 1e-8  # smallest possible value that can be returned.
		assert len(self._range) == 2, "Value range must be of length 2."
		assert self._exp_range is None or len(self._exp_range) == 2, "Exponent range must be of length 2."

	def sample_exp(self):
		return np.random.randint(*self._exp_range) if self._exp_range else 0

	def sample(self):
		if self._value_type == ValueType.DISCRETE:
			# Sample a discrete value
			value = int(np.random.randint(*self._range) * 10**self.sample_exp())
		elif self._value_type == ValueType.CONTINUOUS:
			# Sample a continuous value
			value = np.around(np.random.randint(*self._range) * 10**self.sample_exp(), self._round_to)
		elif self._value_type == ValueType.PROBABILITY:
			# Sample uniformly in between the range for a probabilistic value, like dropout
			print('randuniform', self._range)
			return np.around(np.random.uniform(*self._range), self._round_to)
		return max(self._eps, value)


class ModelConfigMetadataWrapper(ConfigObjFromDict):
	"""Wrapper around a configuration object that tracks associated statistics metadata."""

	def __init__(self, sampled_config, metrics_dir, *args, **kwargs):
		super(ModelConfigMetadataWrapper, self).__init__(*args, **kwargs)
		# These public values can be set by the main program generating statistics from
		# the model config. They are of type MetricsCsvSaver.
		self.sampled_config = sampled_config

		# TODO: Auto-initialize these. Requires the directory to the config.
		self.train_acc_saver = AccuracySaver(
			os.path.join(metrics_dir, '{0}.{1}'.format(self, 'train_acc')))
		self.valid_acc_saver = AccuracySaver(
			os.path.join(metrics_dir, '{0}.{1}'.format(self, 'valid_acc')))
		self.train_loss_saver = LossSaver(
			os.path.join(metrics_dir, '{0}.{1}'.format(self, 'train_loss')))
		self.preds_saver = PredictionSaver(
			os.path.join(metrics_dir, '{0}.{1}'.format(self, 'preds')),
				num_classes=len(kwargs['gesture_labels']))

	def __str__(self):
		if not self.sampled_config:
			# If we aren't sweeping, then just return 'default' + the time of the
			# experiment because it's just using the default parameters.
			return 'default.{0}'.format(time.time())
		args = []
		for k, v in sorted(self.sampled_config.items()):
			args.append('{0}={1}'.format(k, v))
		return '_'.join(args)

	def analyze(self):
		# TODO: Add useful analysis code.
		metric_name_to_saver = {
			'training loss': self.train_loss_saver,
			'training accuracy': self.train_acc_saver,
			'validation accuracy': self.valid_acc_saver
		}
		logging.info('\n====== Experiment: {0} ======'.format(self.sampled_config))
		for (name, saver) in metric_name_to_saver.items():
			logging.info('{0} - min:{1}\tmax:{2}\tavg:{3}'.format(
				name,
				np.around(saver.meter.min, 2),
				np.around(saver.meter.max, 2),
				np.around(saver.meter.avg, 2))
			)


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
			print('Sampling', attr)
			config[attr] = option.sample()
		return config

	def analyze_performance(self):
		for config in self._generated_configs:
			config.analyze()

	def analyze_hyperparameter(self, hyperparameter):
		"""Plots data on the same graph across all hyperparameters."""

		# TODO: Clean up this logic. Sorry for the mess!
		sorted_configs = sorted(self._generated_configs, key=lambda x: x.sampled_config[hyperparameter])

		plt.figure()
		plt.xlabel('Epochs')
		plt.ylabel('Training Loss')
		plt.title('Average Training Loss Across Epochss')
		for config in sorted_configs:
			plt.plot(*config.train_loss_saver.get_plot_points(),
				label='{0} = {1}'.format(hyperparameter, config.sampled_config[hyperparameter]))
		train_loss_output_path = os.path.join(self._plots_dir, 'train_loss.{0}.{1}.{2}.png'.format(
			hyperparameter, self._model_config.experiment, time.time()))
		logging.info('Saving {0} hyperparameter comparison plot to: {1}'.format(
			hyperparameter, train_loss_output_path))
		plt.legend()
		plt.savefig(train_loss_output_path)

		plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Validation Accuracy')
		plt.title('Validation Accuracy Across Epochs')
		for config in sorted_configs:
			plt.plot(*config.valid_acc_saver.get_plot_points(),
				label='{0} = {1}'.format(hyperparameter, config.sampled_config[hyperparameter]))
		valid_acc_output_path = os.path.join(self._plots_dir, 'valid_acc.{0}.{1}.{2}.png'.format(
			hyperparameter, self._model_config.experiment, time.time()))
		logging.info('Saving {0} hyperparameter comparison plot to: {1}'.format(
			hyperparameter, valid_acc_output_path))
		plt.legend()
		plt.savefig(valid_acc_output_path)

		plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Training Accuracy')
		plt.title('Training Accuracy Across Epochs')
		for config in sorted_configs:
			plt.plot(*config.train_acc_saver.get_plot_points(),
				label='{0} = {1}'.format(hyperparameter, config.sampled_config[hyperparameter]))
		train_acc_output_path = os.path.join(self._plots_dir, 'train_acc.{0}.{1}.{2}.png'.format(
			hyperparameter, self._model_config.experiment, time.time()))
		logging.info('Saving {0} hyperparameter comparison plot to: {1}'.format(
			hyperparameter, train_acc_output_path))
		plt.legend()
		plt.savefig(train_acc_output_path)

	def analyze_train_vs_valid_accuracy(self):
		for config in self._generated_configs:
			plt.figure()
			plt.xlabel('Epoch')
			plt.ylabel('Accuracy')
			plt.title('Training vs. Validation Accuracy Across Epochs')
			output_path = os.path.join(self._plots_dir, 'train_vs_val_acc.{0}.{1}.{2}.png'.format(
				self._model_config.experiment, config, time.time()))
			plt.plot(*config.train_acc_saver.get_plot_points(), label='Train')
			plt.plot(*config.valid_acc_saver.get_plot_points(), label='Valid')
			plt.legend()
			plt.savefig(output_path)


	def analyze_confusion(self):
		for config in self._generated_configs:
			config.preds_saver.plot(
				os.path.join(self._plots_dir, 'heatmap.{0}.{1}.png'.format(
					self._model_config.experiment,
					time.time())),
				'Gesture classification confusion matrix')

	def number_of_completed_sweeps(self):
		return len(self._generated_configs)

