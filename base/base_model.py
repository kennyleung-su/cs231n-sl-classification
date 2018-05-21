"""Base Pytorch models for visual gesture recognition."""

import os
import time
import logging
from configs import config
import torch
import torch.nn as nn
from torch.autograd import Variable

class BaseModel(nn.Module):
	"""Base model for the project, primarily for reading in
	configurations and initializing the logger."""

	def __init__(self, model_config):
		super(BaseModel, self).__init__()
		self._model_config = model_config
		self._num_output_classes = len(self._model_config.gesture_labels)
		self.training_epoch = 0
		self.best_accuracy = -1

	def load_checkpoint(self, filename):
		# The checkpoint is a dictionary consisting of keys 'best_accuracy', 'epoch', 'state_dict'
		checkpoint = torch.load(filename)
		self.load_state_dict(checkpoint.get('state_dict'))
		self.training_epoch = checkpoint.get('epoch')
		self.best_accuracy = checkpoint.get('best_accuracy')

	def save_to_checkpoint(self, path, is_best=False):
		""""Saves information about the model to the checkpoint file.

		If is_best = True, append to the filename and also copies this model to a 
		directory containing the best-performing models.
		"""
		filename = os.path.join(path, '{}-{}'.format(self._model_config.experiment, time.time()))

		if is_best:
			filename += '-best-{:d}.pkl'.format(self.best_accuracy)
			''' TODO: If is_best = True, also copies this model information to a directory
			containing the best-performing models.'''
		else:
			filename += '.pkl'

		logging.info('Model saved to checkpoint: {}'.format(filename))
		torch.save({
			'best_accuracy': self.best_accuracy,
			'epoch': self.training_epoch,
			'state_dict': self.state_dict()
		}, filename)

	def forward(self, input):
		"""To be implemented by each derived class."""
		raise NotImplemented