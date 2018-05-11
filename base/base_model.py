"""Base Pytorch models for visual gesture recognition."""

from configs import config
import torch
import torch.nn as nn
from torch.autograd import Variable

class BaseModel(nn.Module):
	"""TODO: Implement the base model for the project, primarily for reading in
	configurations and initializing the logger."""

	def __init__(self, model_config):
		super(BaseModel, self).__init__()
		self._model_config = model_config
		self._num_output_classes = len(self._model_config.gesture_labels)
		self.training_epoch = 0
		self._best_accuracy = -1

	def save_to_checkpoint(self, filename, is_best=False):
		""""Saves information about the model to the checkpoint file.

		If is_best = True, also copies this model information to a directory
		containing the best-performing models.
		"""
		torch.save({
			'epoch': self.training_epoch,
			'state_dict': self.state_dict()
		}, filename)
		if is_best:
			raise NotImplemented

	def forward(self, input):
		"""To be implemented by each derived class."""
		raise NotImplemented