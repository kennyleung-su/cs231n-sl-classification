"""Base Pytorch models for visual gesture recognition."""

import torch
import torch.nn as nn
from torch.autograd import Variable

class BaseModel(nn.Module):
	"""TODO: Implement the base model for the project, primarily for reading in
	configurations and initializing the logger."""

	def __init__(self, model_config):
		super(BaseModel, self).__init__()
		self._model_config = model_config

	def forward(self, input):
		raise NotImplemented
