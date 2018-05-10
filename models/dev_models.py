"""Dummy models for visual gesture recognition."""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel

class DummyModel(BaseModel):
	"""TODO: Implement a basic ConvNet model for gesture recognition."""

	def __init__(self, *args, **kwargs):
		super(DummyModel, self).__init__(*args, **kwargs)
		self._num_output_classes = len(self._model_config.gesture_labels)

		# Image dimensions
		self._H = 240
		self._W = 360
		self._C = 3

		# Fully-connected layer with bias
		self._fc = nn.Linear(self._H * self._W * self._C, self._num_output_classes)


	def convert_label_to_one_hot(self, label):
		"""TODO: Convert between classes and one-hot label vectors. This may need to happen
		in a separate module, perhaps the trainer module.

		Suppose we have three gesture classes: 18, 43, and 100. We need to convert these
		indices to three separate one hot vectors (or integers starting from zero) in order
		to compute softmax CE-loss.
		"""
		pass


	def forward(self, input):
		"""
		In the forward function we accept a Tensor of input data of size (N, T, H, W, C)
		where N represents batch size, T represents time frames, H and W represent
		height and width, and C represents channels (typically, RGB). This function
		returns a Tensor of output classification labels (N,) by flattening the input
		and returning a fully-connected layer for a single frame T=0.
		"""
		N = input.shape[0]
		return np.random.randint(low=0, high=self._num_output_classes, size=N)

		# TODO: Use some sort of feed forward neural network to train parameters.
		N, T, H, W, C = input.shape
		single_frames = input[:, 0, :, :, :].view(N, -1)
		return F.softmax(self._fc(single_frames))
