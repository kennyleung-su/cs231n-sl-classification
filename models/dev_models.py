"""Dummy models for visual gesture recognition."""

from base.base_model import BaseModel

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


	def forward(self, input):
		"""
		In the forward function we accept a Tensor of input data of size (N, T, H, W, C)
		where N represents batch size, T represents time frames, H and W represent
		height and width, and C represents channels (typically, RGB). This function
		returns a Tensor of output classification labels (N,) by flattening the input
		and returning a fully-connected layer for a single frame T=0.
		"""
		N = input.shape[0]
		logits = torch.from_numpy(np.random.rand(N, self._num_output_classes))
		return F.log_softmax(logits, dim=1)

		# TODO: Use some sort of feed forward neural network to train parameters.
		# N, T, H, W, C = input.shape
		# single_frames = input[:, 0, :, :, :].view(N, -1)
		# return F.softmax(self._fc(single_frames))
