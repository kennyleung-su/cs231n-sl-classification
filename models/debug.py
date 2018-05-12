"""Dummy models for visual gesture recognition."""

from base.base_model import BaseModel

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomClassifier(BaseModel):
	def __init__(self, *args, **kwargs):
		super(RandomClassifier, self).__init__(*args, **kwargs)
		self._num_output_classes = len(self._model_config.gesture_labels)

	def forward(self, input):
		"""Returns a random probability distribution across all output classes."""
		N = input.shape[0]
		logits = np.random.rand(N, self._num_output_classes)
		return F.log_softmax(logits, dim=1)


class LinearClassifier(BaseModel):
	def __init__(self, *args, **kwargs):
		super(LinearClassifier, self).__init__(*args, **kwargs)

		# Hardcoded image dimensions
		# TODO: Infer this from the configurations or pass them into the model.
		self._H = 224
		self._W = 224
		self._C = 3

		# Fully-connected layer with bias
		self._fc = nn.Linear(self._H * self._W * self._C, self._num_output_classes)


	def forward(self, input):
		"""Uses a single fully-connected layer to classify all videos based
		only on the flattened RGB values for the first timeframe."""
		X = input['X']
		N, C, T, H, W = X.shape
		flattened_frames = X[:, :, 0, :, :].contiguous().view(N, -1)
		logits = self._fc(flattened_frames)
		return F.log_softmax(logits, dim=1)
