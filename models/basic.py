"""Pytorch models for visual gesture recognition."""

import numpy as np
import torch
from base.base_model import BaseModel
from torchvision import transforms, utils, models


class PretrainedConvLSTMClassifier(BaseModel):
	"""TODO: Implement a basic ConvNet model for gesture recognition."""

	def __init__(self, *args, **kwargs):
		super(PretrainedConvLSTMClassifier, self).__init__(*args, **kwargs)
		self._resnet = self._model_config.pretrained_cnn_model(pretrained=True)
		for param in self._resnet.parameters():
			param.requires_grad = False

		# LSTM unrolls a len <= max_len_seq sequence of 1000d frame vectors.
		self._lstm = None

	def forward(self, input):
		"""Feeds the input through a pretrained ResNet, then through a variable-length LSTM network
		followed by a softmax classification layer."""
		N, C, T, H, W = input.shape
		example = torch.squeeze(input[:, :, 0, :, :])
		# Sanity purposes.
		print(self._resnet(example).shape)
		cnn_out = torch.stack([self._resnet(torch.squeeze(input[:, :, t, :, :])) for t in range(T)], dim=2)

		# TODO(kenny): Run an LSTM on cnn_out(dim=3) across each of N elements. Will need seq_len.
		pass
