"""Pytorch models for visual gesture recognition."""

# TODO: Add more logging.

from base.base_model import BaseModel

import logging
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class PretrainedResNetClassifier(BaseModel):
	""" A ResNet model that takes in frames and outputs the gesture recognition."""
	# TODO Add in regularizer for ResNet
	def __init__(self, *args, **kwargs):
		super(PretrainedResNetClassifier, self).__init__(*args, **kwargs)

		resnet_layer = self._model_config.resnet_num_layers
		self._resnet = None

		if resnet_layer == 18:
			self._resnet = models.resnet18(pretrained=True)
		elif resnet_layer == 34:
			self._resnet = models.resnet34(pretrained=True)
		elif resnet_layer == 50:
			self._resnet = models.resnet50(pretrained=True)
		elif resnet_layer == 101:
			self._resnet = models.resnet101(pretrained=True)
		elif resnet_layer == 152:
			self._resnet = models.resnet152(pretrained=True)
		else:
			raise ValueError('There are no models with the given number of layers.')

		self._resnet_relu = nn.ReLU(inplace=True)
		self._fc = nn.Linear(1000, self._num_output_classes)

		# set generate_encoding to True only when pickling the encodings
		self.generate_encoding = False

		# TODO freeze all the conv layers and only update the FC layers
		'''if self._model_config.freeze:
			for param in self._resnet.parameters():
				param.requires_grad = False

		self._resnet.fc.requires_grad = True'''

	def forward(self, X):
		""" Feeds frames into the ResNet"""
		logging.debug('Feeding input through pretrained resnet.')
		X = self._resnet(X)

		if self.generate_encoding:
			return X

		X = self._resnet_relu(X)

		logging.debug('Feeding input through the fully-connected layer.')
		return self._fc(X)
