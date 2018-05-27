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
		pretrained = self._model_config.pretrained
		self._resnet = None

		if resnet_layer == 18:
			self._resnet = models.resnet18(pretrained=pretrained)
		elif resnet_layer == 34:
			self._resnet = models.resnet34(pretrained=pretrained)
		elif resnet_layer == 50:
			self._resnet = models.resnet50(pretrained=pretrained)
		elif resnet_layer == 101:
			self._resnet = models.resnet101(pretrained=pretrained)
		elif resnet_layer == 152:
			self._resnet = models.resnet152(pretrained=pretrained)
		else:
			raise ValueError('There are no models with the given number of layers.')

		# An additional FC layer is used to map ResNet encodings to an additional
		# 1000d latent representation, so that we can freeze the original deep ResNet
		# model but train this additional FC layer to learn better encodings.
		self._resnet_relu_fc = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Linear(self._RESNET_OUTPUT_SIZE, self._RESNET_OUTPUT_SIZE)
		)

		# This final layer handles classification by mapping the previous 1000d layer
		# into the output-space dimension.
		self._relu_fc = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Linear(self._RESNET_OUTPUT_SIZE, self._num_output_classes)
		)

		# Set generate_encoding to True only when pickling the encodings
		self.generate_encoding = False

		# Freeze all the conv layers and only update the FC layers
		if self._model_config.freeze:
			for name, param in self._resnet.named_parameters():
				logging.info('Freezing ResNet parameter: {0}'.format(name))
				param.requires_grad = False
		self._resnet.fc.requires_grad = True

	def forward(self, X):
		""" Feeds frames into the ResNet"""
		logging.debug('Feeding input through pretrained resnet.')
		X = self._resnet(X)

		logging.debug('Feeding input through an additional hidden/encoding layer.')
		X = self._resnet_relu_fc(X)

		if self.generate_encoding:
			return X

		logging.debug('Feeding input through the fully-connected layer.')
		return self._relu_fc(X)
