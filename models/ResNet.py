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

		if pretrained:
			logging.info('Using model pretrained on imagenet.')
		else:
			logging.info('Model initialized with fresh weights.')

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
		self.generate_encoding = self._model_config.mode == 'pickle'

		# Freeze all the conv layers and only update the FC layers
		if self._model_config.freeze:
			for name, param in self._resnet.named_parameters():
				logging.info('Freezing ResNet parameter: {0}'.format(name))
				param.requires_grad = False
		self._resnet.fc.requires_grad = True

		# set initializer
		if not pretrained:
			for m in self.modules():
				classname = m.__class__.__name__
				if classname in modules_to_initialize:
					for name, param in m.named_parameters():
						if 'bias' in name:
							# TODO(kenny): Consider a dynamic bias initialization.
							logging.info('Initializing bias {0}.{1} with zeros.'.format(
								classname, name))
							nn.init.constant_(param, 0.0)
						elif 'weight' in name:
							logging.info('Initializing weight {0} using {1}.'.format(
								m, self._model_config.initializer))
							self._model_config.initializer_fn(param)

	def forward(self, X):
		""" Feeds frames into the ResNet"""
		logging.debug('Feeding input through pretrained resnet.')
		X = self._resnet(X)

		if self.generate_encoding and not self._model_config.load:
			# We return the original ResNet encodings since we aren't loading
			# a custom ResNet model.
			return X

		logging.debug('Feeding input through an additional hidden/encoding layer.')
		X = self._resnet_relu_fc(X)

		if self.generate_encoding:
			# We want the transfer-learned resnet if a model has been loaded.
			# This transfer-learned model has an additional 1000 x 1000 hidden
			# layer.
			return X

		logging.debug('Feeding input through the fully-connected layer.')
		return self._relu_fc(X)
