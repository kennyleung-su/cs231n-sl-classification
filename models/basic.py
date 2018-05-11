"""Pytorch models for visual gesture recognition."""

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
		self._resnet.features = torch.nn.DataParallel(self._resnet.features)
		self._resnet.cuda()

		# LSTM unrolls a len <= max_len_seq sequence of 1000d frame vectors.
		self._lstm = None

	def forward(self, input):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary (differentiable) operations on Tensors.
		"""
		print(input.size())
		cnn_out = self._resnet(input)
		return cnn_out
