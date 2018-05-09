"""Pytorch models for visual gesture recognition."""

import torch
from base.base_model import BaseModel

class Model(BaseModel):
	"""TODO: Implement a basic ConvNet model for gesture recognition."""

	def __init__(self):
		super(Model, self).__init__()
		pass

	def forward(self, input):
		"""
	    In the forward function we accept a Tensor of input data and we must return
	    a Tensor of output data. We can use Modules defined in the constructor as
	    well as arbitrary (differentiable) operations on Tensors.
	    """
		raise NotImplemented
