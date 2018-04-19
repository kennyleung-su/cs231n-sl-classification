"""Base Pytorch models for visual gesture recognition."""

import torch
import torch.nn as nn
from torch.autograd import Variable

class BaseModel(nn.Module):
	"""TODO: Implement the base model for the project, primarily for reading in
	configurations and initializing the logger."""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, input):
        raise NotImplemented
