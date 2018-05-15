"""Pytorch models for visual gesture recognition."""

# TODO: Add more logging.

from base.base_model import BaseModel

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, models


class PretrainedConvLSTMClassifier(BaseModel):
	""" A basic fixed-ConvNet LSTM model for gesture recognition.

	Assumes that the input consists of ResNet18 (1000d) encoded DxT frames.
	"""

	_RESNET_OUTPUT_SIZE = 1000

	def __init__(self, *args, **kwargs):
		super(PretrainedConvLSTMClassifier, self).__init__(*args, **kwargs)

		# TODO: Allow for the configs to toggle all of these hyperparameters.
		self._lstm = nn.LSTM(
			input_size=self._RESNET_OUTPUT_SIZE,
			hidden_size=self._model_config.lstm_hidden_size,
			num_layers=1,
			bias=False,
			batch_first=True,
			dropout=0.0,
			bidirectional=False
		)
		self._fc = nn.Linear(self._model_config.lstm_hidden_size, self._num_output_classes)

	def forward(self, input):
		"""Feeds the pretrained ResNet-encoded input through a variable-length LSTM network
		followed by a softmax classification layer."""
		X, seq_lens = input['X'], input['seq_lens']
		print(X.shape)
		N, D, T = X.shape

		# Packing takes (N, T, *) if batch_first=True.
		# https://pytorch.org/docs/master/nn.html#torch.nn.utils.rnn.pack_padded_sequence
		X = X.permute(0, 2, 1)
		packed_resnet = torch.nn.utils.rnn.pack_padded_sequence(X, seq_lens,
			batch_first=True)

		# LSTM unrolls a len <= max_len_seq sequence of 1000d frame vectors.
		logging.debug('Feeding input through LSTM.')
		packed_lstm_out, (packed_h_t, packed_c_t) = self._lstm(packed_resnet)

		# At this point, LSTM output yields a (max_seq_len, N, lstm_hidden_size) tensor.
		# We extract the last frame from the LSTM as the sequence's final encoding.
		logging.debug('Feeding input through fully-connected layer.')
		return F.log_softmax(self._fc(packed_h_t.view(N, -1)), dim=1)
