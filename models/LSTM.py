"""Pytorch models for visual gesture recognition."""

# TODO: Add more logging.

from base.base_model import BaseModel

import logging
import numpy as np
import torch
import torch.nn as nn


class EncodingLSTMClassifier(BaseModel):
	""" A basic LSTM model that takes in ResNet encodings for gesture recognition.

	Assumes that the input consists of ResNet (1000d) encoded DxT frames.
	"""

	# TODO Add in regularizer for LSTM

	_RESNET_OUTPUT_SIZE = 1000	# Fixed variable

	def __init__(self, *args, **kwargs):
		super(EncodingLSTMClassifier, self).__init__(*args, **kwargs)

		self._lstm = nn.LSTM(
			input_size=self._RESNET_OUTPUT_SIZE,
			hidden_size=self._model_config.lstm_hidden_size,
			num_layers=self._model_config.lstm_num_layers,
			bias=self._model_config.lstm_bias,
			batch_first=self._model_config.lstm_,
			dropout=self._model_config.lstm_batch_first,
			bidirectional=self._model_config.lstm_bidirectional
		)

		H = self._model_config.lstm_hidden_fc_size 
		self._fc = nn.Sequential(
          	nn.Linear(self._model_config.lstm_hidden_size, H),
			nn.ReLU(),
			nn.Linear(H, H),
			nn.ReLU(),
			nn.Linear(H, self._num_output_classes),
		)

		self._fc_basic = nn.Linear(self._model_config.lstm_hidden_size,
								   self._num_output_classes)

		# pass the weights through the initializer
		for m in self.modules():
			print(m) #TODO set proper initialization


	def forward(self, input):
		""" Feeds the pretrained ResNet-encoded input through a variable-length LSTM network. """
		X, seq_lens = input['X'], input['seq_lens']
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
		return self._fc_basic(packed_h_t.view(N, -1))
