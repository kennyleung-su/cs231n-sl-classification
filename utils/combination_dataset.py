# Load a combination of ResNet encodings / pose estimation for LSTM

import glob
import imageio
import logging
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset


class CombinationEncodingDataset(Dataset):

	def __init__(self, gesture_labels, data_dir, data_type, max_example_per_label):
		self._labels_to_indices_dict = self.map_labels_to_indices(gesture_labels)
		self._max_example_per_label = max_example_per_label
		logging.info('Reindexed labels: {0}'.format(self._labels_to_indices_dict))
		self.data, self.max_seq_len = self.populate_encoding_data(data_dir, gesture_labels, data_type)
		self.len = len(self.data)
		logging.info('Initialized a ResnetEncodingDataset of size {0}.'.format(self.len))
		super(ResnetEncodingDataset, self).__init__()

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return self.len