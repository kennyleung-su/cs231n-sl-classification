# Load ResNet encodings for LSTM

import glob
import imageio
import logging
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset



class ResnetEncodingDataset(Dataset):
	"""Dataset of tensors corresponding to gesture video.

	Args:
		data_dir: str, path to the directory with folders of videos and frames
	"""
	def __init__(self, gesture_labels, data_dir, data_type, max_example_per_label):
		self._labels_to_indices_dict = self.map_labels_to_indices(gesture_labels)
		self._max_example_per_label = max_example_per_label
		logging.info('Reindexed labels: {0}'.format(self._labels_to_indices_dict))
		self.data, self.max_seq_len = self.populate_encoding_data(data_dir, gesture_labels, data_type)
		self.len = len(self.data)
		logging.info('Initialized a ResnetEncodingDataset of size {0}.'.format(self.len))
		super(ResnetEncodingDataset, self).__init__()

	def __getitem__(self, idx):
		# TODO(kenny): Figure out how to sample with balanced labels.
		return self.data[idx]

	def __len__(self):
		return self.len

	def populate_encoding_data(self, data_dir, gesture_labels, data_type):
		"""Returns a list of dicts with keys:
			'frames': 4D tensor (T, H, W, C) representing the spatiotemporal frames for a video
			'label': y (ground truth)
			'seq_len': number of frames in the video
		"""
		logging.info('Populating frame tensors for {0} specified labels in data dir {1}: {2}'.format(
			len(gesture_labels), data_dir, gesture_labels))

		if self._max_example_per_label:
			logging.info('Max example per label: {}'.format(self._max_example_per_label))

		data = []
		max_seq_len = -1

		for label in gesture_labels:
			label_dir = os.path.join(data_dir, str(label))
			video_dirs = self.get_video_dirs(label_dir, data_type)

			# cap the number of images per label
			if self._max_example_per_label:
				video_dirs = video_dirs[:self._max_example_per_label]

			logging.info('Reading frame tensors for label {0} ({1} video)'.format(label, len(video_dirs)))
			for video_dir in video_dirs:
				frames = self.read_frame_tensors_from_dir(os.path.join(data_dir, video_dir), data_type)
				seq_len = frames.size(1)
				max_seq_len = max(max_seq_len, seq_len)
				data.append({
					'frames': frames,
					'label': self._labels_to_indices_dict[label],
					'seq_len': seq_len
				})

		return data, max_seq_len

	def read_frame_tensors_from_dir(self, directory, data_type):
		location = os.path.join(directory, '{}-encoding.pkl'.format(data_type))
		return torch.load(location)

	def get_video_dirs(self, label_dir, data_type):

		# return a list of paths for the images
		prefix = None
		if data_type.endswith('RGB'):
			prefix = 'M_'
		elif data_type.endswith('RGBD'):
			prefix = 'K_'
		else:
			raise ValueError('Data type for pickling is invalid')

		return glob.glob(os.path.join(label_dir, '{0}*/'.format(prefix)))

	@staticmethod
	def map_labels_to_indices(gesture_labels):
		"""Returns a dict mapping the gesture labels to integer class labels."""
		return dict(zip(gesture_labels, range(len(gesture_labels))))