# Load ResNet encodings for LSTM

import glob
import imageio
import logging
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset

try:
   import cPickle as pickle
except:
   import pickle


class ResnetEncodingDataset(Dataset):
	"""Dataset of tensors corresponding to gesture video.

	Args:
		data_dir: str, path to the directory with folders of videos and frames
	"""
	def __init__(self, gesture_labels, data_dir, data_type, max_example_per_label):
		self._labels_to_indices_dict = self.map_labels_to_indices(gesture_labels)
		self._max_example_per_label = max_example_per_label
		logging.info('Reindexed labels: {0}'.format(self._labels_to_indices_dict))
		self._data_dir = data_dir
		self._data_type = data_type
		# Initialize data locations for lazy loading
		self.data, self.max_seq_len = self.populate_encoding_data(gesture_labels)
		self.len = len(self.data)
		logging.info('Initialized a ResnetEncodingDataset of size {0}.'.format(self.len))
		super(ResnetEncodingDataset, self).__init__()

	def __getitem__(self, idx):
		# TODO(kenny): Figure out how to sample with balanced labels.
		item = self.data[idx]

		# Tack on the (D, T) tensor representing the spatiotemporal frames for a video.
		video_dir = os.path.join(self._data_dir, item['video_dir'])
		logging.debug('Fetching gesture video for: {0}: {1}'.format(idx, video_dir)) 
		item['frames'] = self.read_frame_tensors_from_dir(video_dir, self._data_type)
		return item

	def __len__(self):
		return self.len

	def populate_encoding_data(self, gesture_labels):
		"""Returns a list of dicts with keys:
			'label': y (ground truth)
			'seq_len': number of frames in the video
			'video_dir': data-dir-relative directory where videos are stored
		"""
		logging.info('Populating frame tensors for {0} specified labels in data dir {1}: {2}'.format(
			len(gesture_labels), self._data_dir, gesture_labels))

		if self._max_example_per_label:
			logging.info('Max example per label: {}'.format(self._max_example_per_label))

		data = []
		max_seq_len = -1

		for label in gesture_labels:
			label_dir = os.path.join(self._data_dir, str(label))
			video_dirs = self.get_video_dirs(label_dir, self._data_type)

			# cap the number of images per label
			if self._max_example_per_label:
				video_dirs = video_dirs[:self._max_example_per_label]

			logging.info('Assigning frame tensor locations for label: {0} ({1} videos)'.format(
				label, len(video_dirs)))
			for video_dir in video_dirs:
				# Keep track of the global max seq len for batch RNN unrolling purposes.
				seq_len = len(glob.glob(os.path.join(video_dir, '*.png')))
				max_seq_len = max(max_seq_len, seq_len)
				data.append({
					'label': self._labels_to_indices_dict[label],
					'video_dir': video_dir,
					'seq_len': seq_len
				})

		return data, max_seq_len

	def read_frame_tensors_from_dir(self, directory, data_type):
		location = os.path.join(directory, '{}-encoding.pt'.format(data_type))
		return pickle.load(open(location, 'rb'))

	def get_video_dirs(self, label_dir, data_type):

		# return a list of paths for the images
		dir_prefix = None
		file_prefix = None

		if data_type.startswith('OF'):  # optical flow images, which has both RGB and RGBD variants.
			file_prefix = 'OF'
			data_type = data_type.lstrip('OF')
		if data_type.endswith('RGB'):
			dir_prefix = 'M_'
		elif data_type.endswith('RGBD'):
			dir_prefix = 'K_'
		else:
			raise ValueError('Data type for Gesture Frame Dataloader is invalid')

		return sorted(glob.glob(os.path.join(label_dir, '{0}*/'.format(
			dir_prefix))))

	@staticmethod
	def map_labels_to_indices(gesture_labels):
		"""Returns a dict mapping the gesture labels to integer class labels."""
		return dict(zip(gesture_labels, range(len(gesture_labels))))
