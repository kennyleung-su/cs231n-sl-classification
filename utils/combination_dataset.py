# Load a combination of ResNet encodings / pose estimation for LSTM

import glob
import logging
import numpy as np
import os
import re
import torch

from utils.resnet_encoding_dataset import ResnetEncodingDataset


class CombinationDataset(ResnetEncodingDataset):

	def __getitem__(self, idx):
		# TODO(kenny): Figure out how to sample with balanced labels.
		item = self.data[idx]

		# Tack on the (D, T) tensor representing the spatiotemporal frames for a video.
		video_dir = os.path.join(self._data_dir, item['video_dir'])
		item['frames'] = self.read_frame_tensors_from_dir(video_dir)
		return item

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
			video_dirs = self.get_video_dirs(label_dir)

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
