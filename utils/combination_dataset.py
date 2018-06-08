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
		video_dir_list = item['video_dir_list']
		frames_list = []

		for idx, video_dir in enumerate(video_dir_list):
			video_dir = os.path.join(self._data_dir, video_dir)
			data_type = self._data_types[idx]
			frames_list.append(self.read_frame_tensors_from_dir(video_dir, data_type))
		#try:
		item['frames'] = torch.cat(frames_list, 0)
		#except RuntimeError:
		#	return self.__getitem__(0)

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

		data_type_str = self._data_type[1:-1]
		data_types = data_type_str.split('+')
		self._data_types = data_types
		
		for label in gesture_labels:
			label_dir = os.path.join(self._data_dir, str(label))
			video_dirs_list = []

			for data_type in data_types:
				video_dirs = self.get_video_dirs(label_dir, data_type)

				# cap the number of images per label
				if self._max_example_per_label:
					video_dirs = video_dirs[:self._max_example_per_label]

				video_dirs_list.append(video_dirs)

			logging.info('Assigning frame tensor locations for label: {0} ({1} videos)'.format(
				label, len(video_dirs)))

			for video_idx, video_dir in enumerate(video_dirs_list[0]):
				# Keep track of the global max seq len for batch RNN unrolling purposes.
				seq_len = len(glob.glob(os.path.join(video_dir, '*.png')))
				max_seq_len = max(max_seq_len, seq_len)
				video_dir_list = [video_dir]

				for video_dirs in video_dirs_list[1:]:
					video_dir_list.append(video_dirs[video_idx])

				data.append({
					'label': self._labels_to_indices_dict[label],
					'video_dir_list': video_dir_list,
					'seq_len': seq_len
				})

		return data, max_seq_len
