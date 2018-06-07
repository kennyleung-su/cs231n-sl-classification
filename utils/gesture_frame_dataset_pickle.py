# for debugging
import sys

# Load gesture frames for ResNet
import glob
import imageio
import logging
import numpy as np
import os
import re
import torch

from utils.gesture_frame_dataset import GestureFrameDataset


class GestureFrameDatasetPickle(GestureFrameDataset):
	"""Dataset of tensors corresponding to gesture video."""
	def __init__(self, gesture_labels, data_dir, data_type, transform, max_example_per_label, pickle_config):
		# set pickling variable
		self._overwrite = pickle_config['pickle_overwrite']
		transfer_learning_prefix = 'T' if pickle_config['load'] else ''
		self._encoding_filename = '{0}RN{1}{2}-encoding.pt'.format(transfer_learning_prefix ,
															str(pickle_config['resnet_num_layers']),
													  		data_type)
		super(GestureFrameDatasetPickle, self).__init__(gesture_labels, data_dir, data_type, transform, max_example_per_label)

	def __getitem__(self, idx):	
		video_dir = self.data[idx]
		# get video tensor
		video_tensor = self.get_video_tensor_for_dir(video_dir, self._transform, self._data_type)
		video_path = os.path.join(video_dir, self._encoding_filename)
		return (video_tensor, video_path)

	def populate_gesture_frame_data(self, gesture_labels):
		"""Returns a list of dicts with keys:
			'frames': 3D tensor (N, W, C) representing the frames for a video
			'label': y (ground truth)
		"""
		logging.info('Populating frame tensors for {0} specified labels in data dir {1}: {2}'.format(
			len(gesture_labels), self._data_dir, gesture_labels))

		if self._max_example_per_label:
			logging.info('Max example per label: {}'.format(self._max_example_per_label))

		data = []

		for label in gesture_labels:
			label_dir = os.path.join(self._data_dir, str(label))
			video_dirs = self.get_video_dirs(label_dir, self._data_type)

			# cap the number of images per label
			if self._max_example_per_label:
				video_dirs = video_dirs[:self._max_example_per_label]

			logging.info('Assigning video paths for label: {0} ({1} videos)'.format(label, len(video_dirs)))
			for video_dir in video_dirs:
				video_path = os.path.join(video_dir, self._encoding_filename)
				# skip videos that have already been pickled
				if not self._overwrite and os.path.exists(video_path):
					continue
				# if it exist, and overwrite is set to true, overwrite the files
				if self._overwrite and os.path.exists(video_path):
					os.remove(video_path)

				data.append(video_dir)

		return data

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

	def get_video_tensor_for_dir(self, video_dir, transform, data_type):
		if data_type.startswith('OF'):
			file_prefix = 'OF'
		elif data_type.endswith('RGB'):
			file_prefix = 'M'
		elif data_type.endswith('RGBD'):
			file_prefix = 'K'

		filenames = glob.glob(os.path.join(video_dir, '{0}_*.png'.format(file_prefix)))
		matches = [re.match(r'.*_(\d+)\.png', name) for name in filenames]
		# sorted list of (frame_number, frame_path) tuples
		frames = sorted([(int(match.group(1)), match.group(0)) for match in matches])
		sorted_filenames = [f[1] for f in frames] 
		frames_list = []
		for frame_file in sorted_filenames:
			# Read an (H, W, C) shaped tensor.
			frame_ndarray = imageio.imread(frame_file)
			# Transform into a (C, H, W) shaped tensor where for Resnet H = W = 224
			frame_ndarray = transform(frame_ndarray)
			frames_list.append(frame_ndarray)
		# Stacks up to a (T, C, H, W) tensor.
		tensor = torch.stack(frames_list, dim=0)

		return tensor