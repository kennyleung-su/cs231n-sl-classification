# Load gesture frames for ResNet
import glob
import imageio
import logging
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GestureFrameDataset(Dataset):
	"""Dataset of tensors corresponding to gesture video.

	Args:
		data_dir: str, path to the directory with folders of videos and frames
		transform: list of callable Classes, e.g. from torchvision.transforms,
			with which the video frames should be preprocessed
	"""
	def __init__(self, gesture_labels, data_dir, data_type, transform, max_example_per_label):
		self._labels_to_indices_dict = self.map_labels_to_indices(gesture_labels)
		self._transform = transform
		self._max_example_per_label = max_example_per_label
		logging.info('Reindexed labels: {0}'.format(self._labels_to_indices_dict))
		self._data_type = data_type
		self._data_dir = data_dir
		# Populate data about where to find and load the videos.
		self.data = self.populate_gesture_frame_data(gesture_labels)
		self.len = len(self.data)
		logging.info('Initialized a GestureFrameDataset of size {0}.'.format(self.len))
		super(GestureFrameDataset, self).__init__()

	def __getitem__(self, idx):
		# TODO(kenny): Figure out how to sample with balanced labels.
		# Lazy load image from the given path.
		image_data = self.data[idx]
		label, image_path = image_data['label'], image_data['image_path']
		# Read a (H, W, C) shaped tensor
		logging.debug('Reading image: {0}'.format(image_path))
		try:
			image_ndarray = imageio.imread(image_path)
		except:
			logging.debug('Could not read image: {0}'.format(image_path))
			return self.__getitem__(np.random.randint(self.len))
		# Transform into a (C, H, W) shaped tensor where for Resnet H = W = 224
		# print('before:', frame_ndarray)
		image_ndarray = self._transform(image_ndarray)
		return (image_ndarray, self._labels_to_indices_dict[label])

	def __len__(self):
		return self.len

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
			imagefiles = self.get_imagefiles(label_dir)

			# cap the number of images per label
			if self._max_example_per_label:
				imagefiles = imagefiles[:self._max_example_per_label]

			logging.info('Assigning imagefile paths for label: {0} ({1} images)'.format(label, len(imagefiles)))
			for imagefile in imagefiles:
				data.append({
					'label': label,
					'image_path': imagefile
				})

		return data

	def get_imagefiles(self, label_dir):
		# return a list of paths for the images
		dir_prefix = None
		file_prefix = None
		data_type = self._data_type

		if data_type.startswith('OF'):  # optical flow images, which has both RGB and RGBD variants.
			file_prefix = 'OF'
			file_suffix = 'stride1'
			data_type = data_type.lstrip('OF')
		if data_type == 'RGB':
			dir_prefix = 'M_'
		elif data_type == 'RGBD':
			dir_prefix = 'K_'
		else:
			raise ValueError('Data type for Gesture Frame Dataloader is invalid')

		return glob.glob(os.path.join(label_dir, '{0}*/{1}*{2}.png'.format(
			dir_prefix, file_prefix or dir_prefix, file_suffix or '')))

	@staticmethod
	def map_labels_to_indices(gesture_labels):
		"""Returns a dict mapping the gesture labels to integer class labels."""
		return dict(zip(gesture_labels, range(len(gesture_labels))))
