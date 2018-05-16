# Implements data loading and preprocessing.

import csv
import glob
import imageio
import logging
import numpy as np
from utils.pad_utils import PadCollate
import pandas as pd
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader


class GestureFramesDataset(Dataset):
	"""Dataset of tensors corresponding to gesture video.

	Args:
		data_dir: str, path to the directory with folders of videos and frames
		transform: list of callable Classes, e.g. from torchvision.transforms,
			with which the video frames should be preprocessed
	"""
	def __init__(self, gesture_labels, data_dir, transform, pretrained_cnn, max_example_per_label,
		repickle_frames=False):
		self._pretrained_cnn = pretrained_cnn
		self._labels_to_indices_dict = self.map_labels_to_indices(gesture_labels)
		self._transform = transform
		self._max_example_per_label = max_example_per_label
		self._repickle_frames = repickle_frames
		logging.info('Reindexed labels: {0}'.format(self._labels_to_indices_dict))
		self.data, self.max_seq_len = self.populate_gesture_frames_data(data_dir, gesture_labels)
		self.len = len(self.data)
		logging.info('Initialized a GestureFramesDataset of size {0}.'.format(self.len))
		super(GestureFramesDataset, self).__init__()

	def __getitem__(self, idx):
		# TODO(kenny): Figure out how to sample with balanced labels.
		return self.data[idx]

	def __len__(self):
		return self.len

	def get_stacked_tensor_from_dir(self, directory):
		filenames = glob.glob(os.path.join(directory, "*.png"))
		matches = [re.match('.*_(\d+)\.png', name) for name in filenames]

		# sorted list of (frame_number, frame_path) tuples
		frames = sorted([(int(match.group(1)), match.group(0)) for match in matches])
		sorted_filenames = [f[1] for f in frames] 
		frames_list = []
		
		for frame_file in sorted_filenames:
			# Read an (H, W, C) shaped tensor.
			frame_ndarray = imageio.imread(frame_file)
			# Transform into a (C, H, W) shaped tensor where for Resnet H = W = 224
			frame_ndarray = self._transform(frame_ndarray)
			frames_list.append(frame_ndarray)
		# Stacks up to a (C, T, H, W) tensor.
		return torch.stack(frames_list, dim=1)

	def read_pretrained_cnn_encoded_frame_tensors_from_dir(self, directory):
		"""TODO: Description."""
		# TODO: Accommodate multiple types of pretrained CNNs. We might later want to
		# try other pretrained models, so we need to differentiate the pickled names.
		location = os.path.join(directory, '{0}-encoding.pkl'.format('resnet'))
		if os.path.isfile(location) and not self._repickle_frames:
			return torch.load(location)

		tensor = self.get_stacked_tensor_from_dir(directory)
		C, T, H, W = tensor.shape
		encoded_tensor = torch.stack(
			[self._pretrained_cnn(torch.unsqueeze(torch.squeeze(tensor[:, t, :, :]), dim=0)) for t in range(T)], dim=2
		)
		# Squeezing results in a (D, T) tensor.
		encoded_tensor = torch.squeeze(encoded_tensor, dim=0)

		# TODO: Figure out whether to rename this if we decide to use separate pretrained CNNs.
		# This name `pretrained_cnn' is kept general from now (as opposed to simply ResNet).
		if self._repickle_frames:
			logging.info('Pickling an encoded tensor of shape {0} to {1}.'.format(encoded_tensor.shape, location))
			torch.save(encoded_tensor, location)
		return encoded_tensor

	def read_original_frame_tensors_from_dir(self, directory):
		"""TODO: Description."""
		location = os.path.join(directory, 'original.pkl')
		if os.path.isfile(location):
			return torch.load(location)

		tensor = self.get_stacked_tensor_from_dir(directory)
		logging.info('Pickling a RGB tensor of shape {0} to {1}.'.format(tensor.shape, location))
		torch.save(tensor, location)
		return tensor

	def read_frame_tensors_from_dir(self, directory):
		"""TODO: Description."""
		if self._pretrained_cnn:
			return self.read_pretrained_cnn_encoded_frame_tensors_from_dir(directory)
		return self.read_original_frame_tensors_from_dir(directory)

	def populate_gesture_frames_data(self, data_dir, gesture_labels, type_data="kinect"):
		"""Returns a list of dicts with keys:
			'frames': 4D tensor (T, H, W, C) representing the spatiotemporal frames for a video
			'label': y (ground truth)
			'seq_len': number of frames in the video
		"""
		logging.info('Populating frame tensors for {0} specified labels in data dir {1}: {2}'.format(
			len(gesture_labels), data_dir, gesture_labels))

		labels_file = os.path.join(data_dir, '{0}_list.txt'.format(os.path.split(data_dir)[-1]))
		data = pd.read_csv(labels_file, sep=" ", header=None)
		data.columns = ["rgb", "kinect", "label"]

		if self._max_example_per_label:
			logging.info('Max example per label: {}'.format(self._max_example_per_label))

		label_to_dirs = {}
		for label in gesture_labels:
			directory_labels = data.loc[data['label'] == label][type_data].tolist()
			# cap the number of examples per label
			if self._max_example_per_label:
				directory_labels = directory_labels[:self._max_example_per_label]
			# strip .avi from the end of the filename
			directories = [''.join(dir_label.split('.avi')[:-1]) for dir_label in directory_labels]
			label_to_dirs[label] = directories

		data = []
		max_seq_len = -1


		for label, directories in label_to_dirs.items():
			logging.info('Reading frame tensors for label {0} ({1} videos)'.format(label, len(directories)))
			for directory in directories:
				frames = self.read_frame_tensors_from_dir(os.path.join(data_dir, directory))
				seq_len = frames.size(1)
				max_seq_len = max(max_seq_len, seq_len)
				data.append({
					'frames': frames,
					'label': self._labels_to_indices_dict[label],
					'seq_len': seq_len
				})

		return data, max_seq_len

	@staticmethod
	def map_labels_to_indices(gesture_labels):
		"""Returns a dict mapping the gesture labels to integer class labels."""
		return dict(zip(gesture_labels, range(len(gesture_labels))))


def GenerateGestureFramesDataLoader(gesture_labels, data_dir, max_seq_len,
				batch_size, transform, num_workers, pretrained_cnn, max_example_per_label,
				repickle_frames):
	"""Returns a configured DataLoader instance."""

	# Build a gesture frames dataset using the configuration information.
	# This is just dummy code to be replaced.
	transformed_dataset = GestureFramesDataset(gesture_labels, data_dir, transform,
		pretrained_cnn, max_example_per_label, repickle_frames)
	return DataLoader(transformed_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=PadCollate(transformed_dataset.max_seq_len, dim=1)
	)

def GetGestureFramesDataLoaders(data_dirs, model_config):
	"""Returns a tuple consisting of the train, valid, and test GestureFramesDataLoader objects."""
	# TODO: Support pickling to speed up the process. Perhaps we can hash the
	# list of gesture labels to a checksum and check if a file with that name exists.
	if model_config.pretrained_cnn_model:
		logging.info('Initializing the pretrained_cnn_model: {0}'.format(model_config.pretrained_cnn_model))
		pretrained_cnn = model_config.pretrained_cnn_model(pretrained=True)
		for param in pretrained_cnn.parameters():
			param.requires_grad = False
	else:
		pretrained_cnn = None
	return (GenerateGestureFramesDataLoader(
		model_config.gesture_labels,
		data_directory,
		model_config.max_seq_len,
		model_config.batch_size,
		model_config.transform,
		model_config.num_workers or 0,
		pretrained_cnn,
		model_config.max_example_per_label,
		model_config.repickle_frames,
		) for data_directory in data_dirs)
