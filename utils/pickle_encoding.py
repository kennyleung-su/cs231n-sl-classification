# Pickles the output of the model into the individual folders
import glob
import imageio
import logging
import numpy as np
import os
import re
import torch

def pickle_encoding(data_dirs, model_config, model):
	# set generate_encoding to True to capture the encodings before the last layer
	model.eval()
	model.generate_encoding = True
	max_example_per_label = model_config.max_example_per_label
	transform = model_config.transform
	gesture_labels = model_config.gesture_labels
	data_type, _ = model_config.dataloader_type.split('-')
	transfer_learning_prefix = 'T' if model_config.load else ''
	encoding_filename = '{0}RN{1}{2}-encoding.pkl'.format(transfer_learning_prefix ,str(model_config.resnet_num_layers),
												  data_type)
	# encoding with the prefix T means transfer learning has been carried out. else, it takes the output of 
	# the original imagenet pretrained resnet

	if max_example_per_label:
		logging.info('Max example per label: {}'.format(max_example_per_label))

	for data_dir in data_dirs:
		for label in gesture_labels:
			label_dir = os.path.join(data_dir, str(label))
			video_dirs = get_video_dirs(label_dir, data_type)

			if max_example_per_label:
				video_dirs = video_dirs[:max_example_per_label]

			for video_dir in video_dirs:
				save_video_encoding_to_dir(video_dir, model, transform, encoding_filename)


def save_video_encoding_to_dir(video_dir, model, transform, encoding_filename):
	filenames = glob.glob(os.path.join(video_dir, '*.png'))
	matches = [re.match(r'.*_(\d+)\.png', name) for name in filenames]
	# sorted list of (frame_number, frame_path) tuples
	frames = sorted([(int(match.group(1)), match.group(0)) for match in matches])
	sorted_filenames = [f[1] for f in frames] 
	frames_list = []
	for frame_file in sorted_filenames:
		# Read an (H, W, C) shaped tensor.
		frame_ndarray = imageio.imread(frame_file)
		# Transform into a (C, H, W) shaped tensor where for Resnet H = W = 224
		# print('before:', frame_ndarray)
		frame_ndarray = transform(frame_ndarray)
		frames_list.append(frame_ndarray)
	# Stacks up to a (C, T, H, W) tensor.
	tensor = torch.stack(frames_list, dim=1)
	C, T, H, W = tensor.shape
	encoded_tensor = torch.stack(
		[model(torch.unsqueeze(torch.squeeze(tensor[:, t, :, :]), dim=0)) for t in range(T)], dim=2
	)
	# Squeezing results in a (D, T) tensor.
	encoded_tensor = torch.squeeze(encoded_tensor, dim=0)
	print(encoded_tensor.shape)
	
	location = os.path.join(video_dir, encoding_filename)
	logging.info('Pickling an encoded tensor of shape {0} to {1}.'.format(encoded_tensor.shape, location))
	torch.save(encoded_tensor, location)

def get_video_dirs(label_dir, data_type):
	# return a list of paths for the images
	prefix = None
	if data_type == 'RGB':
		prefix = 'M_'
	elif data_type == 'RGBD':
		prefix = 'K_'
	else:
		raise ValueError('Data type for pickling is invalid')

	return glob.glob(os.path.join(label_dir, '{0}*/'.format(prefix)))