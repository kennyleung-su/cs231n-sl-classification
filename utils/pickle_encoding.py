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
	use_cuda = model_config.use_cuda
	overwrite = model_config.pickle_overwrite
	max_example_per_label = model_config.max_example_per_label
	transform = model_config.transform
	gesture_labels = model_config.gesture_labels
	data_type, _ = model_config.dataloader_type.split('-')
	transfer_learning_prefix = 'T' if model_config.load else ''
	encoding_filename = '{0}RN{1}{2}-encoding.pkl'.format(transfer_learning_prefix ,str(model_config.resnet_num_layers),
												  data_type)
	# encoding with the prefix T means transfer learning has been carried out. else, it takes the output of 
	# the original imagenet pretrained resnet
	if overwrite:
		logging.info('Pickle will now overwrite all the existing encodings.')

	if max_example_per_label:
		logging.info('Max example per label: {}'.format(max_example_per_label))

	for data_dir in data_dirs:
		for label in gesture_labels:
			label_dir = os.path.join(data_dir, str(label))
			video_dirs = get_video_dirs(label_dir, data_type)

			if max_example_per_label:
				video_dirs = video_dirs[:max_example_per_label]

			FIXED_BATCH_SIZE = 20
			num_batches = (len(video_dirs) + (FIXED_BATCH_SIZE - 1)) // FIXED_BATCH_SIZE
			for batch_idx in range(num_batches):
				segment_lengths = []
				video_paths = []

				# Elements consist of (T, C, H, W) tensor.
				batch_video_tensors = []

				start_idx = batch_idx * FIXED_BATCH_SIZE
				end_idx = min(len(video_dirs), start_idx + FIXED_BATCH_SIZE)
				for video_dir in video_dirs[start_idx : end_idx]:
					logging.info('Parsing video directory: {0}'.format(video_dir))
					video_path = os.path.join(video_dir, encoding_filename)
					if not overwrite and os.path.exists(video_path):
						continue
					if overwrite and os.path.exists(video_path):
						# remove the file before overwriting
						os.remove(video_path)

					video_tensor = get_video_tensor_for_dir(video_dir, transform)
					segment_lengths.append(video_tensor.shape[0])
					video_paths.append(video_path)
					batch_video_tensors.append(video_tensor)

				if not batch_video_tensors:
					# No videos to pickle, continue on with the next batch.
					continue

				# Expect the output to be (N, D) where N = sum of all time lengths of all videos.
				input_tensor = torch.cat(batch_video_tensors, dim=0)

				if use_cuda:
					logging.info('Using cuda for pickling label {}'.format(label))
					input_tensor.cuda()

				encodings = model(input_tensor)
				video_start = 0
				for video_len, video_path in zip(segment_lengths, video_paths):
					logging.info('Saving encodings for {0} of shape: {1}'.format(video_path, (encodings[video_start : video_start+video_len, :]).shape))
					torch.save(torch.t(encodings[video_start : video_start+video_len, :]), video_path)
					video_start += video_len


def get_video_tensor_for_dir(video_dir, transform):
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
		frame_ndarray = transform(frame_ndarray)
		frames_list.append(frame_ndarray)
	# Stacks up to a (C, T, H, W) tensor.
	tensor = torch.stack(frames_list, dim=1)
	C, T, H, W = tensor.shape

	# (C, T, H, W) => (T, C, H, W)
	tensor = torch.transpose(tensor, 0, 1)
	return tensor


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