"""Runs the gesture classification model, either in train or testing mode
according to config.py."""

import logging
import numpy as np

from configs import config
from configs.config import MODEL_CONFIG
from utils import data_loader
from models.basic import PretrainedConvLSTMClassifier
from models.debug import RandomClassifier, LinearClassifier
from trainer import train_utils

import torch
import torch.nn.functional as F
import torch.optim as optim


def main():
	logging.info('Running experiment <{0}> in {1} mode.\n'
		'Description of model: {2}'.format(MODEL_CONFIG.name,
			MODEL_CONFIG.mode, MODEL_CONFIG.description))

	# Convert the gesture video frames into data samples inside of a DataLoader.
	# TODO: Support pickling to speed up the process. Perhaps we can hash the
	# list of gesture labels to a checksum and check if a file with that name exists.
	train_dataloader = data_loader.GenerateGestureFramesDataLoader(MODEL_CONFIG.gesture_labels,
		config.TRAIN_DATA_DIR, MODEL_CONFIG.max_frames_per_sample, MODEL_CONFIG.batch_size)

	# Initialize the model, or load a pretrained one.
	if MODEL_CONFIG.checkpoint_to_load:
		model = torch.load(MODEL_CONFIG.checkpoint_to_load)
	elif MODEL_CONFIG.mode == 'test':
		raise ValueError('Testing the model requires a --checkpoint_to_load argument.')
	else:
		model = LinearClassifier(model_config=MODEL_CONFIG)

	# Train the model.
	if MODEL_CONFIG.mode == 'train':
		model.train()

		# TODO: Extract out to train_utils.
		for epoch in range(1, MODEL_CONFIG.epochs + 1):
			train_utils.train_model(model=model,
									dataloader=train_dataloader,
									epochs=MODEL_CONFIG.epochs,
									loss_fn=F.nll_loss,
									optimizer=optim.SGD(model.parameters(),
														lr=MODEL_CONFIG.learning_rate,
														momentum=0.9),
									epoch=epoch)

	# Run the model on the test set, using a new test dataloader.

	# Save (and maybe visualize or analyze?) the results.
	# Use:
	# 	- config.TRAIN_DIR for aggregate training/validation results
	# 	- config.TEST_DIR for aggregate testing results
	# 	- config.MODEL_DIR for general model information

	# Save the final model to a checkpoint.
	model.save_to_checkpoint(MODEL_CONFIG.checkpoint_path)


if __name__ == '__main__':
	main()
