"""Runs the gesture classification model, either in train or testing mode
according to config.py."""

import logging
import numpy as np
import random

from configs import config
from configs.config import MODEL_CONFIG
from utils import data_loader
from models.basic import PretrainedConvLSTMClassifier
from models.debug import RandomClassifier, LinearClassifier
from trainer import train_utils, validate_utils

import torch
import torch.nn.functional as F
import torch.optim as optim


__EXP_MODELS__ = {
	'debug': LinearClassifier,
	'basic': PretrainedConvLSTMClassifier,
	'basic_dev': PretrainedConvLSTMClassifier
}

DATA_DIRS = [config.TRAIN_DATA_DIR, config.VALID_DATA_DIR, config.TEST_DATA_DIR]


def main():
	logging.info('Running experiment <{0}> in {1} mode.\n'
		'Description of model: {2}'.format(MODEL_CONFIG.name,
			MODEL_CONFIG.mode, MODEL_CONFIG.description))

	#(train_dataloader,) = data_loader.GetGestureFramesDataLoaders([config.TRAIN_DATA_DIR], MODEL_CONFIG)

	(train_dataloader, valid_dataloader,
		test_dataloader) = data_loader.GetGestureFramesDataLoaders(DATA_DIRS, MODEL_CONFIG)

	# Initialize the model, or load a pretrained one.
	model = __EXP_MODELS__[MODEL_CONFIG.experiment](model_config=MODEL_CONFIG)
	# Set up the loss function
	loss_fn = torch.nn.CrossEntropyLoss()
	# Random seeds
	random.seed(MODEL_CONFIG.seed)

	# activate cuda if available and enabled
	if MODEL_CONFIG.use_cuda:
		if torch.cuda.is_available():
			logging.info('Running the model using GPUs. (--use_cuda)')
			# passing model into DataParallel allows for data to be
			# computed in parallel through all the available GPUs
			model = torch.nn.DataParallel(model)
			model.cuda()
			loss_fn.cuda()
			# set cuda seeds
			torch.cuda.manual_seed_all(MODEL_CONFIG.seed)
		else:
			logging.info('Sorry, no GPUs are available. Running on CPU.')
			# toggle use_cuda off
			MODEL_CONFIG.use_cuda = False

	if MODEL_CONFIG.checkpoint_to_load:
		# The checkpoint is a dictionary consisting of keys 'epoch', 'state_dict'
		checkpoint = torch.load(MODEL_CONFIG.checkpoint_to_load)
		model.load_state_dict(checkpoint.get('state_dict'))
		checkpoint_epoch = checkpoint.get('epoch')
	elif MODEL_CONFIG.mode == 'test':
		raise ValueError('Testing the model requires a --checkpoint_to_load argument.')

	# Train the model.
	if MODEL_CONFIG.mode == 'train':
		logging.info("Model will now begin training.")
		if not MODEL_CONFIG.checkpoint_to_load:
			checkpoint_epoch = 0
		for epoch in range(checkpoint_epoch, MODEL_CONFIG.epochs + checkpoint_epoch):
			train_utils.train_model(model=model,
									dataloader=train_dataloader,
									epochs=MODEL_CONFIG.epochs,
									loss_fn=loss_fn,
									optimizer=optim.SGD(
											filter(
												lambda p: p.requires_grad,
												model.parameters()),
										lr=MODEL_CONFIG.learning_rate,
										momentum=0.9),
									epoch=epoch,
									use_cuda=MODEL_CONFIG.use_cuda)
			if MODEL_CONFIG.use_cuda:
				model.module.training_epoch += 1
			else:
				model.training_epoch += 1

			# TODO: Run on the validation set, with frequency depending on
			# on the configuration.

			# TODO: Pickle the best seen model. Use model._best_accuracy
			# (change to model.best_val_accuracy?) to determine if this is the case.

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
