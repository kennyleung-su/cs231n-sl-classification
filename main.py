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


__EXP_MODELS__ = {
	'debug': LinearClassifier,
	'basic': PretrainedConvLSTMClassifier
}

DATA_DIRS = [config.TRAIN_DATA_DIR, config.VALID_DATA_DIR, config.TEST_DATA_DIR]


def main():
	logging.info('Running experiment <{0}> in {1} mode.\n'
		'Description of model: {2}'.format(MODEL_CONFIG.name,
			MODEL_CONFIG.mode, MODEL_CONFIG.description))

	(train_dataloader,) = data_loader.GetGestureFramesDataLoaders([config.TRAIN_DATA_DIR], MODEL_CONFIG)

	# (train_dataloader, valid_dataloader,
	# 	test_dataloader) = data_loader.GetGestureFramesDataLoaders(DATA_DIRS, MODEL_CONFIG)

	# Initialize the model, or load a pretrained one.
	model = __EXP_MODELS__[MODEL_CONFIG.experiment](model_config=MODEL_CONFIG)
	if torch.cuda.is_available() and MODEL_CONFIG.use_cuda:
		model.cuda()

	if MODEL_CONFIG.checkpoint_to_load:
		# The checkpoint is a dictionary consisting of keys 'epoch', 'state_dict'
		checkpoint = torch.load(MODEL_CONFIG.checkpoint_to_load)
		model.load_state_dict(checkpoint.get('state_dict'))
		checkpoint_epoch = checkpoint.get('epoch')
	elif MODEL_CONFIG.mode == 'test':
		raise ValueError('Testing the model requires a --checkpoint_to_load argument.')

	# Train the model.
	if MODEL_CONFIG.mode == 'train':
		model.train()
		# TODO: Extract out to train_utils.
		if not MODEL_CONFIG.checkpoint_to_load:
			checkpoint_epoch = 0
		for epoch in range(checkpoint_epoch, MODEL_CONFIG.epochs + checkpoint_epoch):
			train_utils.train_model(model=model,
									dataloader=train_dataloader,
									epochs=MODEL_CONFIG.epochs,
									loss_fn=F.nll_loss,
									optimizer=optim.SGD(
											filter(
												lambda p: p.requires_grad,
												model.parameters()),
										lr=MODEL_CONFIG.learning_rate,
										momentum=0.9),
									epoch=epoch)
			model.training_epoch += 1

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
