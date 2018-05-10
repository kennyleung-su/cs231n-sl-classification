"""Runs the gesture classification model, either in train or testing mode
according to config.py."""

from configs import config
from configs.config import MODEL_CONFIG
from utils import data_loader
from models import model, dev_models
from trainer import train_utils

import logging

def main():
	logging.info('Running experiment <{0}> in {1} mode.\n'
		'Description of model: {2}'.format(MODEL_CONFIG.name,
			MODEL_CONFIG.mode, MODEL_CONFIG.description))

	# Convert the gesture video frames into data samples inside of a DataLoader.
	# TODO: Support pickling to speed up the process. Perhaps we can hash the
    # list of gesture labels to a checksum and check if a file with that name exists.
	dataloader = data_loader.GenerateGestureFramesDataLoader(MODEL_CONFIG.gesture_labels,
		config.TRAIN_DATA_DIR, MODEL_CONFIG.max_frames_per_sample)

	# Initialize the model, or load a pretrained one.
	if MODEL_CONFIG.checkpoint_to_load:
		model = torch.load(MODEL_CONFIG.checkpoint_to_load)
	elif MODEL_CONFIG.mode == 'test':
		raise ValueError('Testing the model requires a --checkpoint_to_load argument.')
	else:
		model = dev_models.DummyModel(model_config=MODEL_CONFIG)

	# Train the model.
	if MODEL_CONFIG.mode == 'train':
		X = None
		loss_fn = None
		optimizer = None
		train_utils.train_model(X,
								model,
								MODEL_CONFIG.epochs,
								loss_fn,
								optimizer,
								MODEL_CONFIG.learning_rate)

	# Run the model on the test set, using the dataloader.

	# Save (and maybe visualize or analyze?) the results.
	# Use:
	# 	- config.TRAIN_DIR for aggregate training/validation results
	# 	- config.TEST_DIR for aggregate testing results
	# 	- config.MODEL_DIR for general model information

	# Save the final model to a checkpoint.
	model.save_to_checkpoint(MODEL_CONFIG.checkpoint_path)


if __name__ == '__main__':
	main()
