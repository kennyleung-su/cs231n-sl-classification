"""Runs the gesture classification model, either in train or testing mode
according to config.py."""

import logging
import numpy as np

from configs import config
from configs.config import MODEL_CONFIG
from utils import data_loader
from models import model, dev_models
from trainer import train_utils


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
		model.train()

		# Make a dummy dataset of scalars, for now.
		num_samples = 100
		num_classes = 10
		sample_dim = 5
		data = {
			'label': np.random.randint(low=0, high=num_classes, size=num_samples),
			'frames': np.random.rand(num_samples, sample_dim)
		}
		loss_fn = None
		optimizer = None
		train_utils.train_model(X=data['frames'],
								y=data['label'],
								model=model,
								epochs=MODEL_CONFIG.epochs,
								loss_fn=loss_fn,
								optimizer=optimizer,
								lr=MODEL_CONFIG.learning_rate)

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
