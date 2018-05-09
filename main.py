"""Runs the gesture classification model, either in train or testing mode
according to config.py."""

from configs import config
from configs.config import MODEL_CONFIG
from utils import data_loader
from models import model
import logging

def main():
	logging.info('Running experiment <{0}> in {1} mode.\n'
		'Description of model: {2}'.format(MODEL_CONFIG.name,
			MODEL_CONFIG.mode, MODEL_CONFIG.description))

	# TODO(everyone): Implement the rest! Use the values stored
	# in config to initialize our DataLoader, Model, training/testing modules.
	# Please log whenever possible! We can fine-tune the the logger down the
	# road but for now as we are experimenting, verbosity is good.
	dataloader = data_loader.GenerateGestureFramesDataLoader(MODEL_CONFIG.gesture_labels,
		config.TRAIN_DATA_DIR, MODEL_CONFIG.max_frames_per_sample)
	# Initialize the dataloader.
	# Use: 
	# 	- config.{TEST/TRAIN/VALID}_DIR
	# 	- MODEL_CONFIG.gesture_labels
	#	- MODEL_CONFIG.max_frames_per_sample
	# 	- etc

	# Initialize the model, using an existing checkpoint if applicable.
	# Use:
	#	- MODEL_CONFIG.checkpoint_to_load
	# 	- MODEL_CONFIG.pretrained_cnn_model
	# 	- MODEL_CONFIG.normalize
	# 	- MODEL_CONFIG.minibatch_size
	# 	- MODEL_CONFIG.lstm_hidden_size
	# 	- MODEL_CONFIG.initializer_fn

	# If in training mode:
	# 	Initialize and run the optimizer/trainer.
	# 	Use:
	# 		- MODEL_CONFIG.mode
	# 		- MODEL_CONFIG.learning_rate
	# 		- MODEL_CONFIG.optimizer_fn
	#		- MODEL_CONFIG.epochs

	# Run the model on the test set, using the dataloader.

	# Save (and maybe visualize or analyze?) the results.
	# Use:
	# 	- config.TRAIN_DIR for aggregate training/validation results
	# 	- config.TEST_DIR for aggregate testing results
	# 	- config.MODEL_DIR for general model information

	# Save model to a new checkpoint.
	# Use:
	# 	- config.checkpoint


if __name__ == '__main__':
	main()
