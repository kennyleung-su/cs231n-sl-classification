"""Runs the gesture classification model, either in train or testing mode
according to config.py."""

from configs import config
from utils import data_loader
from models import model
import logging

def main():
	logging.info('Running experiment <{0}> in {1} mode.\n'
		'Description of model: {2}'.format(config.experiment_name,
			config.mode, config.experiment_description))

	# TODO(everyone): Implement the rest! Use the values stored
	# in config to initialize our DataLoader, Model, training/testing modules.
	# Please log whenever possible! We can fine-tune the the logger down the
	# road but for now as we are experimenting, verbosity is good.
	print(config.gesture_test)
	print(config.TRAIN_DATA_DIR)
	print(config.max_frames_per_sample)

	dataloader = data_loader.GenerateGestureFramesDataLoader(config.gesture_test,
		config.VALID_DATA_DIR, config.max_frames_per_sample)
	# Initialize the dataloader.
	# Use: 
	# 	- config.{TEST/TRAIN/VALID}_DIR
	# 	- config.gesture_labels
	#	- config.max_frames_per_sample
	# 	- etc

	# Initialize the model, using an existing checkpoint if applicable.
	# Use:
	#	- config.checkpoint_to_load
	# 	- config.pretrained_cnn_model
	# 	- config.normalize
	# 	- config.minibatch_size
	# 	- config.lstm_hidden_size
	# 	- config.initializer_fn

	# If in training mode:
	# 	Initialize and run the optimizer/trainer.
	# 	Use:
	# 		- config.mode
	# 		- config.learning_rate
	# 		- config.optimizer_fn
	#		- config.epochs

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
