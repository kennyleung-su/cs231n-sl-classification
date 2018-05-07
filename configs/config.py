import os
import logging
import time

import torch
import torchvision.models as models
import torchvision.transforms as transforms

##############
# Directories
##############

def mkdir(path):
	"""Recursive mkdir, without overwriting existing directories."""
	parent, child = os.path.split(path)
	if not os.path.exists(parent):
		mkdir(parent)

	if not os.path.exists(path):
		os.mkdir(path)

# Major directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))	# /home/shared
PROJECT_DIR = os.path.join(ROOT_DIR, 'cs231n-sl-classification')		# cs231n-sl-classification

EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, 'experiments')
MODEL_DIR = os.path.join(EXPERIMENTS_DIR, 'models')		# cs231n-sl-classification/experiments/models/{checkpoints, etc}
LOG_DIR = os.path.join(EXPERIMENTS_DIR, 'logs')			# cs231n-sl-classification/experiments/logs/
TRAIN_DIR = os.path.join(EXPERIMENTS_DIR, 'train')		# cs231n-sl-classification/experiments/models/train/{logs, plots, etc}
TEST_DIR = os.path.join(EXPERIMENTS_DIR, 'test')		# cs231n-sl-classification/experiments/models/test/{logs, plots, etc}

TEST_DATA_DIR = os.path.join(ROOT_DIR, 'test')
TRAIN_DATA_DIR = os.path.join(ROOT_DIR, 'train')
VALID_DATA_DIR = os.path.join(ROOT_DIR, 'valid')

#########################
# Model Saving & Loading
#########################
experiment_name = 'basic_resnet_lstm_model'
experiment_description = """
Applies a pretrained ConvNet architecture to a small sample of
training data (from 10 gesture classes) and uses an LSTM network
to process these frame embeddings as input to a softmax classifier.
"""

# Either 'train' or 'test'.
mode = 'train'

# File to seralize our PyTorch model to. Existing checkpoint will be overwritten.
# https://pytorch.org/docs/stable/torch.html#torch.save
# Used for mode = 'training'.
checkpoint = os.path.join(MODEL_DIR, '{0}-checkpoint.pkl'.format(experiment_name))

# If this set to a file, then we will use that to initialize our model.
# Can be used for all modes; required for testing.
checkpoint_to_load = None

# File in which to log output.
# Just import logging, config in a library to log to the same location.
logfile = os.path.join(LOG_DIR, '{0}-{1}-info.txt'.format(experiment_name, time.time()))

# How many training iterations after which to log training and validation results.
log_interval = 100

########################
# Dataset Configuration
########################

gesture_labels = [
	5,		# peace sign (851 samples)
	19,		# 'F' (492 samples)
	38,		# number 1 (459 samples)
	37,		# number 5 (377 samples)
	8,		# Index pointing to head (375 samples)
	29,		# Thumbs down (285 samples)
	213, 	# Timeout (257 samples)
	241, 	# Circular motion (256 samples)
	18, 	# "C" (349 samples)
	92 		# Thumbs up (349 samples)
]

gesture_test = [18]

# Threshold for how many frames a video is allowed to have.
max_frames_per_sample = 75

# TODO: Add a configuration for the sampling scheme.

########################
# Model Hyperparameters
########################

# https://pytorch.org/docs/master/torchvision/models.html
# 
# All pre-trained models expect input images normalized in the same way,
# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where
# H and W are expected to be at least 224. The images have to be loaded
# in to a range of [0, 1] and then normalized using
pretrained_cnn_model = models.resnet18
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

minibatch_size = 50
lstm_hidden_size = 256
learning_rate = 1e-2
epochs = 1e3
optimizer_fn = torch.optim.SGD
initializer_fn = torch.nn.init.xavier_normal_

################
# Miscellaneous
################

disabled_cuda = False
seed = 1

##################
# Directory setup
##################

# Creates all prerequisite directories for our model.
mkdir(MODEL_DIR)
mkdir(LOG_DIR)

train_output_dir = os.path.join(TRAIN_DIR, experiment_name)
mkdir(train_output_dir)

test_output_dir = os.path.join(TEST_DIR, experiment_name)
mkdir(test_output_dir)

# Set up logging.
try:
    file = open(logfile, 'r')
except IOError:
    file = open(logfile, 'w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(message)s')

# Set up a streaming logger.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

fh = logging.FileHandler(logfile)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
