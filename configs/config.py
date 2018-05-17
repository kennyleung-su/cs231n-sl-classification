import argparse
import logging
import os
import time

from collections import namedtuple
from configobj import ConfigObj
from validate import Validator

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.models as models

parser = argparse.ArgumentParser(description='Gesture classification task.')

parser.add_argument('--experiment', type=str, default='basic',
	help='Name of the experiment. Defaults to basic.')
parser.add_argument('--mode', type=str, default='train',
	help='Running mode: "train" or "test".')

parser.add_argument('--max_example_per_label', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lstm_hidden_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--max_seq_len', type=int,
					help='Maximum temporal depth of video frames on which to train.')

parser.add_argument('--debug', action='store_true')
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=0,
					help='Number of separate processes with which to run the DataLoader. '
					'Set to a value, e.g. 4, when running on a VM with high compute.')
parser.add_argument('--checkpoint_to_load', type=str)

# Include this flag to overwrite pickled frames anew.
parser.add_argument('--repickle_frames', action='store_true')

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet18)')
args = parser.parse_args()


class ConfigObjFromDict(object):
	"""Handy class for creating an object updated as a dict but accessed as an obj."""
	def __init__(self, **entries):
		self.__dict__.update(entries)

	def __setattr__(self, name, value):
		self.__dict__[name] = value

	def __getattr__(self, name):
		return self.__dict__.get(name, None)

	def __str__(self):
		return ' '.join(['{0}: {1}\n'.format(k, v) for k, v in self.__dict__.items()])


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

CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, 'experiments')
MODEL_DIR = os.path.join(EXPERIMENTS_DIR, 'models')		# cs231n-sl-classification/experiments/models/{checkpoints, etc}
LOG_DIR = os.path.join(EXPERIMENTS_DIR, 'logs')			# cs231n-sl-classification/experiments/logs/
TRAIN_DIR = os.path.join(EXPERIMENTS_DIR, 'train')		# cs231n-sl-classification/experiments/models/train/{logs, plots, etc}
TEST_DIR = os.path.join(EXPERIMENTS_DIR, 'test')		# cs231n-sl-classification/experiments/models/test/{logs, plots, etc}

TEST_DATA_DIR = os.path.join(ROOT_DIR, 'test')
TRAIN_DATA_DIR = os.path.join(ROOT_DIR, 'train')
VALID_DATA_DIR = os.path.join(ROOT_DIR, 'valid')

#################
# Configurations
#################

# Read the flag and configuration file values and validate them to cast them to their
# specified data types.
exp_config = ConfigObj(os.path.join(CONFIGS_DIR, '{0}.ini'.format(args.experiment)),
	configspec=os.path.join(CONFIGS_DIR, 'configspec.ini'))
exp_config.validate(Validator())

# Merge the flag and configuration file values, overwriting exp_config values with flag values.
for k, v in vars(args).items():
	if v:
		exp_config[k] = v
MODEL_CONFIG = ConfigObjFromDict(**exp_config)

#########################
# Model Saving & Loading
#########################

# File to seralize our PyTorch model to. Existing checkpoint will be overwritten.
# https://pytorch.org/docs/stable/torch.html#torch.save
# Used for mode = 'training'.
MODEL_CONFIG.checkpoint_path = os.path.join(MODEL_DIR, '{0}-checkpoint.pkl'.format(args.experiment))

# File in which to log output.
# Just import logging, config in a library to log to the same location.
logfile = os.path.join(LOG_DIR, '{0}-{1}-info.txt'.format(args.experiment, time.time()))

########################
# Dataset Configuration
########################

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
MODEL_CONFIG.pretrained_cnn_model = models.__dict__[args.arch]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
MODEL_CONFIG.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])

MODEL_CONFIG.optimizer_fn = torch.optim.SGD
MODEL_CONFIG.initializer_fn = torch.nn.init.xavier_normal_

################
# Miscellaneous
################

MODEL_CONFIG.seed = 1

if args.debug:
	MODEL_CONFIG.pretrained_cnn_model = None  # do not use pretrained CNNs
	MODEL_CONFIG.experiment = 'debug'
	MODEL_CONFIG.gesture_labels = [1, 2, 3]

# TODO: Complete with other model types.
# if MODEL_CONFIG.experiment == 'debug':
# 	MODEL_CONFIG.model = dev_models.DummyModel

##################
# Directory setup
##################

# Creates all prerequisite directories for our model.
mkdir(MODEL_DIR)
mkdir(LOG_DIR)

train_output_dir = os.path.join(TRAIN_DIR, args.experiment)
mkdir(train_output_dir)

test_output_dir = os.path.join(TEST_DIR, args.experiment)
mkdir(test_output_dir)

# Set up logging.
try:
	file = open(logfile, 'r')
except IOError:
	file = open(logfile, 'w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')

# Set up a streaming logger.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

fh = logging.FileHandler(logfile)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
