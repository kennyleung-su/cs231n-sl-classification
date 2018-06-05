"""Performs optical flow on the K videoframes within a directory to optical flow images.

Stores the optical flow computed from each pair of time frames as a new image: OF_*.png.
"""
import argparse
import cv2
import glob
import logging
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
import numpy as np
import os
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Preprocesses frames using optical flow..')

parser.add_argument('--prefix', type=str, default='K', help='Either K or M".')
parser.add_argument('--num_workers', type=int, default=16,
		help='Number of separate processes with which to run the DataLoader. '
		'Set to a value, e.g. 4, when running on a VM with high compute.')
parser.add_argument('--clear_all', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--stride', type=int, default=1, help='Optical flow processing stride')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no_save', action='store_true')
args = parser.parse_args()

folders = ['train', 'valid', 'test']
prefices = ['M_', 'K_']

ROOT_DIR, _ = os.path.split(os.getcwd())
LOG_DIR = os.path.join(ROOT_DIR, 'dataset/log')
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')

##########
# Logging
##########
# File in which to log output.
# Just import logging, config in a library to log to the same location.
logfile = os.path.join(LOG_DIR, 'of_preprocess.{0}.{1}.INFO'.format(args.prefix, time.time())) 
if not os.path.exists(LOG_DIR):
	os.mkdir(LOG_DIR)

try:
	f = open(logfile, 'r')
except IOError:
	f = open(logfile, 'w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
fh_info = logging.FileHandler(logfile)
fh_info.setLevel(logging.INFO)
fh_info.setFormatter(formatter)

if args.verbose:	
	# Set up a streaming logger.
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)

logger.addHandler(fh_info)


class Counter(object):
	def __init__(self):
		self.val = multiprocessing.Value('i', 0)
		self.pbar = tqdm()

	def __del__(self):
		self.pbar.close()

	def increment(self, n=1):
		with self.val.get_lock():
			self.val.value += n
			self.pbar.update(1)

	@property
	def value(self):
		return self.val.value


def process_rgb_optical_flow(data_dir, stride, prefix='M_', counter=None, overwrite=False, clear_all=False):
	"""Processes optical flow frames from pairwise RGB video frames in the data directory."""
	frames = np.array(glob.glob(os.path.join(data_dir, '{0}*.png'.format(prefix))))
	indices = [int(f.split('/')[-1].rstrip('.png').split('_')[-1]) for f in frames]
	sorted_indices = np.argsort(indices)
	sorted_frames = frames[sorted_indices]
	# Delete existing optical flow images if applicable.
	if overwrite:
		for f in glob.glob(os.path.join(data_dir, 'OF_*.png' if clear_all else 'OF_*stride{0}.png'.format(stride))):
			logging.info('Removing file: {0}'.format(f))
			os.remove(f)
	# Feed each adjacent pair of frames into the dense optical flow model.
	if not args.no_save:
		save_optical_flow(sorted_frames, data_dir, stride, counter, overwrite)


def save_optical_flow(frames, output_dir, stride, counter=None, overwrite=False):
	"""Computes and saves optical flow frames in the output directory."""
	frame1_path = frames[0]
	prev_frame = cv2.cvtColor(cv2.imread(frame1_path), cv2.COLOR_RGB2GRAY)
	for i in range(stride, len(frames), stride):
		# Hue, saturation, value (3-channel representation of the optical flow vector)

		output_file = os.path.join(output_dir, 'OF_{0}_stride{1}.png'.format(i, stride))
		if os.path.isfile(output_file) and not overwrite:
			continue

		hsv = np.zeros((*prev_frame.shape[:2], 3))
		next_frame = cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_RGB2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

		# Convert cartesian vectors in flow into polar coordinates
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		# Hue = direction, value = magnitude.
		hsv[...,1] = cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_RGB2HSV)[..., 1]
		hsv[...,0] = ang * (180/ np.pi / 2)
		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		# Convert to int32 values.
		hsv = np.asarray(hsv, dtype= np.float32)
		rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

		# Save to file if it doesn't already exist
		cv2.imwrite(output_file, rgb_flow)
		logging.info('Saved optical flow file: {0}'.format(output_file))
		prev_frame = next_frame
		if counter:
			counter.increment()


def main():
	pool = ThreadPool(args.num_workers)
	results = []
	counter = Counter()  # max
	for folder in folders:
		# print('Saving optical flow images for frames in folder: {0}'.format(folder))
		label_dirs = glob.glob('{0}/*/{1}*'.format(os.path.join(DATASET_DIR, folder), args.prefix))
		label_dirs = [x for x in label_dirs if os.path.isdir(x)]
		# print('Saving optical flow images for frames with prefix {0} in folder: {1}'.format(
		# args.prefix, os.path.join(DATASET_DIR, folder)))
		for label_dir in label_dirs:
			pool.apply_async(process_rgb_optical_flow,
				(label_dir, args.stride, args.prefix, counter, args.overwrite, args.clear_all))
	pool.close()
	pool.join()


if __name__ == '__main__':
	main()
