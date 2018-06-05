"""Performs optical flow on the K videoframes within a directory to optical flow images.

Stores the optical flow computed from each pair of time frames as a new image: OF_*.png.
"""
import cv2
import glob
import numpy as np
import os

folders = ['train', 'valid', 'test']
prefices = ['M_', 'K_']

ROOT_DIR, _ = os.path.split(os.getcwd())
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')


def process_rgb_optical_flow(data_dir, prefix='M_'):
	"""Processes optical flow frames from pairwise RGB video frames in the data directory."""
	frames = np.array(glob.glob(os.path.join(data_dir, '{0}*.png'.format(prefix))))
	indices = [int(f.split('/')[-1].rstrip('.png').split('_')[-1]) for f in frames]
	sorted_indices = np.argsort(indices)
	sorted_frames = frames[sorted_indices]
	
	# Feed each adjacent pair of frames into the dense optical flow model.
	print('Saving optical flow images for frames in directory: {0}'.format(data_dir))
	save_optical_flow(sorted_frames, data_dir)


def save_optical_flow(frames, output_dir):
	"""Computes and saves optical flow frames in the output directory."""
	frame1_path = frames[0]
	prev_frame = cv2.cvtColor(cv2.imread(frame1_path), cv2.COLOR_RGB2GRAY)
	for i in range(1, len(frames)):
		# Hue, saturation, value (3-channel representation of the optical flow vector)
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
		output_file = os.path.join(output_dir, 'OF_{0}.png'.format(i))
		if not os.path.isfile(output_file):
			cv2.imwrite(output_file, rgb_flow)
		prev_frame = next_frame


def main():
	for folder in folders:
		for prefix in prefices:
			label_dirs = glob.glob('{0}/*/{1}*'.format(os.path.join(DATASET_DIR, folder), prefix))
			for label_dir in label_dirs:
				process_rgb_optical_flow(label_dir, prefix=prefix)


if __name__ == '__main__':
	main()
