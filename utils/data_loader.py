# TODO: Implement data loading and preprocessing.

import csv
import glob
import imageio
import logging
import numpy as np
import pandas as pd
import os
import re
import torch

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # for visualizations

class GestureFramesDataset(Dataset):
    """Dataset of tensors corresponding to gesture video.

    Args:
        data_dir: str, path to the directory with folders of videos and frames
        transform: list of callable Classes, e.g. from torchvision.transforms,
            with which the video frames should be preprocessed
    """
    def __init__(self, gesture_labels, data_dir, max_frames_per_sample, transform=None):
        # TODO: Clarify the shape of self.data.
        self.data = self.populate_gesture_frames_data(self, data_dir, gesture_labels,
            max_frames_per_sample)
        self.len = len(self.data)
        logging.info('Initialized a GestureFramesDataset of size {0}.', self.len)
        super(GestureFramesDataset, self).__init__()

    def __getitem__(self, gesture_label):
        # TODO(kenny): Figure out how to sample with balanced labels.
        return self.data[gesture_label]

    def __len__(self):
        return self.len


    @staticmethod
    def read_frame_tensors_from_dir(directory):
        filenames = glob.glob("{0}/*.png".format(directory))
        matches = [re.match('.*_(\d+)\.png', name) for name in filenames]
        frames = sorted([(int(match.group(1)), match.group(0)) for match in matches])  # sorted list of (frame_number, frame_path) tuples
        sorted_filenames = [f[1] for f in frames] 
        frame_arrays = []
        for frame_file in sorted_filenames:
            frame_arrays.append(imageio.imread(frame_file))
        return np.stack(frame_arrays)


    @staticmethod
    def populate_gesture_frames_data(self, data_dir, gesture_labels, max_frames_per_sample, type_data="kinect"):
        """Returns a list of ...

        Example usage: gestures(gesture_list=[5,19,38,37,8,29,213,241,18,92],
                                textfile = 'train_list_2.txt',
                                type_data = "kinect")
        TODO(kenny): Incorporate the following method into GestureFramesDataset.
        """
        logging.info('Populating frame tensors for {0} specified labels in data dir {1}: {2}'.format(
            len(gesture_labels), data_dir, gesture_labels))
        labels_file = os.path.join(data_dir, '{0}_list.txt'.format(data_dir.split('/')[-1]))
        data = pd.read_csv(labels_file, sep=" ", header=None)
        data.columns = ["rgb", "kinect", "label"]
        label_to_dirs = {}
        for label in gesture_labels:
            directory_labels = data.loc[data['label'] == label][type_data].tolist()
            # strip .avi from the end of the filename
            directories = [''.join(label.split('.avi')[:-1]) for label in directory_labels]
            label_to_dirs[label] = directories
        data = []
        for label, directories in label_to_dirs.items():
            for directory in directories:
                data.append({
                    'frames': self.read_frame_tensors_from_dir(os.path.join(data_dir, directory)),
                    'label': label,
                    'directory': directory
                })
	
        ### Testing ###
        # TODO: Delete debugging code.
        logging.debug('Writing out test_image.png for the first data frame.')
        imageio.imwrite('test_image.png', data[0]['frames'][0])
        return data


def GenerateGestureFramesDataLoader(gesture_labels, data_dir, max_frames_per_sample):
    """Returns a configured DataLoader instance."""

    # Build a gesture frames dataset using the configuration information.
    # This is just dummy code to be replaced.
    transformed_dataset = GestureFramesDataset(gesture_labels, data_dir, max_frames_per_sample)
    return DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
