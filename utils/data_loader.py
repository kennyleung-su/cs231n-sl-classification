# TODO: Implement data loading and preprocessing.

import csv
import glob
import imageio
import numpy as np
import pandas as pd
import os
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
        # TODO(kenny): Read in and transform the video frames from the
        # given information into 4D frame tensors, keeping track of the
        # classification label along the way.

        # self.data {
        #     'label': int
        #     'frames': np.ndarray of shape (frames, 3 (RGB), width, height)
        # }
        self.data = self.populate_gesture_frames_data(self, data_dir, gesture_labels,
            max_frames_per_sample)
        self.len = len(self.data)
        super(self).__init__()

    def __getitem__(self, gesture_label):
        # TODO(kenny): Figure out how to sample randomly while keeping the
        # proportion of labels uniform, despite having a varying number of
        # videos per label.
        return self.data[gesture_label]

    def __len__(self):
        return self.len


    @staticmethod
    def read_frame_tensors_from_dir(directory):
        print(directory)
        filenames = glob.glob("{0}/*.txt".format(directory))
        print(filenames)
        frames = [re.match('.*_(\d+)\.avi', name) for name in filenames]
        print(frames)
        sorted_filenames = filenames[np.argsort(frames)]
        frame_arrays = []
        for f in sorted_filenames:
            frame_arrays.append(imageio.imread(f))
        return np.vstack(frame_arrays)


    @staticmethod
    def populate_gesture_frames_data(self, data_dir, gesture_labels, max_frames_per_sample, type_data="kinect"):
        """Returns a list of ...

        Example usage: gestures(gesture_list=[5,19,38,37,8,29,213,241,18,92],
                                textfile = 'train_list_2.txt',
                                type_data = "kinect")
        TODO(kenny): Incorporate the following method into GestureFramesDataset.
        """
        labels_file = os.path.join(data_dir, '{0}_list.txt'.format(data_dir.split('/')[-1]))
        #print(labels_file)
        #print(data_dir)
        data = pd.read_csv(labels_file, sep=" ", header=None)
        data.columns = ["rgb", "kinect", "label"]
        label_to_dirs = {}
        for label in gesture_labels:

            directory_labels = data.loc[data['label'] == label][type_data].tolist()
            # strip .avi from the end of the filename
            
            directories = [label[:-4] for label in directory_labels]
            label_to_dirs[label] = directories
            #self
            #print(label_to_dirs)

        data = []
        for label, directories in label_to_dirs.items():
            for directory in directories:
                data.append({
                    'frames': self.read_frame_tensors_from_dir(os.path.join(data_dir, directory)),
                    'label': label
                })


def GenerateGestureFramesDataLoader(gesture_labels, data_dir, max_frames_per_sample):
    """Returns a configured DataLoader instance."""

    # Build a gesture frames dataset using the configuration information.
    # This is just dummy code to be replaced.
    transformed_dataset = GestureFramesDataset(gesture_labels, data_dir, max_frames_per_sample)
    return DataLoader(transformed_dataset, batch_size=4,
                      shuffle=True, num_workers=4)
