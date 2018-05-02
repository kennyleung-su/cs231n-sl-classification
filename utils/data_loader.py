# TODO: Implement data loading and preprocessing.

import csv
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
    def __init__(self, data_dir, transform):
        # TODO(kenny): Read in and transform the video frames from the
        # given information into 4D frame tensors, keeping track of the
        # classification label along the way.
        self.data = None
        self.len = None

    def __getitem__(self, gesture_label):
        # TODO(kenny): Figure out how to sample randomly while keeping the
        # proportion of labels uniform, despite having a varying number of
        # videos per label.
        return self.data[gesture_label]

    def __len__(self):
        return self.len


    def gestures(gesture_list, textfile= "", type_data="kinect"):
        """Returns dictionary with labels and the locations for the frames.

        Example usage: gestures(gesture_list=[5,19,38,37,8,29,213,241,18,92],
                                textfile = 'train_list_2.txt',
                                type_data = "kinect")
        TODO(kenny): Incorporate the following method into GestureFramesDataset.
        """
        data = pd.read_csv(textfile, sep=" ", header=None)
        data.columns = ["rgb", "kinect", "label"]
        gesture_labels ={}
        
        for number in args:
            labels =[]
            directory_labels = data.loc[data['label'] == number][type_data].tolist()
            for element in directory_labels:
                labels.append(element[:-4])
            gesture_labels[number] = labels
        return gesture_labels


def GenerateGestureFramesDataLoader(config):
    """Returns a configured DataLoader instance.

    Args:
        config: dict, defines the hyperparameters used when processing and
            loading dataset information
    """

    # Build a gesture frames dataset using the configuration information.
    # This is just dummy code to be replaced.
    transformed_dataset = GestureFramesDataset(data_dir=None,
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))
    return DataLoader(transformed_dataset, batch_size=4,
                      shuffle=True, num_workers=4)
