# TODO: Implement data loading and preprocessing.

import csv
import numpy as np
import pandas as pd
import os
import torch

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # for visualizations


class GestureVideoDataLoader(DataLoader):
    """Implements the gesture video data loader class.

    Useful for fetching tensors corresponding to batches of RGBD frames.
    Uses a configuration file to dynamically construct the dataset, and
    uses caches/pkls whenever possible, to boost efficiency.
    """

    # TODO(kenny): Figure out whether we want to incorporate Hiroshi's library
    # in this class to construct the dataset.
    def __init__(self, config):
        self._config = config
        # TODO(kenny): Initialize the superclass DataLoader using the config
        # variables, e.g. for batch size and shuffling.
        dataset = None
        super(GestureVideoDataLoader, self).__init__(dataset)


# TODO(kenny): Incorporate the following method into GestureVideoDataLoader.
#returns dictionary with labels and the locations for the frames 
def gestures(*args, textfile= "", type_data="kinect"):
    data = pd.read_csv(textfile, sep=" ", header=None)
    data.columns = ["rgb", "kinect", "label"]
    gesture_labels ={}
    
    for number in args:
        labels =[]
        print(number)
        directory_labels = data.loc[data['label'] == number][type_data].tolist()
        for element in directory_labels:
            labels.append(element[:-4])
        gesture_labels[number] = labels
    return gesture_labels
gestures = gestures(5,19,38,37,8,29,213,241,18,92, textfile = 'train_list_2.txt', type_data = "kinect")
