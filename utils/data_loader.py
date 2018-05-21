# Implements data loading and preprocessing.
import logging
import pandas as pd
import os
import re
import torch
from torch.utils.data import DataLoader

from utils.pad_utils import PadCollate
from utils.gesture_frame_dataset import GestureFrameDataset
from utils.resnet_encoding_dataset import ResnetEncodingDataset

def GenerateDataLoader(gesture_labels, data_type, data_dir, max_seq_len,
                batch_size, transform, num_workers, max_example_per_label, shuffle=False):
    """Returns a configured DataLoader instance."""

    # Build a dataloader using the configuration information. Subjected to change
    dataset_config_dict = {
        'RGB-image': {'dataset': 'image', 'type': 'RGB'},
        'RGBD-image': {'dataset': 'image', 'type': 'RGBD'},
        'RN18-RGB-encoding': {'dataset': 'encoding', 'type': 'RN18-RGB'},
        'RN18-RGBD-encoding': {'dataset': 'encoding', 'type': 'RN18-RGBD'},
    }

    dataset_config = dataset_config_dict[data_type]

    if dataset_config['dataset'] == 'image':
        transformed_dataset = GestureFrameDataset(gesture_labels, data_dir, dataset_config['type'], transform, max_example_per_label)

        return DataLoader(transformed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    if dataset_config['dataset'] == 'encoding':
        dataset = ResnetEncodingDataset(gesture_labels, data_dir, dataset_config['type'], max_example_per_label)

        return DataLoader(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=PadCollate(transformed_dataset.max_seq_len, dim=1)
        )


def GetDataLoaders(data_dirs, model_config):
    """Returns a tuple consisting of the train, valid, and test dataloaders."""
    return (GenerateDataLoader(
        model_config.gesture_labels,
        model_config.data_type,
        data_directory,
        model_config.max_seq_len,
        model_config.batch_size,
        model_config.transform,
        model_config.num_workers or 0,
        model_config.max_example_per_label,
        model_config.shuffle
        ) for data_directory in data_dirs)