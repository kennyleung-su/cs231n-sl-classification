from configs import config
from configs.config import MODEL_CONFIG
from utils import data_loader
from tqdm import tqdm
import logging

import torch

DATA_DIRS = [config.TRAIN_DATA_DIR]

# get the mean and std for the training dataset
def main():
    with torch.no_grad():
        train_dataloader = data_loader.GetDataLoaders(DATA_DIRS, MODEL_CONFIG)[0]
        X_list = []

        for batch_idx, (X, y) in enumerate(tqdm(train_dataloader)):
            X_list.append(X)

        all_X = torch.cat(X_list, 0)
        rgb = all_X.transpose(0, 1).reshape(3, -1)
        mean = rgb.mean(dim=1)
        std = rgb.std(dim=1)
        logging.info("Mean: {}".format(mean))
        logging.info("Std: {}".format(std))

if __name__ == '__main__':
    main()