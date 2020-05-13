
import json
import os, sys
import math

import yaml
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from codae.dataset import MixedVariableDataset
from codae.model import MixedVariableDenoisingAutoencoder
from codae.tool import simple_collate

def parse():

    parser = argparse.ArgumentParser(
        description='Train DAE on Abalone dataset.')

    parser.add_argument('--dataset_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=False)

    parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--debug', type=bool, default=False)

    return parser.parse_args()


if __name__=="__main__":

    ############################################################################
    # configuration ############################################################

    print("===== Train DAE on Abalone dataset =====")

    args = parse()

    # open config file
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise e


    ############################################################################
    # dataset preprocessing ####################################################

    # load dataset
    data_path = os.path.join(args.dataset_path, "abalone.data")
    with open(data_path, 'r') as f:
        dataset = pd.read_csv(f, sep=",")

    # min-max normalization
    dataset.iloc[:, 1:] = MinMaxScaler().fit_transform(dataset.iloc[:, 1:])

    # instantiate dataset
    dataset = MixedVariableDataset(dataset)

    indices = list(range(dataset.nb_observation))

    nb_train_observation = math.floor(
        dataset.nb_observation * config["DATASET"]["SPLIT"][0])
    nb_validation_observation = dataset.nb_observation - nb_train_observation

    if config["DATASET"]["SHUFFLE"]:
        np.random.seed(config["SEED"])
        np.random.shuffle(indices)

    train_indices = indices[nb_train_observation:]
    validation_indices = indices[:nb_validation_observation]

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config["MODEL"]["BATCH_SIZE"],
        collate_fn=simple_collate,
        sampler=SubsetRandomSampler(train_indices))

    validation_loader = DataLoader(
        dataset=dataset,
        batch_size=config["MODEL"]["BATCH_SIZE"],
        collate_fn=simple_collate,
        sampler=SubsetRandomSampler(validation_indices))


    ############################################################################
    # training #################################################################

    