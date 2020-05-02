# Mixed User Item Deep Auto Encoder

import sys, os, gc
import time
import math
import psutil
import logging
import time
import json
import argparse
import glob
import hashlib
import pickle
import random

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import yaml

from codae.tool import set_logging, display_info, get_date
from codae.tool import get_object_size, get_rmse, LossAnalyzer, PlotDrawer, export_parameters_to_json
from codae.tool import load_dataset_of_embeddings, parse, my_collate

from codae.model import DenoisingAutoencoder

from codae.dataset import ConcatenatedEmbeddingDataset


def parse():

    parser = argparse.ArgumentParser(
        description='Train denoising autoencoder.')

    parser.add_argument('--embedding_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--debug', type=bool, default=True)

    return parser.parse_args()


################################################################################
################################################################################
# main #########################################################################

if __name__ == "__main__":

    args = parse()

    vars(args)["log"] = set_logging(
        logging_level=(logging.DEBUG if args.debug else logging.INFO),
        log_file_path="log/")

    # open config file
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise e


    ############################################################################
    # load/process embeddings ##################################################

    args.log.info("Loading dataset.")

    dataset = load_dataset_of_embeddings(
        embedding_path=args.embedding_path,
        config=config,
        cache_dir="tmp/")


    ############################################################################
    # split dataset ############################################################

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
        collate_fn=my_collate,
        sampler=SubsetRandomSampler(train_indices))

    validation_loader = DataLoader(
        dataset=dataset,
        batch_size=config["MODEL"]["BATCH_SIZE"],
        collate_fn=my_collate,
        sampler=SubsetRandomSampler(validation_indices))

    # randomized corruption index
    c = list( range( len( config["DATASET"]["USED_CATEGORY"] ) ) )
    augment_index=[random.sample(c, len(c)) for i in range(dataset.nb_observation)]


    ############################################################################
    # initialize model #########################################################

    args.log.info("Initializing the model.")

    io_size = config["DATASET"]["EMBEDDING_SIZE"]*len(config["DATASET"]["USED_CATEGORY"])

    model = DenoisingAutoencoder(
        io_size=io_size,
        z_size=config["MODEL"]["Z_SIZE"],
        embedding_size=config["DATASET"]["EMBEDDING_SIZE"],
        nb_input_layer=config["MODEL"]["NB_INPUT_LAYER"],
        nb_output_layer=config["MODEL"]["NB_OUTPUT_LAYER"],
        steep_layer_size=config["MODEL"]["STEEP_LAYER_SIZE"]) 

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        args.log.info("CUDA available, loading GPU device\n")
    else:
        args.log.info("No CUDA device available, using CPU\n") 
        
    device = torch.device("cuda:0" if use_gpu else "cpu")
    model.to(device)
    dataset.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["MODEL"]["LEARNING_RATE"],
        weight_decay=config["MODEL"]["WEIGHT_DECAY"])

    criterion = torch.nn.MSELoss()

    # display information about the model & dataset
    args.log.info(model)
    display_info(config)


    ############################################################################
    # train ####################################################################

    nb_train_batch = len(train_indices) / config["MODEL"]["BATCH_SIZE"]
    nb_validation_batch = len(validation_indices) / config["MODEL"]["BATCH_SIZE"]
    
    for epoch in range( config["MODEL"]["EPOCH"] ):

        args.log.info("===================================================== EPOCH = %d" %epoch)

        full_training_loss, partial_training_loss = 0, 0

        # corrupt each of the possibly missing input embeddings
        for augment_run in range(dataset.nb_used_category):

            # go over training data ############################################
            for c, (input_data, idx) in enumerate(train_loader):

                #print("BATCH NUMBER = %d/%d" %(c, nb_train_batch), end="\r")

                corrupt_idx = [ augment_index[i][augment_run] for i in idx ]
                
                # corrupt input data using zero_continuous noise
                c_input_data, c_mask = model.corrupt(
                    input_data=input_data,
                    device=torch.device(device),
                    corruption_type="zero_continuous",
                    indices=corrupt_idx)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # compute the global training loss
                ftl = torch.sqrt(criterion(input_data, output_data))
                full_training_loss += ftl

                # backpropagate global training loss
                optimizer.zero_grad()
                ftl.backward()
                optimizer.step()
                
                # compute training loss of missing embedding
                input_data[c_mask], output_data[c_mask] = 0., 0.
                partial_training_loss += torch.sqrt(criterion(
                    input_data,
                    output_data))

        full_training_loss /= nb_train_batch*dataset.nb_used_category
        partial_training_loss /= nb_train_batch*dataset.nb_used_category
        args.log.info("TRAINING FULL RMSE      = %f" %full_training_loss)
        args.log.info("TRAINING PARTIAL RMSE   = %f\n" %partial_training_loss)
        
        full_validation_loss, partial_validation_loss = 0, 0

        # corrupt each of the possibly missing input embeddings
        for augment_run in range(dataset.nb_used_category):

            # go over validation data ##########################################
            for c, (input_data, idx) in enumerate(validation_loader):

                corrupt_idx = [ augment_index[i][augment_run] for i in idx ]

                # corrupt input data using zero_continuous noise
                c_input_data, c_mask = model.corrupt(
                    input_data=input_data,
                    device=torch.device(device),
                    corruption_type="zero_continuous",
                    indices=corrupt_idx)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # compute the global validation loss
                ftl = torch.sqrt(criterion(input_data, output_data))
                full_validation_loss += ftl
                
                # compute validation loss of missing embedding
                input_data[c_mask], output_data[c_mask] = 0., 0.
                partial_validation_loss += torch.sqrt(criterion(
                    input_data,
                    output_data))

        full_validation_loss /= nb_validation_batch*dataset.nb_used_category
        partial_validation_loss /= nb_validation_batch*dataset.nb_used_category
        args.log.info("VALIDATION FULL RMSE    = %f" %full_validation_loss)
        args.log.info("VALIDATION PARTIAL RMSE = %f\n" %partial_validation_loss)

    args.log.info("DONE.")


    ############################################################################
    # log/plot #################################################################
    
    """
    plot_drawer.add( data=[ training_rmses, validation_rmses ], title="RMSE", legend=["training rmse", "validation_rmse"], display=True )
    plot_drawer.export_to_png(idx = 0, export_path=output_dir)
    export_parameters_to_json(args, output_dir)
    """