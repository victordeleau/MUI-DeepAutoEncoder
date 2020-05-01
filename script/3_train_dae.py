# Mixed User Item Deep Auto Encoder

import sys, os
import time
import math
import gc
import psutil
import logging
import time
import json
import argparse
import yaml
import glob
import hashlib
import pickle

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np

from codae.tool import set_logging, display_info
from codae.tool import get_object_size, get_rmse, LossAnalyzer, PlotDrawer, export_parameters_to_json
from codae.tool import parse
from codae.tool import get_day_month_year_hour_minute_second
from codae.tool import BatchBuilder
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


def my_collate(batch):

    return torch.stack(batch)


if __name__ == "__main__":

    print()

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

    using_cache = False
    dataset_cache = glob.glob('tmp/*_dataset.bin')

    if len(dataset_cache) > 0: # check for cached dataset

        dataset_cache_path = dataset_cache[0]
        dataset_hash = dataset_cache_path.split("/")[-1].split("_")[0]

        # load cached dataset if corresponding embedding file hasn't changed
        if dataset_hash == hashlib.sha1( str( os.stat(
            args.embedding_path)[9]).encode('utf-8') ).hexdigest():

            args.log.info("Using cached dataset.")
            using_cache = True

            try:
                with open(dataset_cache_path, 'rb') as f:
                    dataset = pickle.load(f)
            except:
                raise Exception("Error while reading embedding json file.")

        else: # delete old cached dataset
            os.remove(dataset_cache_path)

    if not using_cache: # then load from json of embeddings
        
        try:
            with open(args.embedding_path, 'r') as f:
                embeddings = json.load(f)
        except:
            raise Exception("Error while reading embedding json file.")

        dataset = ConcatenatedEmbeddingDataset(
            embeddings=embeddings,
            used_category=config["USED_CATEGORY"])

        # write new cache to disk
        if not os.path.exists("tmp/"):
            os.makedirs("tmp/")
        with open("tmp/new_dataset_tmp.bin", "wb") as f:
            pickle.dump( dataset, f )
        dataset_cache_name = "tmp/" + hashlib.sha1( str(\
            os.stat(args.embedding_path)[9])\
            .encode('utf-8') ).hexdigest() + "_dataset.bin"
        os.rename("tmp/new_dataset_tmp.bin", dataset_cache_name)

    # instantiate dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["MODEL"]["BATCH_SIZE"],
        shuffle=True,
        collate_fn=my_collate)


    ############################################################################
    # initialize model #########################################################

    args.log.info("Initializing the model.")

    io_size = config["MODEL"]["EMBEDDING_SIZE"]*len(config["USED_CATEGORY"])

    model = DenoisingAutoencoder(
        io_size=io_size,
        z_size=config["MODEL"]["Z_SIZE"],
        embedding_size=config["MODEL"]["EMBEDDING_SIZE"],
        nb_input_layer=config["MODEL"]["NB_INPUT_LAYER"],
        nb_output_layer=config["MODEL"]["NB_OUTPUT_LAYER"],
        steep_layer_size=config["MODEL"]["STEEP_LAYER_SIZE"]) 

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        args.log.info("CUDA available, loading GPU device\n")
    else:
        args.log.info("No CUDA device available, using CPU\n") 

    device = torch.device("cuda:0" if use_gpu else "cpu")
    torch_device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["MODEL"]["LEARNING_RATE"],
        weight_decay=config["MODEL"]["WEIGHT_DECAY"])

    criterion = torch.nn.MSELoss()

    # display some information about the model & dataset
    args.log.info(model)
    display_info(config)

    ############################################################################
    # train ####################################################################

    nb_batch = dataset.nb_observation / config["MODEL"]["BATCH_SIZE"]
    
    for epoch in range( config["MODEL"]["EPOCH"] ):

        args.log.info("EPOCH = %d =====================================================" %epoch)

        full_training_loss, partial_training_loss = 0, 0
        full_validation_loss, partial_validation_loss = 0, 0

        for c, input_data in enumerate(dataloader):

            print("BATCH NUMBER = %d/%d" %(c, nb_batch), end="\r")
            
            input_data = input_data.to(device)

            # corrupt input data using zero_continuous noise
            c_input_data, c_mask, c_indices = model.corrupt(
                input_data=input_data,
                device=torch_device,
                corruption_type="zero_continuous",
                nb_corrupted=config["MODEL"]["NB_CORRUPTED"])

            # apply forward pass to data
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

        full_training_loss /= nb_batch
        partial_training_loss /= nb_batch

        args.log.info("FULL RMSE = %f" %full_training_loss)
        args.log.info("PARTIAL RMSE = %f\n" %partial_training_loss)


    ############################################################################
    # log/plot #################################################################
    
    """
    plot_drawer.add( data=[ training_rmses, validation_rmses ], title="RMSE", legend=["training rmse", "validation_rmse"], display=True )
    plot_drawer.export_to_png(idx = 0, export_path=output_dir)
    export_parameters_to_json(args, output_dir)
    """