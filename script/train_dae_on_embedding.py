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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import yaml

from codae.tool import set_logging, display_info, get_date
from codae.tool import export_parameters_to_json
from codae.tool import load_dataset_of_embeddings, parse, collate_embedding
from codae.tool import Corrupter, CombinedCriterion, RankingLoss

from codae.model import EmbeddingDenoisingAutoencoder

from codae.dataset import ConcatenatedEmbeddingDataset


def parse():

    parser = argparse.ArgumentParser(
        description='Train denoising autoencoder.')

    parser.add_argument('--embedding_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--rank', type=bool, default=False)

    # train with missing variables from 1 to m-1
    parser.add_argument('--nb_missing', type=int, default=1)

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

    dataset_std = torch.std(dataset.data)
    args.log.info("Dataset STD = " + str(dataset_std))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        args.log.info("CUDA available, loading GPU device")
    else:
        args.log.info("No CUDA device available, using CPU") 
        
    device = torch.device("cuda:0" if use_gpu else "cpu")


    ############################################################################
    # split dataset ############################################################

    indices = list(range(dataset.nb_observation))

    nb_train = math.floor(
        dataset.nb_observation * config["DATASET"]["SPLIT"][0])
    nb_validation = dataset.nb_observation - nb_train

    nb_train_batch = nb_train / config["MODEL"]["BATCH_SIZE"]
    nb_validation_batch = nb_validation / config["MODEL"]["BATCH_SIZE"]

    if config["DATASET"]["SHUFFLE"]:
        np.random.seed(config["SEED"])
        np.random.shuffle(indices)

    train_indices = indices[:nb_train]
    validation_indices = indices[nb_train:]

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config["MODEL"]["BATCH_SIZE"],
        collate_fn=collate_embedding,
        sampler=SubsetRandomSampler(train_indices))

    validation_loader = DataLoader(
        dataset=dataset,
        batch_size=config["MODEL"]["BATCH_SIZE"],
        collate_fn=collate_embedding,
        sampler=SubsetRandomSampler(validation_indices))

    # randomized corruption index
    c = list( range( len( config["DATASET"]["USED_CATEGORY"] ) ) )
    augment_index=[random.sample(c, len(c)) for i in range(dataset.nb_observation)]

    # augment corrupted indices
    corrupter = Corrupter(
        nb_observation=dataset.nb_observation,
        arch=dataset.arch,
        k_max=args.nb_missing,
        device=device)


    ############################################################################
    # initialize model #########################################################

    args.log.info("Initializing the model.")

    io_size = config["DATASET"]["EMBEDDING_SIZE"]*len(config["DATASET"]["USED_CATEGORY"])

    model = EmbeddingDenoisingAutoencoder(
        io_size=io_size,
        z_size=config["MODEL"]["Z_SIZE"],
        embedding_size=config["DATASET"]["EMBEDDING_SIZE"],
        nb_input_layer=config["MODEL"]["NB_INPUT_LAYER"],
        nb_output_layer=config["MODEL"]["NB_OUTPUT_LAYER"],
        steep_layer_size=config["MODEL"]["STEEP_LAYER_SIZE"]) 
    
    model.to(device)
    dataset.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["MODEL"]["LEARNING_RATE"],
        weight_decay=config["MODEL"]["WEIGHT_DECAY"])

    criterion = torch.nn.MSELoss()

    # display/save information about the model & dataset
    metric_log = {}
    metric_log = display_info(config, dataset.nb_observation, metric_log)
    args.log.info(model)

    # mix regression + classification loss
    mean_criterion = torch.nn.MSELoss(reduction="mean")
    full_criterion = torch.nn.MSELoss(reduction="none")

    book = {} # list( KxM )
    book["ftl"], book["ptl"] = [], []
    book["fvl"], book["pvl"] = [], []
    book["rl"] = []

    ranking_loss = RankingLoss(dataset, validation_indices, device=device)


    ############################################################################
    # train ####################################################################
    
    for epoch in range( config["MODEL"]["EPOCH"] ):

        args.log.info("===================================================== EPOCH = %d" %epoch)

        ftl, ptl = 0, 0

        # go over training data ############################################
        for c, (input_data, batch_indices) in enumerate(train_loader):

            print("BATCH NUMBER = %5d/%5d" %(c, nb_train_batch), end="\r")

            masks, fmask = corrupter.get_masks(batch_indices, 0)

            c_input_data = model.corrupt(input_data=input_data, mask=fmask)

            # apply forward pass to corrupted input data
            output_data = model( c_input_data )

            # compute the global training loss
            loss = mean_criterion(input_data, output_data)

            # backpropagate training loss
            optimizer.zero_grad()
            loss.backward()

            if config["MODEL"]["TRUNK_GRAD"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                
            optimizer.step()

            # get full loss
            loss = full_criterion(input_data, output_data)
            loss = loss.cpu().detach().numpy()
            ftl += np.sum(loss)

            # get partial loss
            ptl += np.sum((1-fmask.cpu().numpy())*loss)

        ftl /= dataset.nb_predictor*nb_train
        ptl /= nb_train*dataset.nb_predictor/dataset.nb_used_category

        ftl, ptl = np.sqrt(ftl), np.sqrt(ptl)

        book["ftl"].append( ftl )
        book["ptl"].append( ptl )

        args.log.info("TRAINING FULL ERROR      = %7f" %book["ftl"][-1])
        args.log.info("TRAINING PARTIAL ERROR   = %7f" %book["ptl"][-1])

        ####################################################################
        # go over validation data ##########################################

        fvl, pvl, rl = 0, 0, 0

        for c, (input_data, batch_indices) in enumerate(validation_loader):

            print("BATCH NUMBER = %5d/%5d" %(c, nb_validation_batch), end="\r")

            masks, fmask = corrupter.get_masks(batch_indices, 0)

            c_input_data = model.corrupt(input_data=input_data, mask=fmask)

            # apply forward pass to corrupted input data
            output_data = model( c_input_data )

            # get full loss
            loss = full_criterion(input_data, output_data)
            loss = loss.cpu().detach().numpy()
            fvl += np.sum(loss)

            # get partial loss
            pvl += np.sum((1-fmask.cpu().numpy())*loss)

            # get ranking loss
            rl += ranking_loss.get(output_data, fmask, batch_indices)

        fvl /= dataset.nb_predictor*nb_validation
        pvl /= nb_validation*dataset.nb_predictor/dataset.nb_used_category
        rl /= nb_validation

        fvl, pvl = np.sqrt(fvl), np.sqrt(pvl)

        book["fvl"].append( fvl )
        book["pvl"].append( pvl )
        book["rl"].append( rl )

        args.log.info("VALIDATION FULL ERROR    = %7f" %book["fvl"][-1])
        args.log.info("VALIDATION PARTIAL ERROR = %7f" %book["pvl"][-1])
        args.log.info("VALIDATION RANKING ERROR = %7f" %book["rl"][-1])

    args.log.info("TRAINING HAS ENDED.")


    ############################################################################
    # log/plot #################################################################

    # ensure output dir exists
    now = get_date()
    d = os.path.join(args.output_path,now+"_train_"+config["DATASET"]["NAME"])
    if not os.path.exists(d):
        os.makedirs(d)

    epoch_axis = np.arange(0, config["MODEL"]["EPOCH"])

    # plot full RMSE #######################################################

    plt.plot(epoch_axis, book["ftl"],
        label="Training")
    plt.plot(epoch_axis, book["fvl"],
        label="Validation")

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='best')

    plt.savefig(os.path.join(d,"full_RMSE.png"))
    plt.clf()

    # plot partial RMSE #####################################################

    plt.plot(epoch_axis, book["ptl"],
        label="Training")
    plt.plot(epoch_axis, book["pvl"],
        label="Validation")
    plt.plot(epoch_axis, [dataset_std for i in range(len(epoch_axis))],
        label="Validation standard deviation")

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='best')

    plt.savefig(os.path.join(d,"partial_RMSE.png"))
    plt.clf()

    # plot ranking loss ########################################################

    if args.rank: # takes much longer

        plt.plot(epoch_axis, book["rl"])
        plt.plot(epoch_axis, [0.5 for i in range(len(epoch_axis))], "Random rank")

        plt.xlabel('Epoch')
        plt.ylabel('RIRE')
        plt.legend(loc='best')

        plt.savefig(os.path.join(d,"partial_RIRE.png"))
        plt.clf()
    
    # export metric ############################################################

    args.log.info("Data saved in directory %s" %d)
    