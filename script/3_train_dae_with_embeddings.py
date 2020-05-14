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
from codae.tool import get_object_size, get_rmse, LossAnalyzer, PlotDrawer, export_parameters_to_json, get_ranking_loss
from codae.tool import load_dataset_of_embeddings, parse, collate_embedding

from codae.model import EmbeddingDenoisingAutoencoder

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

    # display/save information about the model & dataset
    metric_log = {}
    metric_log = display_info(config, metric_log)
    args.log.info(model)


    ############################################################################
    # train ####################################################################

    nb_train_batch = len(train_indices) / config["MODEL"]["BATCH_SIZE"]
    nb_validation_batch = len(validation_indices) / config["MODEL"]["BATCH_SIZE"]

    metric_log["mean"], metric_log["std"] = [], []
    metric_log["training_ranking_loss"] = []
    metric_log["full_training_loss"] = []
    metric_log["partial_training_loss"] = []
    metric_log["validation_ranking_loss"] = []
    metric_log["full_validation_loss"] = []
    metric_log["partial_validation_loss"] = []
    
    for epoch in range( config["MODEL"]["EPOCH"] ):

        args.log.info("===================================================== EPOCH = %d" %epoch)

        full_training_loss, partial_training_loss = 0, 0
        training_ranking_loss, mean, std = 0, 0, 0

        # corrupt each of the possibly missing input embeddings
        for augment_run in range(dataset.nb_used_category):

            # go over training data ############################################
            for c, (input_data, idx) in enumerate(train_loader):

                print("AUGMENTATION RUN %2d/%2d BATCH NUMBER = %5d/%5d" %(
                    augment_run+1, dataset.nb_used_category, c, nb_train_batch), end="\r")

                corrupt_embedding = [ augment_index[i][augment_run] for i in idx ]
                
                # corrupt input data using zero_continuous noise
                c_input_data, c_mask = model.corrupt(
                    input_data=input_data,
                    device=torch.device(device),
                    corruption_type="zero_continuous",
                    indices=corrupt_embedding)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # compute ranking loss
                trl = get_ranking_loss(output_data, dataset, corrupt_embedding, idx, train_indices)
                training_ranking_loss += trl

                # compute the global training loss
                ftl = torch.sqrt(criterion(input_data, output_data))
                full_training_loss += ftl.item()

                # backpropagate global training loss
                optimizer.zero_grad()
                ftl.backward()
                optimizer.step()
                
                # compute training loss of missing embedding
                input_data[c_mask], output_data[c_mask] = 0., 0.
                partial_training_loss += torch.sqrt(criterion(
                    input_data,
                    output_data)).item()

                mean += input_data.mean().item()
                std += input_data.std().item()

        full_training_loss /= nb_train_batch*dataset.nb_used_category
        partial_training_loss /= nb_train_batch*dataset.nb_used_category
        training_ranking_loss /= nb_train_batch*dataset.nb_used_category

        args.log.info("TRAINING FULL RMSE      = %f" %full_training_loss)
        args.log.info("TRAINING PARTIAL RMSE   = %f" %partial_training_loss)
        args.log.info("TRAINING RANKING        = %f" %training_ranking_loss)

        metric_log["training_ranking_loss"].append( training_ranking_loss )
        metric_log["full_training_loss"].append( full_training_loss )
        metric_log["partial_training_loss"].append( partial_training_loss )

        full_validation_loss, partial_validation_loss = 0, 0
        validation_ranking_loss = 0

        # corrupt each of the possibly missing input embeddings
        for augment_run in range(dataset.nb_used_category):

            # go over validation data ##########################################
            for c, (input_data, idx) in enumerate(validation_loader):

                print("AUGMENTATION RUN %2d/%2d BATCH NUMBER = %5d/%5d" %(
                    augment_run+1, dataset.nb_used_category, c, nb_validation_batch), end="\r")

                corrupt_embedding = [ augment_index[i][augment_run] for i in idx ]

                # corrupt input data using zero_continuous noise
                c_input_data, c_mask = model.corrupt(
                    input_data=input_data,
                    device=torch.device(device),
                    corruption_type="zero_continuous",
                    indices=corrupt_embedding)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # compute ranking loss
                vrl = get_ranking_loss(output_data, dataset, corrupt_embedding, idx, validation_indices)
                validation_ranking_loss += vrl

                # compute the global validation loss
                ftl = torch.sqrt(criterion(input_data, output_data))
                full_validation_loss += ftl.item()
                
                # compute validation loss of missing embedding
                input_data[c_mask], output_data[c_mask] = 0., 0.
                partial_validation_loss += torch.sqrt(criterion(
                    input_data,
                    output_data)).item()

                mean += input_data.mean().item()
                std += input_data.std().item()

        full_validation_loss /= nb_validation_batch*dataset.nb_used_category
        partial_validation_loss /= nb_validation_batch*dataset.nb_used_category
        validation_ranking_loss /= nb_validation_batch*dataset.nb_used_category

        mean /= (nb_train_batch+nb_validation_batch)*dataset.nb_used_category
        std /= (nb_train_batch+nb_validation_batch)*dataset.nb_used_category

        args.log.info("VALIDATION FULL RMSE    = %f" %full_validation_loss)
        args.log.info("VALIDATION PARTIAL RMSE = %f" %partial_validation_loss)
        args.log.info("VALIDATION RANKING      = %f" %validation_ranking_loss)
        args.log.info("DATA MEAN               = %f" %mean)
        args.log.info("DATA STD                = %f\n" %std)

        metric_log["validation_ranking_loss"].append( validation_ranking_loss )
        metric_log["full_validation_loss"].append( full_validation_loss )
        metric_log["partial_validation_loss"].append( partial_validation_loss )
        metric_log["mean"].append( mean )
        metric_log["std"].append( std )

    args.log.info("TRAINING HAS ENDED.")


    ############################################################################
    # log/plot #################################################################

    # ensure output dir exists
    now = get_date()
    d = os.path.join(args.output_path,now+"_train_"+config["DATASET"]["NAME"])
    if not os.path.exists(d):
        os.makedirs(d)

    # plot RMSE ################################################################

    epoch_axis = np.arange(0, config["MODEL"]["EPOCH"])

    plt.plot(epoch_axis, metric_log["full_validation_loss"],
        label="Full validation RMSE")
    plt.plot(epoch_axis, metric_log["partial_validation_loss"],
        label="Partial validation RMSE")
    plt.plot(epoch_axis, metric_log["full_training_loss"],
        label="Full training RMSE")
    plt.plot(epoch_axis, metric_log["partial_training_loss"],
        label="Partial training RMSE")

    plt.plot(epoch_axis, metric_log["mean"],
        label="Input data mean")

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='best')

    plt.savefig(os.path.join(d,"RMSE.png"))
    plt.clf()

    plt.plot(epoch_axis, metric_log["training_ranking_loss"],
        label="Training ranking accuracy")
    plt.plot(epoch_axis, metric_log["validation_ranking_loss"],
        label="Validation ranking accuracy")
    plt.plot(epoch_axis, [0.5 for i in range(config["MODEL"]["EPOCH"])], label="Mean ranking accuracy")

    plt.xlabel('Epoch')
    plt.ylabel('Ranking accuracy')
    plt.ylim(ymin=0, ymax=1)

    plt.savefig(os.path.join(d,"RANKING.png"))
    plt.clf()

    # export metric ############################################################

    with open(os.path.join(d, "metric_log.json"),'w+') as f:
        json.dump(metric_log, f)

    args.log.info("Data saved in directory %s" %d)