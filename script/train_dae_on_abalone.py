
import json
import os, sys
import math
import logging
import argparse
import random
from operator import add

import yaml
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from codae.dataset import MixedVariableDataset
from codae.model import MixedVariableDenoisingAutoencoder
from codae.tool import collate_embedding, set_logging, get_date
from codae.tool import CombinedCriterion


def parse():

    parser = argparse.ArgumentParser(
        description='Train DAE on Abalone dataset.')

    parser.add_argument('--dataset_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--debug', type=bool, default=False)

    return parser.parse_args()


if __name__=="__main__":

    ############################################################################
    # configuration ############################################################

    print("===== Train DAE on Abalone dataset =====")

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
    # dataset preprocessing ####################################################

    # load dataset
    data_path = os.path.join(args.dataset_path, "abalone.data")
    with open(data_path, 'r') as f:
        dataset = pd.read_csv(f, sep=",")

    # min-max normalization
    normalizer = MinMaxScaler()
    dataset.iloc[:, 1:] = normalizer.fit_transform(dataset.iloc[:, 1:])

    # instantiate dataset
    dataset = MixedVariableDataset(dataset)

    indices = list(range(dataset.nb_observation))

    nb_train_observation = math.floor(
        dataset.nb_observation * config["DATASET"]["SPLIT"][0])
    nb_validation_observation = dataset.nb_observation - nb_train_observation

    if config["DATASET"]["SHUFFLE"]:
        np.random.seed(config["SEED"])
        np.random.shuffle(indices)

    train_indices = indices[:nb_train_observation]
    validation_indices = indices[nb_train_observation:]

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

    # randomized corrupted index
    c = list( range( dataset.nb_predictor ) )
    augment_index=[random.sample(c, len(c)) for i in range(dataset.nb_observation)]


    ############################################################################
    # initialize model #########################################################

    args.log.info("Initializing the model.")

    model = MixedVariableDenoisingAutoencoder(
        input_arch=dataset.arch,
        io_size=dataset.io_size,
        z_size=config["MODEL"]["Z_SIZE"],
        nb_input_layer=config["MODEL"]["NB_INPUT_LAYER"],
        nb_output_layer=config["MODEL"]["NB_OUTPUT_LAYER"],
        steep_layer_size=config["MODEL"]["STEEP_LAYER_SIZE"]) 

    print(model)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        args.log.info("CUDA available, loading GPU device")
    else:
        args.log.info("No CUDA device available, using CPU") 
        
    device = torch.device("cuda:0" if use_gpu else "cpu")
    model.to(device)
    dataset.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["MODEL"]["LEARNING_RATE"],
        weight_decay=config["MODEL"]["WEIGHT_DECAY"])

    # mix regression + classification loss
    combined_criterion = CombinedCriterion(dataset.arch)

    # display/save information about the model & dataset
    metric_log = {}

    ############################################################################
    # training #################################################################

    nb_train_batch = len(train_indices) / config["MODEL"]["BATCH_SIZE"]
    nb_validation_batch = len(validation_indices) / config["MODEL"]["BATCH_SIZE"]

    metric_log["full_training_loss"] = []
    metric_log["full_validation_loss"] = []
    metric_log["partial_training_loss"] = {}
    metric_log["partial_validation_loss"] = {}
    for variable_name in dataset.variable_names:
        metric_log["partial_training_loss"][variable_name] = []
        metric_log["partial_validation_loss"][variable_name] = []

    for epoch in range( config["MODEL"]["EPOCH"] ):

        args.log.info("===================================================== EPOCH = %d" %epoch)

        # training #############################################################

        full_training_loss = 0
        full_validation_loss = 0
        partial_training_loss = [0 for i in range(dataset.nb_predictor)]
        partial_validation_loss = [0 for i in range(dataset.nb_predictor)]

        for augment_run in range(dataset.nb_predictor):

            for c, (input_data, idx) in enumerate(train_loader):

                print("AUGMENTATION RUN %2d/%2d BATCH NUMBER = %5d/%5d" %(
                    augment_run+1, dataset.nb_predictor, c, nb_train_batch), end="\r")

                corrupt_embedding = [ augment_index[i][augment_run] for i in idx ]
                
                # corrupt input data using zero_continuous noise
                c_input_data, c_mask = model.corrupt(
                    input_data=input_data,
                    device=torch.device(device),
                    corruption_type="zero_continuous",
                    indices=corrupt_embedding,
                    arch=dataset.arch)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # compute the global training loss
                ftl = combined_criterion.full_loss(input_data, output_data)
                full_training_loss += ftl.item()

                # backpropagate global training loss
                optimizer.zero_grad()
                ftl.backward()
                optimizer.step()

                ptl = combined_criterion(input_data*c_mask.float(),output_data*c_mask.float())
                partial_training_loss = [a+b.item() for a, b in zip(partial_training_loss, ptl)]

        full_training_loss /= nb_train_batch*dataset.nb_predictor
        partial_training_loss=[x/len(train_indices) for x in partial_training_loss]

        args.log.info("TRAINING FULL RMSE                                 = %7f" %full_training_loss)
        metric_log["full_training_loss"].append( full_training_loss )

        # denormalize loss
        partial_training_loss[1:] = normalizer.inverse_transform(np.array([partial_training_loss[1:]]))[0]

        for i, name in enumerate(dataset.variable_names):
            args.log.info("TRAINING PARTIAL RMSE :     %s     = %7f" %(name.ljust(18, " "), partial_training_loss[i]))
            metric_log["partial_training_loss"][name].append( partial_training_loss[i] )

        # validation ###########################################################

        # corrupt each of the possibly missing input embeddings
        for augment_run in range(dataset.nb_predictor):

            # go over validation data ##########################################
            for c, (input_data, idx) in enumerate(validation_loader):

                print("AUGMENTATION RUN %2d/%2d BATCH NUMBER = %5d/%5d" %(
                    augment_run+1, dataset.nb_predictor, c, nb_validation_batch), end="\r")

                corrupt_embedding = [ augment_index[i][augment_run] for i in idx ]

                # corrupt input data using zero_continuous noise
                c_input_data, c_mask = model.corrupt(
                    input_data=input_data,
                    device=torch.device(device),
                    corruption_type="zero_continuous",
                    indices=corrupt_embedding,
                    arch=dataset.arch)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # compute the global validation loss
                fvl = combined_criterion.full_loss(input_data, output_data)
                full_validation_loss += fvl.item()

                # compute the partial validation loss
                pvl = combined_criterion(input_data*c_mask.float(),output_data*c_mask.float())
                partial_validation_loss = [a+b.item() for a, b in zip(partial_validation_loss, pvl)]

        full_validation_loss /= nb_validation_batch*dataset.nb_predictor
        partial_validation_loss=[x/len(validation_indices) for x in partial_validation_loss]

        args.log.info("VALIDATION FULL RMSE                               = %7f" %full_validation_loss)
        metric_log["full_validation_loss"].append( full_validation_loss )

        # denormalize loss
        partial_validation_loss[1:] = normalizer.inverse_transform(np.array([partial_validation_loss[1:]]))[0]

        for i, name in enumerate(dataset.variable_names):
            args.log.info("VALIDATION PARTIAL RMSE :   %s       = %7f" %(name.ljust(16, " "), partial_validation_loss[i]))
            metric_log["partial_validation_loss"][name].append( partial_validation_loss[i] )

    args.log.info("TRAINING HAS ENDED.")


    ############################################################################
    # log/plot #################################################################

    # ensure output dir exists
    now = get_date()
    d = os.path.join(args.output_path,now+"_train_"+config["DATASET"]["NAME"])
    if not os.path.exists(d):
        os.makedirs(d)

    epoch_axis = np.arange(0, config["MODEL"]["EPOCH"])

    # plot partial RMSE ########################################################

    for i, name in enumerate(dataset.variable_names):

        plt.plot(epoch_axis, metric_log["partial_training_loss"][name],
            label="Partial training RMSE")
        plt.plot(epoch_axis, metric_log["partial_validation_loss"][name],
            label="Partial validation RMSE")

        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend(loc='best')
        plt.title(name + " prediction")

        plt.savefig(os.path.join(d,"partial_RMSE_" + name + ".png"))
        plt.clf()

    # plot full RMSE ###########################################################

    plt.plot(epoch_axis, metric_log["full_training_loss"],
        label="Full training RMSE")
    plt.plot(epoch_axis, metric_log["full_validation_loss"],
        label="Full validation RMSE")

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='best')

    plt.savefig(os.path.join(d,"full_RMSE.png"))
    plt.clf()
    
    # export metric ############################################################

    with open(os.path.join(d, "metric_log.json"),'w+') as f:
        json.dump(metric_log, f)

    args.log.info("Data saved in directory %s" %d)