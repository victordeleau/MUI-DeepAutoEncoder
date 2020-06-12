
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
from torch.utils.data import dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from codae.dataset import MixedVariableDataset
from codae.model import MixedVariableDenoisingAutoencoder
from codae.tool import set_logging, get_date
from codae.tool import CombinedCriterion, LossManager
from codae.tool import collate_embedding, Corrupter


def parse():

    parser = argparse.ArgumentParser(
        description='Train DAE on Abalone data.')

    parser.add_argument('--dataset_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--debug', type=bool, default=False)

    # train with missing variables from 1 to m-1
    parser.add_argument('--nb_missing', type=int, default=1)

    # infer with missing variables from 1 to m-1
    #parser.add_argument('--inference_robustness', type=int, default=1)

    return parser.parse_args()


if __name__=="__main__":

    ############################################################################
    # configuration ############################################################

    print("===== Train DAE on Abalone data =====")

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

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        args.log.info("CUDA available, loading GPU device")
    else:
        args.log.info("No CUDA device available, using CPU") 
        
    device = torch.device("cuda:0" if use_gpu else "cpu")


    ############################################################################
    # data preprocessing ####################################################

    # load data
    dataset_path = os.path.join(args.dataset_path, "abalone.data")
    with open(dataset_path, 'r') as f:
        dataset = pd.read_csv(f, sep=",")

    # min-max normalization
    normalizer = MinMaxScaler()
    dataset.iloc[:, 1:] = normalizer.fit_transform(dataset.iloc[:, 1:])

    # instantiate data
    dataset = MixedVariableDataset(dataset)

    indices = list(range(dataset.nb_observation))

    nb_train = math.floor(
        dataset.nb_observation * config["DATASET"]["SPLIT"][0])
    nb_validation = dataset.nb_observation - nb_train

    # shuffle data
    if config["DATASET"]["SHUFFLE"]:
        np.random.seed(config["SEED"])
        np.random.shuffle(indices)

    # split data
    train_indices = indices[:nb_train]
    validation_indices = indices[nb_train:]

    nb_train_batch = nb_train / config["MODEL"]["BATCH_SIZE"]
    nb_validation_batch = nb_validation / config["MODEL"]["BATCH_SIZE"]

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

    # augment corrupted indices
    corrupter = Corrupter(
        nb_observation=dataset.nb_observation,
        arch=dataset.arch,
        k_max=args.nb_missing,
        device=device)


    ############################################################################
    # initialize model #########################################################

    args.log.info("Initializing the model.")

    model = MixedVariableDenoisingAutoencoder(
        arch=dataset.arch,
        io_size=dataset.io_size,
        z_size=config["MODEL"]["Z_SIZE"],
        device=device,
        nb_input_layer=config["MODEL"]["NB_INPUT_LAYER"],
        nb_output_layer=config["MODEL"]["NB_OUTPUT_LAYER"],
        steep_layer_size=config["MODEL"]["STEEP_LAYER_SIZE"]) 

    print(model)
    model.to(device)
    dataset.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["MODEL"]["LEARNING_RATE"],
        weight_decay=config["MODEL"]["WEIGHT_DECAY"])

    # mix regression + classification loss
    combined_criterion = CombinedCriterion(
        arch=dataset.arch,
        k_max=args.nb_missing,
        device=torch.device(device))

    loss_manager = LossManager(device=torch.device(device))
    loss_manager.add_book("ftl", (args.nb_missing, dataset.nb_predictor))
    loss_manager.add_book("ptl", (args.nb_missing, dataset.nb_predictor))
    loss_manager.add_book("fvl", (args.nb_missing, dataset.nb_predictor))
    loss_manager.add_book("pvl", (args.nb_missing, dataset.nb_predictor))


    ############################################################################
    # training #################################################################

    for epoch in range( config["MODEL"]["EPOCH"] ):

        args.log.info("===================================================== EPOCH = %d" %epoch)

        # training #############################################################

        for run in range(corrupter.nb_run):

            for c, (input_data, batch_indices) in enumerate(train_loader):

                print("AUGMENTATION RUN %2d/%2d BATCH NUMBER = %5d/%5d" %(
                    run+1, corrupter.nb_run, c, nb_train_batch), end="\r")

                masks, fmask = corrupter.get_masks(batch_indices, run)
                
                c_input_data = model.corrupt(input_data=input_data, mask=fmask)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # compute the global training loss
                ptl = combined_criterion(input_data, output_data, masks)

                loss_manager.get_book("ftl").add(ptl)
                loss_manager.get_book("ptl").add(ptl)
                
                # backpropagate training loss
                optimizer.zero_grad()
                loss_manager.get_mean(ptl).backward()
                optimizer.step()


        loss_manager.get_book("ftl").sum().divide(dataset.nb_predictor*nb_train)
        loss_manager.log_book("ftl")
        loss_manager.log_book("ptl")
        loss_manager.get_book("ftl").purge()
        loss_manager.get_book("ptl").purge()

        args.log.info("TRAINING   FULL RMSE                                = %7f" %loss_manager.get_log("ftl")[-1])

        # denormalize loss
        # ptl[1:] = normalizer.inverse_transform(np.array([ptl[1:]]))[0]

        # validation ###########################################################

        # corrupt each of the possibly missing input embeddings
        for run in range(corrupter.nb_run):

            # go over validation data ##########################################
            for c, (input, indices) in enumerate(validation_loader):

                print("AUGMENTATION RUN %2d/%2d BATCH NUMBER = %5d/%5d" %(
                    run+1, corrupter.nb_run, c, nb_validation_batch), end="\r")

                masks, fmask = corrupter.get_masks(batch_indices, run)

                c_input_data = model.corrupt(input_data=input_data, mask=fmask)

                # apply forward pass to corrupted input data
                output = model( c_input_data )

                # compute the global validation loss
                fvl = combined_criterion(input_data, output_data, masks)

                loss_manager.get_book("fvl").add(fvl)
                loss_manager.get_book("pvl").add(fvl)
                

        loss_manager.get_book("fvl").sum().divide(dataset.nb_predictor*nb_train)
        loss_manager.log_book("fvl")
        loss_manager.log_book("pvl")
        loss_manager.get_book("fvl").purge()
        loss_manager.get_book("pvl").purge()

        args.log.info("VALIDATION FULL RMSE                                = %7f" %loss_manager.get_log("fvl")[-1])

        # denormalize loss
        # pvl[1:] = normalizer.inverse_transform(np.array([pvl[1:]]))[0]

    args.log.info("TRAINING HAS ENDED.")


    ############################################################################
    # log/plot #################################################################

    """
    # ensure output dir exists
    now = get_date()
    d = os.path.join(args.output_path,now+"_train_"+config["data"]["NAME"])
    if not os.path.exists(d):
        os.makedirs(d)

    epoch_axis = np.arange(0, config["MODEL"]["EPOCH"])

    # plot partial RMSE ########################################################

    for i, name in enumerate(data.variable_names):

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
    """