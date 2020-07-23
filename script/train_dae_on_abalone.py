
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
from codae.tool import CombinedCriterion
from codae.tool import collate_embedding, Corrupter, Normalizer


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
    # data preprocessing #######################################################

    # load data
    dataset_path = os.path.join(args.dataset_path, "abalone.data")
    with open(dataset_path, 'r') as f:
        dataset = pd.read_csv(f, sep=",")

    print(dataset.describe(include='all').loc[['count','mean', 'std', 'min', '25%', '50%', '75%', 'max']])

    normazer = MinMaxScaler()
    dataset.iloc[:, 1:] = normazer.fit_transform(dataset.iloc[:, 1:])

    tensor_normazer = Normalizer(normalizer=normazer, device=device)

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

    print(corrupter.nb_corruption_per_k)


    ############################################################################
    # initialize model #########################################################

    print()
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
    print()
    model.to(device)
    dataset.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["MODEL"]["LEARNING_RATE"],
        weight_decay=config["MODEL"]["WEIGHT_DECAY"])

    # mix regression + classification loss
    full_criterion = CombinedCriterion(
        arch=dataset.arch,
        k_max=args.nb_missing,
        device=torch.device(device),
        observation_mask=dataset.type_mask,
        weight=[0.4, 1, 1, 1, 1, 1, 1, 1, 1],
        reduction="mean")

    monitor_criterion = CombinedCriterion(
        arch=dataset.arch,
        k_max=args.nb_missing,
        device=torch.device(device),
        observation_mask=dataset.type_mask,
        reduction="none")

    book = {} # list( KxM )
    book["ftl_per_k"], book["ptl_per_k"] = [], []
    book["fvl_per_k"], book["pvl_per_k"] = [], []
    book["ftl"], book["ptl"] = [], []
    book["fvl"], book["pvl"] = [], []


    ############################################################################
    ############################################################################
    # training #################################################################

    for epoch in range( config["MODEL"]["EPOCH"] ):

        args.log.info("===================================================== EPOCH = %d\n" %epoch)

        ########################################################################
        # training #############################################################

        ftl_per_k = np.zeros((args.nb_missing, len(dataset.arch)))
        ptl_per_k = np.zeros((args.nb_missing, len(dataset.arch)))
        ftl, ptl = 0, 0
        
        for run in range(corrupter.nb_run):

            for c, (input_data, batch_indices) in enumerate(train_loader):

                print("AUGMENTATION RUN %2d/%2d BATCH NUMBER = %5d/%5d" %(
                    run+1, corrupter.nb_run, c, nb_train_batch), end="\r")

                masks, fmask = corrupter.get_masks(batch_indices, run)
                
                c_input_data = model.corrupt(input_data=input_data, mask=fmask)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # compute the global training loss
                loss = full_criterion(x=input_data, y=output_data)

                # backpropagate training loss
                optimizer.zero_grad()
                loss.backward()

                if config["MODEL"]["TRUNK_GRAD"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    
                optimizer.step()

                # inverse normalization
                input_data[:, 3:] = tensor_normazer.undo(input_data[:, 3:])
                output_data[:, 3:] = tensor_normazer.undo(output_data[:, 3:])

                # compute useful other metrics
                loss = monitor_criterion(input_data, output_data, as_numpy=True)
                ftl += np.sum(loss)
                ftl_per_k += monitor_criterion.get_per_k(loss, masks)
                loss = monitor_criterion.get_partial(loss, fmask)
                ptl += np.sum(loss)
                ptl_per_k += monitor_criterion.get_per_k(loss, masks)

        for i in range(len(corrupter.nb_corruption_per_k)):
            ftl_per_k[i, :] = ftl_per_k[i, :]/(nb_train*sum(corrupter.nb_corruption_per_k[:i+1]))
            ptl_per_k[i, :] = ptl_per_k[i, :]/(nb_train*sum(corrupter.nb_corruption_per_k[:i+1])/dataset.nb_predictor)

        ftl /= sum(corrupter.nb_corruption_per_k)*nb_train
        ptl /= sum(corrupter.nb_corruption_per_k)*nb_train/dataset.nb_predictor

        ftl_per_k[:, 1:] = np.sqrt(ftl_per_k[:, 1:])
        ptl_per_k[:, 1:] = np.sqrt(ptl_per_k[:, 1:])

        ftl, ptl = np.sqrt(ftl), np.sqrt(ptl)

        book["ftl_per_k"].append( ftl_per_k )
        book["ptl_per_k"].append( ptl_per_k )
        book["ftl"].append( ftl )
        book["ptl"].append( ptl )

        args.log.info("TRAINING PARTIAL ERROR = %7f" %np.mean(book["ptl_per_k"][-1]))

        print("k ", end="")
        for name in dataset.variable_names:
            print("%s " %name.rjust(12), end="")
        print()
        for i in range(args.nb_missing):
            print("%d     " %(i+1), end="")
            for j in range(len(dataset.arch)):
                print("%f     " %book["ptl_per_k"][-1][i][j], end="")
            print()
        print()
        
        ########################################################################
        # validation ###########################################################

        pvl_per_k = np.zeros((args.nb_missing, len(dataset.arch)))
        fvl_per_k = np.zeros((args.nb_missing, len(dataset.arch)))
        fvl, pvl = 0, 0

        # corrupt each of the possibly missing input embeddings
        for run in range(corrupter.nb_run):

            # go over validation data ##########################################
            for c, (input_data, batch_indices) in enumerate(validation_loader):

                print("AUGMENTATION RUN %2d/%2d BATCH NUMBER = %5d/%5d" %(
                    run+1, corrupter.nb_run, c, nb_validation_batch), end="\r")

                masks, fmask = corrupter.get_masks(batch_indices, run)

                c_input_data = model.corrupt(input_data=input_data, mask=fmask)

                # apply forward pass to corrupted input data
                output_data = model( c_input_data )

                # inverse normalization
                input_data[:, 3:] = tensor_normazer.undo(input_data[:, 3:])
                output_data[:, 3:] = tensor_normazer.undo(output_data[:, 3:])

                # compute useful other metrics
                loss = monitor_criterion(input_data, output_data, as_numpy=True)
                fvl += np.sum(loss)
                fvl_per_k += monitor_criterion.get_per_k(loss, masks)
                loss = monitor_criterion.get_partial(loss, fmask)
                pvl += np.sum(loss)
                pvl_per_k += monitor_criterion.get_per_k(loss, masks)


        for i in range(len(corrupter.nb_corruption_per_k)):
            fvl_per_k[i, :] = fvl_per_k[i, :]/(nb_validation*sum(corrupter.nb_corruption_per_k[:i+1]))
            pvl_per_k[i, :] = pvl_per_k[i, :]/(nb_validation*sum(corrupter.nb_corruption_per_k[:i+1])/dataset.nb_predictor)

        fvl /= sum(corrupter.nb_corruption_per_k)*nb_validation
        pvl /= sum(corrupter.nb_corruption_per_k)*nb_validation/dataset.nb_predictor

        fvl_per_k[:, 1:] = np.sqrt(fvl_per_k[:, 1:])
        pvl_per_k[:, 1:] = np.sqrt(pvl_per_k[:, 1:])

        fvl, pvl = np.sqrt(fvl), np.sqrt(pvl)

        book["fvl_per_k"].append( fvl_per_k )
        book["pvl_per_k"].append( pvl_per_k )
        book["fvl"].append( fvl )
        book["pvl"].append( pvl )

        args.log.info("VALIDATION PARTIAL ERROR = %7f" %np.mean(book["pvl_per_k"][-1]))

        print("k ", end="")
        for name in dataset.variable_names:
            print("%s " %name.rjust(12), end="")
        print()
        for i in range(args.nb_missing):
            print("%d     " %(i+1), end="")
            for j in range(len(dataset.arch)):
                print("%f     " %book["pvl_per_k"][-1][i][j], end="")
            print()
        print()


    ############################################################################
    # log/plot #################################################################

    # ensure output dir exists
    now = get_date()
    d = os.path.join(args.output_path,now+"_train_"+config["DATASET"]["NAME"])
    if not os.path.exists(d):
        os.makedirs(d)

    epoch_axis = np.arange(0, config["MODEL"]["EPOCH"])

    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 18})
    #plt.rcParams['axes.titlepad'] = 15
    plt.subplots_adjust(bottom=0.15)
    #plt.subplots_adjust(top=0.88)

    ############################################################################
    # plot full training vs full validation (model can learn)

    if config["PLOT"]["FULL_ERROR"]:

        plt.plot(epoch_axis, book["ftl"], label="Training")
        plt.plot(epoch_axis, book["fvl"], label="Validation")

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend(loc='best')
        #plt.title("Full Error")

        plt.savefig(os.path.join(d,"full_training_vs_validation_Error.png"))
        plt.clf()


    ############################################################################
    # plot partial training vs partial validation (model can make predictions)

    if config["PLOT"]["PARTIAL_ERROR"]:
        
        plt.plot(epoch_axis, book["ptl"], label="Training")
        plt.plot(epoch_axis, book["pvl"], label="Validation")

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend(loc='best')
        #plt.title("Partial Error")

        plt.savefig(os.path.join(d,"partial_training_vs_validation_Error.png"))
        plt.clf()


    ############################################################################
    # partial training loss per k

    if config["PLOT"]["TRAINING_ERROR_PER_K"]:

        for i, name in enumerate(dataset.variable_names):

            for k in range(args.nb_missing):

                ptl_per_variable_per_k = [ book["ptl_per_k"][j][k][i] for j in range(config["MODEL"]["EPOCH"])]

                plt.plot(epoch_axis, ptl_per_variable_per_k,label=r"$\delta$"+"="+str(k+1))

            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.ylim(2, 5)
            plt.legend(loc='best')
            #plt.title("Partial Training Error per "+r"$\delta$"+" missing variables")

            plt.savefig(os.path.join(d,"partial_training_Error_per_k.png"))
            plt.clf()

    
    ############################################################################
    # partial validation loss per k

    if config["PLOT"]["VALIDATION_ERROR_PER_K"]:

        for i, name in enumerate(dataset.variable_names):

            for k in range(args.nb_missing):

                pvl_per_variable_per_k = [ book["pvl_per_k"][j][k][i] for j in range(config["MODEL"]["EPOCH"])]

                plt.plot(epoch_axis, pvl_per_variable_per_k,label=r"$\delta$"+"="+str(k+1))

            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.ylim(2, 5)
            plt.legend(loc='best')
            #plt.title("Partial Validation Error per "+r"$\delta$"+" missing variables")

            plt.savefig(os.path.join(d,"partial_validation_Error_per_k.png"))
            plt.clf()


    # export metric ############################################################

    args.log.info("Data saved in directory %s" %d)

    """
    with open(os.path.join(d, "metric_log.json"),'w+') as f:
        json.dump(metric_log, f)

    args.log.info("Data saved in directory %s" %d)
    """