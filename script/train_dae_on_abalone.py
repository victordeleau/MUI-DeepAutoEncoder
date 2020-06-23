
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
    criterion = CombinedCriterion(
        arch=dataset.arch,
        k_max=args.nb_missing,
        device=torch.device(device),
        observation_mask=dataset.type_mask)

    lm = LossManager(device=torch.device(device))

    lm.add_book("ftl", (config["MODEL"]["BATCH_SIZE"], dataset.nb_predictor))
    lm.add_book("ptl", (config["MODEL"]["BATCH_SIZE"], dataset.nb_predictor))
    lm.add_book("fvl", (config["MODEL"]["BATCH_SIZE"], dataset.nb_predictor))
    lm.add_book("pvl", (config["MODEL"]["BATCH_SIZE"], dataset.nb_predictor))

    lm.add_book("ftl_k", (args.nb_missing, dataset.nb_predictor))
    lm.add_book("ptl_k", (args.nb_missing, dataset.nb_predictor))
    lm.add_book("fvl_k", (args.nb_missing, dataset.nb_predictor))
    lm.add_book("pvl_k", (args.nb_missing, dataset.nb_predictor))


    ############################################################################
    # training #################################################################

    for epoch in range( config["MODEL"]["EPOCH"] ):

        args.log.info("===================================================== EPOCH = %d\n" %epoch)

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
                ftl = criterion(x=input_data, y=output_data)

                # backpropagate training loss
                optimizer.zero_grad()
                torch.mean(ftl).backward()
                optimizer.step()

                ftl = ftl.clone().cpu().detach().numpy()

                # full training loss per k
                ftl_per_k = criterion.get_per_k(ftl, masks)

                # extract partial training loss
                ptl = criterion.get_partial(ftl, fmask)
                # extract partial training loss per k
                ptl_per_k = criterion.get_per_k(ptl, masks)

                lm.get_book("ptl").add(ptl)
                lm.get_book("ftl").add(ftl)
                lm.get_book("ptl_k").add(ptl_per_k)
                lm.get_book("ftl_k").add(ftl_per_k)

        lm.get_book("ftl").loss = np.sum( lm.get_book("ftl").loss, axis=0, keepdims=True )/(nb_train*dataset.nb_predictor)

        lm.get_book("ptl").loss = np.sum( lm.get_book("ptl").loss, axis=0, keepdims=True )/(nb_train)

        lm.get_book("ftl").loss[:, 1:] = normalizer.inverse_transform(lm.get_book("ftl").loss[:, 1:])
        lm.get_book("ptl").loss[:, 1:] = normalizer.inverse_transform(lm.get_book("ptl").loss[:, 1:])

        lm.log_book("ptl").log_book("ftl").log_book("ftl_k").log_book("ptl_k")

        # erase loss buffer
        lm.get_book("ftl").purge()
        lm.get_book("ptl").purge()
        lm.get_book("ftl_k").purge()
        lm.get_book("ptl_k").purge()

        #args.log.info("TRAINING FULL RMSE = %7f" %lm.get_log("ftl")[-1])
        args.log.info("TRAINING PARTIAL RMSE = %7f" %np.sum(lm.get_log("ftl")[-1]))

        print("  ", end="")
        for name in dataset.variable_names:
            print("%s " %name.rjust(12), end="")
        print()
        for i in range(args.nb_missing):
            print("%d " %(i+1), end="")
            for j in range(len(dataset.arch)):
                print("%f " %lm.get_log("ptl")[-1][i][j], end="")
            print()
        print()

        # validation ###########################################################

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

                # compute the global validation loss
                fvl = criterion(x=input_data, y=output_data)

                fvl = fvl.clone().cpu().detach().numpy()

                # full validation loss per k
                fvl_per_k = criterion.get_per_k(fvl, masks)

                # extract partial validation loss
                pvl = criterion.get_partial(fvl, fmask)
                # extract partial validation loss per k
                pvl_per_k = criterion.get_per_k(pvl, masks)

                lm.get_book("pvl").add(pvl)
                lm.get_book("fvl").add(fvl)
                lm.get_book("pvl_k").add(pvl_per_k)
                lm.get_book("fvl_k").add(fvl_per_k)

        lm.get_book("fvl").loss = np.sum( lm.get_book("fvl").loss, axis=0, keepdims=True )/(nb_validation*dataset.nb_predictor)
        lm.get_book("pvl").loss = np.sum( lm.get_book("pvl").loss, axis=0, keepdims=True )/(nb_validation)

        lm.get_book("fvl").loss[:, 1:] = normalizer.inverse_transform(lm.get_book("fvl").loss[:, 1:])
        lm.get_book("pvl").loss[:, 1:] = normalizer.inverse_transform(lm.get_book("pvl").loss[:, 1:])

        lm.log_book("pvl").log_book("fvl").log_book("fvl_k").log_book("pvl_k")

        # erase loss buffer
        lm.get_book("fvl").purge()
        lm.get_book("pvl").purge()
        lm.get_book("fvl_k").purge()
        lm.get_book("pvl_k").purge()

        args.log.info("VALIDATION PARTIAL RMSE = %7f" %np.sum(lm.get_log("fvl")[-1]))

        print("  ", end="")
        for name in dataset.variable_names:
            print("%s " %name.rjust(12), end="")
        print()
        for i in range(args.nb_missing):
            print("%d " %(i+1), end="")
            for j in range(len(dataset.arch)):
                print("%f " %lm.get_log("pvl")[-1][i][j], end="")
            print()
        print()

    args.log.info("TRAINING HAS ENDED.")


    ############################################################################
    # log/plot #################################################################

    # ensure output dir exists
    now = get_date()
    d = os.path.join(args.output_path,now+"_train_"+config["DATASET"]["NAME"])
    if not os.path.exists(d):
        os.makedirs(d)

    epoch_axis = np.arange(0, config["MODEL"]["EPOCH"])


    ############################################################################
    # plot sum full training/validation RMSE (model learns to reconstruct input)

    sum_ftl = [ lm.get_log("ftl")[j] for j in range(config["MODEL"]["EPOCH"]) ]

    sum_fvl = [ lm.get_log("fvl")[j] for j in range(config["MODEL"]["EPOCH"]) ]

    plt.plot(epoch_axis, sum_ftl, label="Sum full training RMSE")
    plt.plot(epoch_axis, sum_fvl, label="Sum full validation RMSE")

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.title("Sum of full RMSE, training vs validation")

    plt.savefig(os.path.join(d,"sum_full_RMSE.png"))
    plt.clf()


    ############################################################################
    # plot sum partial training/validation RMSE (model can make predictions)

    sum_ptl = [ np.sum(lm.get_log("ptl")[:][j]) for j in range(config["MODEL"]["EPOCH"])]

    sum_pvl = [ np.sum(lm.get_log("pvl")[:][j]) for j in range(config["MODEL"]["EPOCH"])]

    plt.plot(epoch_axis, sum_ptl, label="Sum partial training RMSE")
    plt.plot(epoch_axis, sum_pvl, label="Sum partial validation RMSE")

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.title("Sum of partial RMSE, training vs validation")

    plt.savefig(os.path.join(d,"sum_partial_RMSE.png"))
    plt.clf()


    ############################################################################
    # plot detail partial RMSE  (ring/ prediction on par with best results)

    for i, name in enumerate(dataset.variable_names):

        ptl_per_variable = [ np.sum(lm.get_log("ptl")[j]) for j in range(config["MODEL"]["EPOCH"])]

        pvl_per_variable = [ np.sum(lm.get_log("pvl")[j]) for j in range(config["MODEL"]["EPOCH"])]

        plt.plot(epoch_axis, ptl_per_variable,
            label="Partial training RMSE")
        plt.plot(epoch_axis, pvl_per_variable,
            label="Partial validation RMSE")

        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend(loc='best')
        plt.title(name + " prediction")

        plt.savefig(os.path.join(d,"partial_RMSE_" + name + ".png"))
        plt.clf()


    # export metric ############################################################

    """
    with open(os.path.join(d, "metric_log.json"),'w+') as f:
        json.dump(metric_log, f)

    args.log.info("Data saved in directory %s" %d)
    """