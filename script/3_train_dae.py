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

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np

from codea.tool import set_logging, display_info
from codea.tool import get_object_size, get_rmse, LossAnalyzer, PlotDrawer, export_parameters_to_json
from codea.tool import parse
from codea.tool import get_day_month_year_hour_minute_second
from codea.tool import BatchBuilder
from codea.dataset import DatasetGetter
from codea.model import Autoencoder


def parse():

    parser = argparse.ArgumentParser(
        description='Train denoising autoencoder.')

    parser.add_argument('--embedding_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    return parser.parse_args()


def train_dae(args, output_dir):

    # load dataset of embeddings

    # normalize the dataset

    # load model
    model = Autoencoder(
        io_size=io_size,
        z_size=args.z_size,
        nb_input_layer=args.nb_layer,
        nb_output_layer=args.nb_layer,
        steep_layer_size=True) 

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        args.log.info("Cuda available, loading GPU device")
    else:
        args.log.info("No Cuda device available, using CPU") 
    device = torch.device("cuda:0" if use_gpu else "cpu")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.regularization)


    ############################################################################

    dataset_getter = DatasetGetter()

    if not dataset_getter.local_dataset_found() or args.reload_dataset:
        args.log.info( "Downloading dataset " + args.dataset )
        dataset_getter.download_dataset(dataset_name=args.dataset)

    dataset = dataset_getter.load_local_dataset(view=args.view, try_load_binary=not args.reload_dataset)

    training_dataset, tmp_dataset = dataset.split(0.6)
    validation_dataset, testing_dataset = tmp_dataset.split(0.5)

    if args.normalize:
        training_dataset.mean_normalize()
        validation_dataset.mean_normalize()
        testing_dataset.mean_normalize()
    
    nb_example = (dataset.nb_item if dataset.get_view() == "item" else dataset.nb_user)

    my_base_dae = BaseDAE(
        io_size=dataset.get_io_size(),
        z_size=args.zsize,
        nb_input_layer=args.nb_layer,
        nb_output_layer=args.nb_layer)

    display_info(args, dataset)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        args.log.info("Cuda available, loading GPU device")
    else:
        args.log.info("No Cuda device available, using CPU") 
    device = torch.device("cuda:0" if use_gpu else "cpu")

    optimizer = torch.optim.Adam( my_base_dae.parameters(), lr=args.learning_rate, weight_decay=args.regularization )
    
    nb_sample_to_process = math.floor(args.redux * nb_example)

    nb_iter = math.ceil(nb_sample_to_process / args.batch_size)

    loss_analyzer = LossAnalyzer(args.max_increasing_cnt, args.max_nan_cnt)

    plot_drawer = PlotDrawer()

    batch_builder = BatchBuilder(dataset.get_io_size(), [0.8, 0.1, 0.1], 128)


    ############################################################################
    # training #################################################################

    training_time_start = time.time()
    training_rmses, validation_rmses = [], []

    for epoch in range(args.nb_epoch):

        training_loss, validation_loss = 0, 0
        increasing_cnt, loss_cnt = 0, 0
        epoch_time_start = time.time()

        my_base_dae.to(device)
        my_base_dae.train()

        for i in range(nb_iter): 

            training_batch, remaining = batch_builder.get_batches( training_dataset, args.batch_size, nb_sample_to_process, i )
            validation_batch, _ = batch_builder.get_batches( validation_dataset, args.batch_size, nb_sample_to_process, i )

            if np.count_nonzero(training_batch) == 0:
                #print("nop")
                continue

            #print( np.count_nonzero(training_batch, 1) )

            input_data = Variable(torch.Tensor(np.squeeze(training_batch))).to(device)
            output_data_to_compare = Variable(torch.Tensor(np.squeeze(validation_batch))).to(device)

            output_data = my_base_dae(input_data)

            training_mmse_loss = my_base_dae.get_mmse_loss(input_data, output_data)
            validation_mmse_loss = my_base_dae.get_mmse_loss(output_data_to_compare, output_data)

            if not loss_analyzer.is_nan(training_mmse_loss.item()) and not loss_analyzer.is_nan(validation_mmse_loss.item()):
                training_loss += training_mmse_loss.item()
                validation_loss += validation_mmse_loss.item()
                loss_cnt += 1
                
            optimizer.zero_grad()

            training_mmse_loss.backward()

            optimizer.step()

            input_data.detach_()

            args.log.debug("Training loss %0.6f" %( math.sqrt(training_mmse_loss.item()) ) )

        training_rmses.append( math.sqrt(training_loss/ loss_cnt) )
        validation_rmses.append( math.sqrt(validation_loss/ loss_cnt ) )

        args.log.info('epoch [{:3d}/{:3d}], training rmse:{:.6f}, validation rmse:{:.6f}, time:{:0.2f}s'.format(
            epoch + 1,
            args.nb_epoch,
            training_rmses[-1],
            validation_rmses[-1],
            time.time() - epoch_time_start))

        if loss_analyzer.is_minimum(validation_rmses[-1]):
            args.log.info("Optimum detected with validation rmse %0.6f at epoch %d" %(loss_analyzer.previous_losses[-1], epoch+1-args.max_increasing_cnt))
            break


    ############################################################################
    # testing ##################################################################

    training_loss, testing_loss, loss_cnt = 0, 0, 0
    testing_time_start = time.time()

    my_base_dae.to(device)
    my_base_dae.eval()

    for i in range(nb_iter):

        training_batch, remaining = batch_builder.get_batches( training_dataset, args.batch_size, nb_sample_to_process, i )
        testing_batch, _ = batch_builder.get_batches( testing_dataset, args.batch_size, nb_sample_to_process, i )

        input_data = Variable(torch.Tensor(np.squeeze(training_batch))).to(device)
        output_data_to_compare = Variable(torch.Tensor(np.squeeze(testing_batch))).to(device)

        output_data = my_base_dae(input_data)

        training_mmse_loss = my_base_dae.get_mmse_loss(input_data, output_data)
        testing_mmse_loss = my_base_dae.get_mmse_loss(output_data_to_compare, output_data)

        if not loss_analyzer.is_nan(training_mmse_loss.item()) and not loss_analyzer.is_nan(testing_mmse_loss.item()):
                training_loss += training_mmse_loss.item()
                testing_loss += testing_mmse_loss.item()
                loss_cnt += 1

        input_data.detach_()

    logging.info("training rmse:{:.6f}, testing rmse:{:.6f}, testing time of {:0.2f}s".format(
        math.sqrt(training_loss/loss_cnt),
        math.sqrt(testing_loss/loss_cnt),
        (time.time() - testing_time_start)))

    args.log.info("Testing has ended.\n")


    # plotting and saving ##############################

    plot_drawer.add( data=[ training_rmses, validation_rmses ], title="RMSE", legend=["training rmse", "validation_rmse"], display=True )
    plot_drawer.export_to_png(idx = 0, export_path=output_dir)
    export_parameters_to_json(args, output_dir)



if __name__ == "__main__":

    args = parse()

    vars(args)["log"] = set_logging(logging_level=(logging.DEBUG if args.debug else logging.INFO))

    args.log.info("Training has started.")

    output_dir = "../out/autoencoder_training_" + get_day_month_year_hour_minute_second()
    
    train_autoencoder(args, output_dir)