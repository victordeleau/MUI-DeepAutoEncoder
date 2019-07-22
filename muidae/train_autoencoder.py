# Mixed User Item Deep Auto Encoder

import sys
import time
import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable
import gc
import psutil
import logging
import time
import json

from tool.logger import set_logging, display_info
from tool.metering import get_object_size, get_rmse, LossAnalyzer, PlotDrawer, export_parameters_to_json
from tool.parser import parse
from tool.date import get_day_month_year_hour_minute_second
from tool.data_tool import BatchBuilder
from dataset.dataset_getter import DatasetGetter
from model.base_dae import BaseDAE


def train_autoencoder(args, output_dir):

    dataset_getter = DatasetGetter()

    if not dataset_getter.local_dataset_found() or args.reload_dataset:
        args.log.info( "Downloading dataset " + args.dataset )
        dataset_getter.download_dataset(dataset_name=args.dataset)

    dataset = dataset_getter.load_local_dataset(view=args.view, try_load_binary=not args.reload_dataset)

    if args.normalize: dataset.normalize()
    
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

    nb_iter = math.floor(nb_sample_to_process / args.batch_size)

    loss_analyzer = LossAnalyzer(args.max_increasing_cnt, args.max_nan_cnt)

    plot_drawer = PlotDrawer()

    batch_builder = BatchBuilder(dataset.get_io_size(), [0.6, 0.2, 0.2], 16)


    # training ##############################

    training_time_start = time.time()
    training_rmses, validation_rmses = [], []

    for epoch in range(args.nb_epoch):

        training_loss, validation_loss = 0, 0
        increasing_cnt = 0, 0
        epoch_time_start = time.time()

        my_base_dae.to(device)
        my_base_dae.train()

        for i in range(nb_iter): 

            bro = batch_builder.get_batches( dataset, args.batch_size, nb_sample_to_process, i )

            training_batch, validation_batch, _, remaining = bro

            input_data = Variable(torch.Tensor(np.squeeze(training_batch))).to(device)
            output_data_to_compare = Variable(torch.Tensor(np.squeeze(validation_batch))).to(device)

            output_data = my_base_dae(input_data)

            training_mmse_loss = my_base_dae.get_mmse_loss(input_data, output_data)
            validation_mmse_loss = my_base_dae.get_mmse_loss(output_data_to_compare, output_data)

            training_loss += (training_mmse_loss.item() if not loss_analyzer.is_nan(training_mmse_loss.item()) else 0)
            validation_loss += (validation_mmse_loss.item() if not loss_analyzer.is_nan(validation_mmse_loss.item()) else 0)
                
            optimizer.zero_grad()

            training_mmse_loss.backward()

            optimizer.step()

            input_data.detach_()

            args.log.debug("Training loss %0.6f" %( math.sqrt(training_mmse_loss.item()) ) )

        training_rmses.append( math.sqrt(training_loss/nb_iter) )
        validation_rmses.append( math.sqrt(validation_loss/nb_iter) )

        args.log.info('epoch [{:3d}/{:3d}], training rmse:{:.6f}, validation rmse:{:.6f}, time:{:0.2f}s'.format(
            epoch + 1,
            args.nb_epoch,
            training_rmses[-1],
            validation_rmses[-1],
            time.time() - epoch_time_start))

        if loss_analyzer.is_minimum(validation_rmses[-1]):
            args.log.info("Optimum detected with validation rmse %0.6f at epoch %d" %(loss_analyzer.previous_losses[-1], epoch+1-args.max_increasing_cnt))
            break


    # testing ##############################

    training_loss, testing_loss = 0, 0
    testing_time_start = time.time()

    my_base_dae.to(device)
    my_base_dae.eval()

    for i in range(nb_iter):

        training_batch, _, testing_batch, remaining = batch_builder.get_batches( dataset, args.batch_size, nb_sample_to_process, i )

        input_data = Variable(torch.Tensor(np.squeeze(training_batch))).to(device)
        output_data_to_compare = Variable(torch.Tensor(np.squeeze(testing_batch))).to(device)

        output_data = my_base_dae(input_data)

        training_mmse_loss = my_base_dae.get_mmse_loss(input_data, output_data)
        testing_mmse_loss = my_base_dae.get_mmse_loss(output_data_to_compare, output_data)

        training_loss += (training_mmse_loss.item() if not loss_analyzer.is_nan(training_mmse_loss.item()) else 0)
        testing_loss += (testing_mmse_loss.item() if not loss_analyzer.is_nan(testing_mmse_loss.item()) else 0)

        input_data.detach_()

    logging.info("training rmse:{:.6f}, testing rmse:{:.6f}, testing time of %0.2f seconds".format(
        math.sqrt(training_loss/nb_iter),
        math.sqrt(testing_loss/nb_iter),
        (time.time() - testing_time_start)))

    args.log.info("Testing has ended.\n")


    # plotting and saving ##############################

    plot_drawer.add( data=[ training_rmses, validation_rmses ], title="RMSE", legend=["training rmse", "validation_rmse"], display=True )
    plot_drawer.export_to_png(idx = 0, export_path=output_dir)
    export_parameters_to_json(args, output_dir)



if __name__ == "__main__":

    args = parse()

    vars(args)["log"] = set_logging(logging_level=(logging.DEBUG if args.debug else logging.INFO))

    args.log.info("Autoencoder trainer has started.")

    output_dir = "../out/autoencoder_training_" + get_day_month_year_hour_minute_second()
    
    train_autoencoder(args, output_dir)