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

from tool.logging import set_logging, display_info
from tool.metering import get_object_size, get_rmse, LossAnalyzer
from tool.parser import parse
from dataset.dataset_getter import DatasetGetter
from model.base_dae import BaseDAE


if __name__ == "__main__":

    args = parse()
    log = set_logging(logging_level=(logging.DEBUG if args.debug else logging.INFO))
    log.info("Mixed User Item Deep Auto Encoder (MUI-DAE) demo")

    dataset_getter = DatasetGetter()

    if not dataset_getter.local_dataset_found() or args.reload_dataset:
        log.info( "Downloading dataset " + args.dataset )
        dataset_getter.download_dataset(dataset_name=args.dataset)

    dataset = dataset_getter.load_local_dataset(view=args.view, try_load_binary=not args.reload_dataset)

    if args.normalize: dataset.normalize()
    
    training_and_validation_dataset, _ = dataset.get_split_sets(split_factor=0.9, view=args.view)
    training_dataset, validation_dataset = training_and_validation_dataset.get_split_sets(split_factor=0.8, view=args.view)
    
    nb_training_example = (training_dataset.nb_item if training_dataset.get_view() == "item" else training_dataset.nb_user)
    nb_validation_example = (validation_dataset.nb_item if validation_dataset.get_view() == "item" else validation_dataset.nb_user)
    #nb_testing_example = (testing_set.nb_item if testing_set.get_view() == "item_view" else testing_set.nb_user)
    nb_testing_example = None

    my_base_dae = BaseDAE(
        io_size=dataset.get_io_size(),
        z_size=args.zsize,
        nb_input_layer=args.nb_layer,
        nb_output_layer=args.nb_layer)

    display_info(args, dataset)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info("Cuda available, loading GPU device")
    else:
        logging.info("No Cuda device available, using CPU") 
    device = torch.device("cuda:0" if use_gpu else "cpu")

    optimizer = torch.optim.Adam( my_base_dae.parameters(), lr=args.learning_rate, weight_decay=args.regularization )
    
    nb_training_sample_to_process = math.floor(args.redux * nb_training_example)
    nb_validation_sample_to_process = math.floor(args.redux * nb_validation_example)
    #nb_testing_sample = math.floor(args.redux * nb_testing_example)

    nb_training_iter = math.ceil(nb_training_sample_to_process / args.batch_size)
    nb_validation_iter = math.ceil(nb_validation_sample_to_process / args.batch_size)
    #nb_testing_iter = np.ceil(nb_testing_sample / args.batch_size)

    log.info("Training has started.")

    torch.set_printoptions(threshold=10000)

    loss_analyzer = LossAnalyzer(args.max_increasing_cnt, args.max_nan_cnt)
    total_time_start = time.time()

    for epoch in range(args.nb_epoch):

        sum_training_loss, sum_validation_loss = 0, 0
        increasing_cnt = 0, 0
        epoch_time_start = time.time()

        my_base_dae.to(device)
        my_base_dae.train()

        for i in range(nb_training_iter):

            if args.batch_size == 1:
                training_batch, remaining = training_dataset[i], 1

            else:
                remaining = (args.batch_size 
                    if (i+1)*args.batch_size < nb_training_sample_to_process
                    else nb_training_sample_to_process-(i*args.batch_size) )
                training_batch = np.stack([training_dataset[i*args.batch_size+j] for j in range(remaining)])

            input_data = Variable(torch.Tensor(np.squeeze(training_batch))).to(device)

            output_data = my_base_dae(input_data)

            mmse_loss = my_base_dae.get_mmse_loss(input_data, output_data)

            sum_training_loss += (mmse_loss.item() if not loss_analyzer.is_nan(mmse_loss.item()) else 0)
                
            optimizer.zero_grad()

            mmse_loss.backward()

            optimizer.step()

            log.debug("Training loss %0.6f" %( math.sqrt(mmse_loss.item()) ) )

        my_base_dae.eval()
        sum_validation_loss = 0

        for i in range(nb_validation_iter):

            if args.batch_size == 1:
                validation_batch, remaining = validation_dataset[i], 1

            else:
                remaining = (args.batch_size 
                    if (i+1)*args.batch_size < nb_validation_sample_to_process
                    else nb_validation_sample_to_process-(i*args.batch_size) )    
                validation_batch = np.stack([validation_dataset[i*args.batch_size+j] for j in range(remaining)])

            input_data = Variable(torch.Tensor(np.squeeze(validation_batch))).to(device)

            output_data = my_base_dae(input_data)

            mmse_loss = my_base_dae.get_mmse_loss(input_data, output_data)

            sum_validation_loss += (mmse_loss.item() if not loss_analyzer.is_nan(mmse_loss.item()) else 0)

            log.debug("Validation loss %0.6f" %( math.sqrt( mmse_loss.item() ) ) )

        training_rmse = math.sqrt(sum_training_loss/nb_training_iter)
        validation_rmse = math.sqrt(sum_validation_loss/nb_validation_iter)

        log.info('epoch [{}/{}], training rmse:{:.6f}, validation rmse:{:.6f}, time:{:0.2f}s'.format(
            epoch + 1,
            args.nb_epoch,
            training_rmse,
            validation_rmse,
            time.time() - epoch_time_start))

        if loss_analyzer.is_minimum(validation_rmse):
            log.info("Optimum detected with validation rmse %0.6f at epoch %d" %(loss_analyzer.previous_losses[-1], epoch+1-args.max_increasing_cnt))
            break

    logging.info("Total training time of %0.2f seconds" %(time.time() - total_time_start) )

    #torch.save(model.state_dict(), './sim_autoencoder.pth')