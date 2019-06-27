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

from tool.logging import set_logging
from tool.metering import get_object_size, get_rmse
from tool.parser import parse
from dataset.dataset_getter import DatasetGetter
from model.base_dae import BaseDAE


if __name__ == "__main__":

    args = parse()
    logging = set_logging()
    logging.info("Mixed User Item Deep Auto Encoder (MUI-DAE) demo")

    dataset_getter = DatasetGetter()
    if not dataset_getter.local_dataset_found() or args.reload_dataset:
        logging.info( "Downloading dataset " + args.dataset )
        dataset_getter.download_dataset(dataset_name=args.dataset)
    dataset = dataset_getter.load_local_dataset(view=args.view, try_load_binary=not args.reload_dataset)
    
    training_and_validation_dataset, _ = dataset.get_split_sets(split_factor=0.9)
    training_dataset, validation_dataset = training_and_validation_dataset.get_split_sets(split_factor=0.8)
    
    nb_training_example = (training_dataset.nb_item if training_dataset.get_view() == "item_view" else training_dataset.nb_user)
    nb_validation_example = (validation_dataset.nb_item if validation_dataset.get_view() == "item_view" else validation_dataset.nb_user)
    #nb_testing_example = (testing_set.nb_item if testing_set.get_view() == "item_view" else testing_set.nb_user)
    nb_testing_example = None

    """training_loader = dataset_getter.get_dataset_loader(
        training_set,
        batch_size=args.batch_size,
        nb_worker=3,
        shuffle=False)

    validation_loader = dataset_getter.get_dataset_loader(
        validation_set,
        batch_size=args.batch_size,
        nb_worker=3,
        shuffle=False)

    testing_loader = dataset_getter.get_dataset_loader(
        testing_set,
        batch_size=args.batch_size,
        nb_worker=3,
        shuffle=False)"""

    my_base_dae = BaseDAE(
        io_size=dataset.get_io_size(),
        z_size=args.zsize,
        nb_input_layer=args.nb_layer,
        nb_output_layer=args.nb_layer)

    print("")
    logging.info("### LEARNING RATE = %f" %args.learning_rate)
    logging.info("### WEIGHT DECAY = %f" %args.regularization)
    logging.info("### EPOCH = %d" %args.nb_epoch)
    logging.info("### BATCH SIZE = %d" %args.batch_size)
    logging.info("### LAYERS = %d" %((args.nb_layer*2)+1))
    logging.info("### Z SIZE = %d" %args.zsize)
    logging.info("### IO SIZE = %s\n" %dataset.get_io_size())

    logging.info("### DATASET NAME = %s" %args.dataset)
    logging.info("### NB USER = %s" %dataset.nb_user)
    logging.info("### NB ITEM = %s" %dataset.nb_item)
    logging.info("### DATASET NAME = %s" %args.dataset)
    logging.info("### DATASET SIZE FACTOR = %f" %args.redux)
    logging.info("### DATASET VIEW = %s" %args.view)
    logging.info("### REDUX FACTOR = %s" %args.redux)
    logging.info("### SUB TRAINING OBSERVATION = %s" %nb_training_example)
    logging.info("### SUB VALIDATION OBSERVATION = %s" %nb_validation_example)
    logging.info("### SUB TESTING OBSERVATION = %s\n" %nb_testing_example)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Cuda available, loading GPU device")
    else:
        device = torch.device('cpu')
        logging.info("No Cuda device available, using CPU")    

    logging.info("Training has started.")

    optimizer = torch.optim.Adam( my_base_dae.parameters(), lr=args.learning_rate, weight_decay=args.regularization )
    sum_training_loss, sum_validation_loss = 0.0, 0.0

    nb_training_sample = math.floor(args.redux * nb_training_example)
    nb_validation_sample = math.floor(args.redux * nb_validation_example)
    
    for epoch in range(args.nb_epoch):

        my_base_dae.train()
        training_iter_nb, sum_training_loss = 0, 0

        for i in range(nb_training_sample):

            input_data = Variable( torch.Tensor( training_dataset[i] ) )

            output_data = my_base_dae( input_data )

            mmse_loss = my_base_dae.get_mmse_loss(input_data, output_data)

            optimizer.zero_grad()

            mmse_loss.backward()

            optimizer.step()

            sum_training_loss += mmse_loss.item()
            training_iter_nb += 1

        my_base_dae.eval()
        validation_iter_nb, sum_validation_loss = 0, 0

        for i in range(nb_validation_sample):

            input_data = Variable( torch.Tensor( validation_dataset[i] ) )

            output_data = my_base_dae( input_data )

            mmse_loss = my_base_dae.get_mmse_loss(input_data, output_data)

            sum_validation_loss += mmse_loss.item()
            validation_iter_nb += 1

        logging.info('epoch [{}/{}], training rmse:{:.6f}, validation rmse:{:.6f}'.format(
            epoch + 1,
            args.nb_epoch,
            math.sqrt(sum_training_loss/training_iter_nb),
            math.sqrt(sum_validation_loss/validation_iter_nb))
        )

        sum_training_loss, sum_validation_loss = 0, 0
        training_iter_nb, validation_iter_nb = 0, 0

    #torch.save(model.state_dict(), './sim_autoencoder.pth')