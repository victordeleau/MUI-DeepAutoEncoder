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
    if not dataset_getter.local_dataset_found():
        logging.info( "No dataset found on disk, downloading ..." )
        dataset_getter.download_dataset(dataset_name=args.dataset_name)
    dataset = dataset_getter.load_local_dataset(view=args.view)
    
    training_and_validation_set, _ = dataset.get_split_sets(split_factor=0.9)
    training_set, validation_set = training_and_validation_set.get_split_sets(split_factor=0.8)
    
    nb_training_example = (training_set.nb_item if training_set.get_view() == "item_view" else training_set.nb_user)
    nb_validation_example = (validation_set.nb_item if validation_set.get_view() == "item_view" else validation_set.nb_user)
    #nb_testing_example = (testing_set.nb_item if testing_set.get_view() == "item_view" else testing_set.nb_user)
    nb_testing_example = None

    training_loader = dataset_getter.get_dataset_loader(
        training_set,
        batch_size=args.batch_size,
        nb_worker=4,
        shuffle=False
    )

    validation_loader = dataset_getter.get_dataset_loader(
        validation_set,
        batch_size=args.batch_size,
        nb_worker=4,
        shuffle=False
    )

    """testing_loader = dataset_getter.get_dataset_loader(
        testing_set,
        batch_size=args.batch_size,
        nb_worker=4,
        shuffle=False
    )"""

    my_base_dae = BaseDAE(
        io_size=dataset.get_io_size(),
        z_size=args.zsize,
        nb_input_layer=args.nb_layer,
        nb_output_layer=args.nb_layer
    )

    print("")
    logging.info("### LEARNING RATE = %f" %args.learning_rate)
    logging.info("### WEIGHT DECAY = %f" %args.regularization)
    logging.info("### EPOCH = %d" %args.nb_epoch)
    logging.info("### BATCH SIZE = %d" %args.batch_size)
    logging.info("### LAYERS = %d" %((args.nb_layer*2)+1))
    logging.info("### Z SIZE = %d" %args.zsize)
    logging.info("### IO SIZE = %s\n" %dataset.get_io_size())

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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam( my_base_dae.parameters(), lr=args.learning_rate, weight_decay=args.regularization )
    sum_training_loss, sum_validation_loss = 0.0, 0.0
    
    if args.redux != 1:
        training_start = math.floor( math.floor(nb_training_example/args.batch_size)
            - (args.redux * math.floor(nb_training_example/args.batch_size)) )
        validation_start = math.floor( math.floor(nb_validation_example/args.batch_size)
            - (args.redux * math.floor(nb_validation_example/args.batch_size)) )
    else:
        training_start, validation_start = 0, 0
    
    for epoch in range(args.nb_epoch):

        my_base_dae.train()
        training_iter_nb, sum_training_loss = 0, 0

        for i, input_data in enumerate(training_loader, training_start):

            input_data = Variable( input_data.float() )

            output_data = my_base_dae( input_data )

            loss = criterion( output_data, input_data )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            sum_training_loss += loss.data

            training_iter_nb += args.batch_size

        my_base_dae.eval()
        validation_iter_nb, sum_validation_loss = 0, 0

        for i, input_data in enumerate(validation_loader, validation_start):

            input_data = Variable( input_data.float() )

            output_data = my_base_dae( input_data )

            loss = criterion( output_data, input_data )

            sum_validation_loss += loss.data
            validation_iter_nb += args.batch_size

        logging.info('epoch [{}/{}], training loss:{:.6f}, validation loss:{:.6f}'.format(
            epoch + 1,
            args.nb_epoch,
            sum_training_loss/training_iter_nb,
            sum_validation_loss/validation_iter_nb)
        )

        sum_training_loss, sum_validation_loss = 0, 0
        training_iter_nb, validation_iter_nb = 0, 0

    #torch.save(model.state_dict(), './sim_autoencoder.pth')