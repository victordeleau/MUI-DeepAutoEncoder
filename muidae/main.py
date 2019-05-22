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

from tool.logging import set_logging
from dataset.dataset_getter import DatasetGetter
from model.base_dae import BaseDAE

if __name__ == "__main__":

    LEARNING_RATE = 5e-5
    DECAY = 0
    NB_EPOCH = 50
    Z_SIZE = 64
    BATCH_SIZE = 2
    REDUX = 0.8
    NB_LAYER = 2

    logging = set_logging()
    logging.info("Mixed User Item Deep Auto Encoder (MUI-DAE) demo")

    dataset = None
    dataset_getter = DatasetGetter()

    if not dataset_getter.local_dataset_found():
        logging.info( "No dataset found on disk, downloading ..." )
        dataset_getter.download_dataset(dataset_name="100k")

    dataset = dataset_getter.load_local_dataset()
    dataset.normalize(global_mean=True, user_mean=True, item_mean=True)

    io_size = (dataset.nb_item+1 if dataset.get_view() == "user_view" else dataset.nb_user+1 )

    logging.info("Dataset view -> %s" %dataset.get_view() )
    logging.info("Input size -> %s, Z size -> %s" %(io_size, Z_SIZE) )

    training_and_validation_set, _ = dataset.get_split_sets(split_factor=0.9)
    training_set, validation_set = training_and_validation_set.get_split_sets(split_factor=0.8)

    #print(dataset.nb_user, dataset.nb_item)
    #print(training_and_validation_set.nb_user, training_and_validation_set.nb_item)
    #print(testing_set.nb_user, testing_set.nb_item)

    #print(training_set.nb_user, training_set.nb_item)
    #print(validation_set.nb_user, validation_set.nb_item)

    training_loader = dataset_getter.get_dataset_loader(
        training_set,
        redux=REDUX,
        batch_size=BATCH_SIZE,
        nb_worker=4,
        shuffle=False
    )

    validation_loader = dataset_getter.get_dataset_loader(
        validation_set,
        redux=REDUX,
        batch_size=BATCH_SIZE,
        nb_worker=4,
        shuffle=False
    )

    """testing_loader = dataset_getter.get_dataset_loader(
        testing_set,
        redux=REDUX,
        batch_size=BATCH_SIZE,
        nb_worker=4,
        shuffle=False
    )"""

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Cuda available, loading GPU device")
    else:
        device = torch.device('cpu')
        logging.info("No Cuda device available, using CPU")

    

    training_example = (training_set.nb_item if training_set.get_view() == "item_view" else training_set.nb_user)
    validation_example = (validation_set.nb_item if validation_set.get_view() == "item_view" else validation_set.nb_user)
    #testing_example = (testing_set.nb_item if testing_set.get_view() == "item_view" else testing_set.nb_user)

    my_base_dae = BaseDAE( io_size=io_size, z_size=Z_SIZE, nb_input_layer=NB_LAYER, nb_output_layer=NB_LAYER ).double()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam( my_base_dae.parameters(), lr=LEARNING_RATE, weight_decay=DECAY )

    sum_training_loss, sum_validation_loss = 0.0, 0.0

    logging.info("Training has started.")
    
    for epoch in range(NB_EPOCH):

        my_base_dae.train()
        training_iter_nb, sum_training_loss = 1, 0

        training_start = math.floor( REDUX * (training_example/BATCH_SIZE) )

        for i, training_batch in enumerate(training_loader, training_start):

            input_data = Variable( training_batch )

            output_data = my_base_dae( input_data )

            loss = criterion( output_data, input_data )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_training_loss += loss.item()
            training_iter_nb += BATCH_SIZE

        my_base_dae.eval()
        validation_iter_nb, sum_validation_loss = 0, 0

        validation_start = math.floor( REDUX * (validation_example/BATCH_SIZE) )

        for i, validation_batch in enumerate(validation_loader, validation_start):

            input_data = Variable( validation_batch )

            output_data = my_base_dae( input_data )

            loss = criterion( output_data, input_data )

            sum_validation_loss += loss.item()
            validation_iter_nb += BATCH_SIZE

        logging.info('epoch [{}/{}], training loss:{:.4f}, validation loss:{:.4f}'.format(
            epoch + 1,
            NB_EPOCH,
            sum_training_loss/training_iter_nb,
            sum_validation_loss/validation_iter_nb)
        )

        sum_training_loss, sum_validation_loss = 0, 0
        training_iter_nb, validation_iter_nb = 0, 0

    #torch.save(model.state_dict(), './sim_autoencoder.pth')

