# Mixed User Item Deep Auto Encoder

import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable

from tool.logging import set_logging
from dataset.dataset_getter import DatasetGetter
from model.base_dae import BaseDAE

"""
from dataset.dataset_getter import dataset_getter
from dataset.dataset import Dataset

from model.algo_base_deep import AlgoBaseDeep
from model.autoencoder import AutoEncoder
"""

if __name__ == "__main__":

    # setup #############################################################################

    logging = set_logging()

    logging.info("Mixed User Item Deep Auto Encoder (MUI-DAE) demo")

    # parse input cmd argument and create dictionnary of parameters

    # download dataset (if necessary)

    # dataset #############################################################################

    dataset = None
    dg = DatasetGetter()

    if dg.local_dataset_found() and sys.argv[1] != "0":

        logging.info( "Dataset found on disk, loading ..." )
        dataset = dg.load_local_dataset()
        logging.info( "---> loading OK" )

    else:

        logging.info( "No dataset found on disk, downloading to disk ..." )
        dataset = dg.get("100k")
        dataset.normalize(global_mean=True, user_mean=True, item_mean=True)

        logging.info( "Writing dataset object to disk ..." )
        dg.export_dataset(dataset)
        logging.info( "---> writing OK" )

    training_and_validation_set, testing_set = dataset.get_split_sets(split_factor=0.9)

    training_set, validation_set = training_and_validation_set.get_split_sets(split_factor=0.8)

    # model #############################################################################

    learning_rate = 1e-4
    weight_decay = 0
    nb_epoch = 50
    z_size = 1024
    batch_size = 64

    # check which computing device should be used
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Cuda available, loading GPU device")
    else:
        device = torch.device('cpu')
        logging.info("No Cuda device available, using CPU")

    myBaseDAE = BaseDAE( io_size=dataset.nb_user+1, z_size=z_size, nb_input_layer=2, nb_output_layer=2 ).double()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam( myBaseDAE.parameters(), lr=learning_rate, weight_decay=weight_decay )

    dataset.set_view("item_view")
    sum_training_loss, sum_validation_loss = 0.0, 0.0

    logging.info("Training has started.")
    
    for epoch in range(nb_epoch):

        myBaseDAE.train()
        training_iter_nb, sum_training_loss = 0, 0

        for training_batch in training_set:

            cuda_input = Variable( torch.from_numpy( training_batch ) )

            cuda_output = myBaseDAE( cuda_input )   

            loss = criterion( cuda_output, cuda_input )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_training_loss += loss.item()
            training_iter_nb += 1

        myBaseDAE.eval()
        validation_iter_nb, sum_validation_loss = 0, 0

        for validation_batch in validation_set:

            cuda_input = Variable( torch.from_numpy( validation_batch ) )

            cuda_output = myBaseDAE( cuda_input )   

            loss = criterion( cuda_output, cuda_input )

            sum_validation_loss += loss.item()
            validation_iter_nb += 1

        logging.info('epoch [{}/{}], training loss:{:.4f}, validation loss:{:.4f}'.format(
            epoch + 1,
            nb_epoch,
            sum_training_loss/training_iter_nb,
            sum_validation_loss/validation_iter_nb)
        )
        sum_training_loss, sum_validation_loss = 0, 0
        training_iter_nb, validation_iter_nb = 0, 0


    #torch.save(model.state_dict(), './sim_autoencoder.pth')

    # measure ###########################################################################

    # train the model

    # display ###########################################################################
