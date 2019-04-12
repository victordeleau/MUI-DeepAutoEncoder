# Mixed User Item Deep Auto Encoder

import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from tool.logging import set_logging
from dataset.dataset_getter import dataset_getter
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

    dataset = None
    dg = dataset_getter()

    if dg.local_dataset_found():

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


    # model #############################################################################

    learning_rate = 1e-3
    weight_decay = 1e-5
    nb_epoch = 50
    z_size = 126

    # check which computing device should be used
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Cuda available, loading GPU device")
    else:
        device = torch.device('cpu')
        logging.info("No Cuda device available, using CPU")

    myBaseDAE = BaseDAE( io_size=dataset.nb_user, z_size=z_size ) # instantiate model

    criterion = nn.MSELoss() # loss method

    # algorithm used to optimize the autoencoder
    optimizer = torch.optim.Adam( myBaseDAE.parameters(), lr=learning_rate, weight_decay=weight_decay )

    dataset.set_view("user_view")
    for vector in dataset:
        print( np.nonzero(vector) )
    
    # train the autoencoder
    """for epoch in range(nb_epoch):

        for data in dataloader:

            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()

            # ===================forward=====================

            output = model(img)
            loss = criterion(output, img)

            # ===================backward====================

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================log========================

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, nb_epoch, loss.data[0]))

        #if epoch % 10 == 0:
            #pic = to_img(output.cpu().data)
            #save_image(pic, './mlp_img/image_{}.png'.format(epoch))
    """

    #torch.save(model.state_dict(), './sim_autoencoder.pth')

    # measure ###########################################################################

    # train the model

    # display ###########################################################################
