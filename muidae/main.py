
import logging

from train_autoencoder import train_autoencoder
from tool.parser import parse
from tool.logging import set_logging



"""
    train an autoencoder with either a user view or an item view
"""
def user_or_item_ae(args):

    train_autoencoder(args)


"""
    extract a user embedding and a item embedding, then train an MLP
    to predict rating of a given user to a given item.
"""
def joint_user_item_ae(args):

    args.view = "user"
    train_autoencoder(args)

    args.view = "item"
    train_autoencoder(args)


"""
    extract a user embedding OR a item embedding, for different z size,
    then combine this embbeding in a pyramidal square matrix (see note),
    then train an MLP on this input to predict the rating of a given user
    to a given item. 
"""
def pyramidal_user_or_item_ae(args):

    # loop over different z size for user OR item view
    for zsize in range(7):
        pass

    # combine those different z size user OR item view


"""
    extract a user embedding AND a item embedding, for different z size,
    then combine those embbedings in a pyramidal square matrix (see note),
    then train an MLP on those inputs to predict the rating of a given user
    to a given item.
"""
def pyramidal_joint_ae(args):

    # loop over different z size for user view
    for zsize in range(7):
        pass

    # combine those different z size user view

    # loop over different z size for item view
    for zsize in range(7):
        pass

    # combine those different z size item view
    

if __name__=="__main__":

    args = parse()

    vars(args)["log"] = set_logging(logging_level=(logging.DEBUG if args.debug else logging.INFO))

    args.log.info("MUIDAE has started.")

    
    if args.mode == 0: # simple user or item autoencoder

        args.log.info("Simple " + args.view + " view autoencoder selected.")

        user_or_item_ae(args)

    
    elif args.mode == 1: #  joint user item autoencoder

        args.log.info("Joint user item view autoencoder selected.")

        joint_user_item_ae(args)


    elif args.mode == 2: # pyramidal user OR item autoencoder

        args.log.info("Pyramidal " + args.view + " view autoencoder selected.")

        pyramidal_user_or_item_ae(args)


    elif args.mode == 3: # pyramidal joint user item autoencoder

        args.log.info("Pyramidal joint user item view autoencoder selected.")

        pyramidal_joint_ae(args)


    else:

        raise Exception("Invalid mode provided, options are [0, 1, 2, 3]")
