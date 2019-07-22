# set the logging environment

import logging
import sys
import datetime
import torch

"""
    setup logging capabilities
"""
def set_logging(log_file_path="/mnt/ramdisk/", log_file_name=None, logging_level=logging.INFO):

    now = datetime.datetime.now()
    log_file_name = ("MUI-DAE_" + now.strftime("%Y-%m-%d %H:%M") + ".log" if log_file_name == None else log_file_name)

    root = logging.getLogger()
    root.setLevel(logging_level)

    # send logging stream to stdout as well
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # try to add log output file
    try:
        fh = logging.FileHandler( log_file_path + log_file_name )
        fh.setLevel(logging_level)
        root.addHandler(fh)
        #logging.info("Logging level %s to file %s" %(loggingLevel, log_file_path + log_file_name) )
    except:
        logging.error("Couln't create log file %s" %(log_file_path + log_file_name) )

    return logging

    
def display_info(args, dataset=None):

    print("")
    logging.info("### LEARNING RATE = %f" %args.learning_rate)
    logging.info("### WEIGHT DECAY = %f" %args.regularization)
    logging.info("### EPOCH = %d" %args.nb_epoch)
    logging.info("### BATCH SIZE = %d" %args.batch_size)
    logging.info("### LAYERS = %d" %((args.nb_layer*2)+1))
    logging.info("### Z SIZE = %d" %args.zsize)
    
    if dataset != None:
        logging.info("### IO SIZE = %s" %dataset.get_io_size())
        logging.info("### DATASET NAME = %s" %args.dataset)
        logging.info("### NB USER = %s" %dataset.nb_user)
        logging.info("### NB ITEM = %s" %dataset.nb_item)
        logging.info("### DATASET NAME = %s" %args.dataset)
        logging.info("### DATASET SIZE FACTOR = %.4f" %args.redux)
        logging.info("### DATASET VIEW = %s" %args.view)
        logging.info("### NORMALIZED = %s\n" %args.normalize)