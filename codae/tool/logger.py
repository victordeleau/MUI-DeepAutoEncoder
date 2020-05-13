# set the logging environment

import logging
import sys
import datetime
import torch
import datetime

"""
    setup logging capabilities
"""
def set_logging(log_file_path="/mnt/ramdisk/", log_file_name=None, logging_level=logging.INFO):

    now = datetime.datetime.now()
    log_file_name = ("CODAE_" + now.strftime("%Y-%m-%d %H:%M") + ".log" if log_file_name == None else log_file_name)

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
    except:
        logging.error("Couln't create log file %s" %(log_file_path + log_file_name) )

    return logging

    
def display_info(config, metric_log=None):

    print("")
    logging.info("### LEARNING RATE = %f" %config["MODEL"]["LEARNING_RATE"])
    logging.info("### WEIGHT DECAY = %f" %config["MODEL"]["WEIGHT_DECAY"])
    logging.info("### NB EPOCH = %d" %config["MODEL"]["EPOCH"])
    logging.info("### BATCH SIZE = %d" %config["MODEL"]["BATCH_SIZE"])
    logging.info("### NB INPUT_LAYER = %d" %config["MODEL"]["NB_INPUT_LAYER"])
    logging.info("### NB OUTPUT LAYER = %d" %config["MODEL"]["NB_OUTPUT_LAYER"])
    logging.info("### STEEP LAYER SIZE = %d" %config["MODEL"]["STEEP_LAYER_SIZE"])
    logging.info("### EMBEDDING SIZE = %d" %config["DATASET"]["EMBEDDING_SIZE"])
    logging.info("### Z SIZE = %d" %config["MODEL"]["Z_SIZE"])
    logging.info("### IO SIZE = %d" %(len(config["DATASET"]["USED_CATEGORY"])*config["DATASET"]["EMBEDDING_SIZE"]))
    logging.info("### NB CATEGORY = %d\n" %(len(config["DATASET"]["USED_CATEGORY"])))

    if metric_log != None:

        metric_log["LEARNING_RATE"] = config["MODEL"]["LEARNING_RATE"]
        metric_log["WEIGHT_DECAY"] = config["MODEL"]["WEIGHT_DECAY"]
        metric_log["EPOCH"] = config["MODEL"]["EPOCH"]
        metric_log["BATCH_SIZE"] = config["MODEL"]["BATCH_SIZE"]
        metric_log["NB_INPUT_LAYER"] = config["MODEL"]["NB_INPUT_LAYER"]
        metric_log["NB_OUTPUT_LAYER"] = config["MODEL"]["NB_OUTPUT_LAYER"]
        metric_log["STEEP_LAYER_SIZE"] = config["MODEL"]["STEEP_LAYER_SIZE"]
        metric_log["EMBEDDING_SIZE"] = config["DATASET"]["EMBEDDING_SIZE"]
        metric_log["Z_SIZE"] = config["MODEL"]["Z_SIZE"]
        metric_log["IO_SIZE"] = len(config["DATASET"]["USED_CATEGORY"])*config["DATASET"]["EMBEDDING_SIZE"]
        metric_log["NB_CATEGORY"] = len(config["DATASET"]["USED_CATEGORY"])

        return metric_log


def get_date():

    d = datetime.date.today()

    t = str( datetime.datetime.now().time() ).split('.')[0].replace(':', '')

    return '{:02d}'.format(d.day) + '{:02d}'.format(d.month) + '{:02d}'.format(d.year) + "_" + t