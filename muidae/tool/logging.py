# set the logging environment

import logging
import sys
import datetime

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

    
    
