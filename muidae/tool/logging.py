# set the logging environment

import logging
import sys
import datetime

"""
    setup logging capabilities
"""
def set_logging(logFilePath="/mnt/ramdisk/", logFileName=None, loggingLevel=logging.INFO):

    now = datetime.datetime.now()
    logFileName = ("MUI-DAE_" + now.strftime("%Y-%m-%d %H:%M") + ".log" if logFileName == None else logFileName)

    root = logging.getLogger()
    root.setLevel(loggingLevel)

    # send logging stream to stdout as well
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(loggingLevel)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # try to add log output file
    try:
        fh = logging.FileHandler( logFilePath + logFileName )
        fh.setLevel(loggingLevel)
        root.addHandler(fh)
        logging.info("Logging level %s to file %s" %(loggingLevel, logFilePath + logFileName) )
    except:
        logging.error("Couln't create log file %s" %(logFilePath + logFileName) )

    return logging

    
    
