# set the logging environment

import logging
import sys, os
import json
import datetime

import matplotlib.pyplot as plt
import torch


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

    
def display_info(config, nb_observation, metric_log=None):

    print("")
    logging.info("### LEARNING RATE   = %f" %config["MODEL"]["LEARNING_RATE"])
    logging.info("### WEIGHT DECAY    = %f" %config["MODEL"]["WEIGHT_DECAY"])
    logging.info("### NB EPOCH        = %d" %config["MODEL"]["EPOCH"])
    logging.info("### BATCH SIZE      = %d" %config["MODEL"]["BATCH_SIZE"])
    logging.info("### NB IN LAYER     = %d" %config["MODEL"]["NB_INPUT_LAYER"])
    logging.info("### NB OUT LAYER    = %d" %config["MODEL"]["NB_OUTPUT_LAYER"])
    logging.info("### STEEP LAYER     = %d" %config["MODEL"]["STEEP_LAYER_SIZE"])
    logging.info("### EMBEDDING SIZE  = %d" %config["DATASET"]["EMBEDDING_SIZE"])
    logging.info("### Z SIZE          = %d" %config["MODEL"]["Z_SIZE"])
    logging.info("### IO SIZE         = %d" %(len(config["DATASET"]["USED_CATEGORY"])*config["DATASET"]["EMBEDDING_SIZE"]))
    logging.info("### NB CATEGORY     = %d" %(len(config["DATASET"]["USED_CATEGORY"])))
    logging.info("### NB OBSERVATION  = %d\n" %(nb_observation))

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
        metric_log["NB_OBSERVATION"] = nb_observation

        return metric_log


def get_date():

    d = datetime.date.today()

    t = str( datetime.datetime.now().time() ).split('.')[0].replace(':', '')

    return '{:02d}'.format(d.day) + '{:02d}'.format(d.month) + '{:02d}'.format(d.year) + "_" + t



"""
    A class to save, export and display plots
"""
class PlotDrawer:

    def __init__(self):

        self.graph_list = []


    """
        add a plot to memory
    """
    def add(self, data, legend=None, title="", display=False):

        self.graph_list.append({
                "data": data,
                "legend": legend,
                "title": title})

        if display:
            self.display(data, legend, title)


    """
        display provided graph, or graph in memory
        this is not idiot proof ...
    """
    def display(self, data=None, legend=None, title=None, idx=None):

        plot_list = []

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if idx != None:

            if self.graph_list[idx]["title"] != None:
                fig.suptitle( self.graph_list[idx]["title"] )

            if isinstance( self.graph_list[idx]["legend"], list):
                for i in range(len(self.graph_list[idx]["legend"])):
                    plot, = ax.plot(self.graph_list[idx]["data"][i])
                    plot_list.append( plot )

                plt.legend(plot_list, self.graph_list[idx]["legend"])
                plt.draw()

            else:

                plot, = plt.plot(self.graph_list[idx]["data"])
                plt.legend(plot, self.graph_list[idx]["legend"])
                plt.show()

        else:

            if isinstance(legend, list):

                for i in range(len(legend)):
                    if title != None:
                        fig.suptitle(title)

                    plot, = ax.plot(data[i])
                    plot_list.append( plot )

                plt.legend(plot_list, legend)
                plt.show()


    """
        export saved or provided plots to png on disk
    """
    def export_to_png(self, data=None, legend=None, title=None, idx=None, export_path="out/"):

        if not os.path.exists(export_path):
            os.makedirs(export_path)

        if idx != None:

            data = self.graph_list[idx]["data"]
            legend = self.graph_list[idx]["legend"]
            title = self.graph_list[idx]["title"]

        path = (export_path + title + ".png" if title != None else export_path + "figure.png")

        fig = plt.figure()
        plot_list = []
        
        if title != None:
            fig.suptitle(title)

        if isinstance( legend, list):

            for i in range(len(legend)):
                plot, = plt.plot(data[i])
                plot_list.append( plot )

            plt.legend(plot_list, legend)
            fig.savefig( path )

        else:

            plot = plt.plot(data)
            plt.legend(plot, legend)
            fig.savefig( path )


def export_parameters_to_json(args, output_dir):

    d = vars(args)
    del d['log']

    content = json.dumps(d)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + "/training_parameters.json", 'w+') as f:
        f.write(content)