# tools related to mesuring stuff

import numpy as np
import os
import math
import json
import matplotlib.pyplot as plt

"""
    Recursively finds size of objects
    input
        obj: to be sized
        seen: ?
"""
def get_object_size(obj, seen=None):
    
    size = sys.getsizeof(obj)

    if seen is None:
        seen = set()

    obj_id = id(obj)

    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])

    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)

    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size


"""
    compute rmse between x and y
    (can be two scalars, two vectors, or two arrays)
    input
        x
        y
    output
        rmse
"""
def get_rmse(x, y):

    return np.sqrt( np.mean( (x-y)**2 ) )


"""
    Keep track of the model loss and provide methods to analyze
    init
        max_increasing_cnt
        max_nan_count
"""
class LossAnalyzer():

    def __init__(self, max_increasing_cnt, max_nan_count):

        self.max_increasing_cnt = max_increasing_cnt
        self.increasing_cnt = 0
        self.previous_losses = []

        self.max_nan_count = 0
        self.nan_count = 0


    def is_minimum(self, loss):

        if self.max_increasing_cnt == 0:
            return False

        loss_to_compare = loss

        if len(self.previous_losses) < self.max_increasing_cnt:

            self.previous_losses.insert( 0, loss )

            return False

        for i in range(len(self.previous_losses)):

            if loss_to_compare > self.previous_losses[i]:

                loss_to_compare = self.previous_losses[i]

            else:

                self.previous_losses.insert( 0, loss )
                
                del self.previous_losses[-1]

                return False

        return self.previous_losses[-1]


    def is_nan(self, loss):

        if math.isnan(loss):

            self.nan_count += 1

            if self.nan_count == self.max_nan_count:
                log.error("%d nan value detected, stopping." %self.max_nan_count)
                
            return True


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


def get_ranking_loss(prediction, dataset, corrupt_idx, idx):
    """
    Compute item ranking loss
    input 
        prediction : torch.Tensor
            the predicted observation
        corrupt_idx : list(list)
            batch size list of list of corrupted indices
        dataset : torch.data.Dataset
            the dataset of original embedding
    output
        ranking_loss : 0 < float < 1
            the ranking loss between 0 and 1
    """

    ranking_loss = 0

    for i in idx: # for each observation

        for j in range(corrupt_idx[i]): # for each corrupted index

            # cosine similarity between prediction and inventory

            # rank of the predicted item

            ranking_loss += 0

