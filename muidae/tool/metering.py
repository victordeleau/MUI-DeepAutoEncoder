# tools related to mesuring stuff

import numpy as np
import math

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

