# tools related to mesuring stuff

import numpy as np

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