import numpy as np
import random

import torch

def my_collate(batch):
    """
    merge list of torch.Tensor into single Tensor by stacking them
    """

    return torch.stack(batch)