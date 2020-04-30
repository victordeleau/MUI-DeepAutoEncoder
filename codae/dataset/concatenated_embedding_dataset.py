
import random

import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class ConcatenatedEmbeddingDataset(Dataset):

    def __init__(self, embeddings, used_category):
        """
        input
            embedding: dict
                dict of embeddings keyed by observation ID
        """

        self.embeddings = embeddings

        self.used_category = used_category
        self.nb_used_category = len( self.used_category )

        # filter observations which do not contain the required categories
        self.filtered_embeddings = {}
        self.index = []
        for k, v in self.embeddings.items():
            ignore = False
            for uc in self.used_category:
                if uc not in v:
                    ignore = True
                    break
            if not ignore:
                self.index.append( k )
                self.filtered_embeddings[k] = v

        self.nb_observation = len(self.filtered_embeddings.keys())

        self.embedding_size = len(
            self.filtered_embeddings[ self.index[0] ][ self.used_category[0] ] )


    def __len__(self):

        return self.nb_observation


    def __getitem__(self, idx):

        r = torch.Tensor()

        for category in self.used_category: 

            r = torch.cat( (r, torch.Tensor(
                self.filtered_embeddings[self.index[idx]][category] ) ) )

        return r
