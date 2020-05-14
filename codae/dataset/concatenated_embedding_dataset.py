
import random

import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class ConcatenatedEmbeddingDataset(Dataset):

    def __init__(self, embeddings, used_category):
        """
        input
            embedding : dict
                dict of embeddings keyed by observation ID
            used_category : list(str)
                subset of category to use in the dataset
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
    
        self.data_per_category = {}
        for category in range(self.nb_used_category):
            self.data_per_category[category] = torch.empty((
                len(self.index),
                self.embedding_size))

        # vectorize data

        self.data = torch.empty((len(self.index),
            self.nb_used_category*self.embedding_size))

        for c, i in enumerate(self.index):

            r = torch.Tensor()

            for n, category in enumerate(self.used_category):

                self.data_per_category[n][c] = torch.Tensor(
                self.filtered_embeddings[i][category] )

                r = torch.cat( (r, self.data_per_category[n][c] ) )

            self.data[c] = r

        for category in range(self.nb_used_category):

            self.data_per_category[category] = torch.transpose(
                self.data_per_category[category], 0, 1)


    def __len__(self):

        return self.nb_observation


    def __getitem__(self, idx):

        return self.data[idx], idx

    
    def to(self, device):
        """
        send to inner data to device
        input
            device : torch.device
        """

        self.data = self.data.to(device)

        for i in range(len(self.data_per_category.keys())):
            self.data_per_category[i] = self.data_per_category[i].to(device)