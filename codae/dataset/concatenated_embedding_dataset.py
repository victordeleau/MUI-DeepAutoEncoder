
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

        self.filtered_embeddings_tensor = torch.empty((len(self.index),
            self.nb_used_category*self.embedding_size))

        for c, i in enumerate(self.index):
            r = torch.Tensor()
            for category in self.used_category:
                r = torch.cat( (r, torch.Tensor(
                self.filtered_embeddings[i][category] ) ) )
            self.filtered_embeddings_tensor[c] = r


    def __len__(self):

        return self.nb_observation


    def __getitem__(self, idx):

        return self.filtered_embeddings_tensor[idx]

    
    def to(self, device):
        """
        send to inner data to device
        input
            device : torch.device
        """

        self.filtered_embeddings_tensor.to(device)

    
    def cosine_similarity(self, query, indices=None):
        """
        compute cosine_similarity between query embedding and dataset of embedding.
        input
            query : torch.Tensor (len == self.embedding_size)
                query tensor for which we want to find the most similar other
            indices : list(int)
                subset of indices of the dataset (default consider all of them)
        output

        """

        pass