
import random

import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class ConcatenatedEmbeddingDataset(Dataset):

    def __init__(self, embeddings, used_category, transform=None):
        """
        input
            embedding : dict
                dict of embeddings keyed by observation ID
            used_category : list(str)
                subset of category to use in the dataset
        """

        self.embeddings = embeddings

        self.transform = transform

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

        self.min = self.data.min()
        self.max = self.data.max()
        self.scale = self.max - self.min
        self.scale = self.scale.item()

        self.data = self.data / self.scale

        for category in range(self.nb_used_category):

            self.data_per_category[category] = torch.transpose(
                self.data_per_category[category], 0, 1) / self.scale

        # create dataset.arch
        self.arch = [] # build architecture
        self.io_size = 0
        for i, name in enumerate(self.used_category):

            self.arch.append({})
            self.arch[-1]["name"] = name

            # weight of variable in loss function [0 : 1] 
            self.arch[-1]["lambda"] = 1 # for now TODO

            self.arch[-1]["size"] = self.embedding_size
            self.arch[-1]["type"] = "regression"
            print("R" + str(self.embedding_size) + " ", end="")
            
            self.arch[-1]["position"] = self.io_size
            self.io_size += self.arch[-1]["size"]
        print("\n")

        self.type_mask = torch.ones((self.io_size))

        self.nb_predictor = self.embedding_size * self.nb_used_category


    def __len__(self):

        return self.nb_observation


    def __getitem__(self, idx):

        if self.transform != None:

            return self.transform(self.data[idx]), idx

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