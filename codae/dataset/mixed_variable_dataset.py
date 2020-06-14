
import random

import torch
from torch.utils.data.dataset import Dataset
import numpy as np

class MixedVariableDataset(Dataset):

    def __init__(self, pd_dataset):
        """
        input
            pd_dataset : pandas.Dataframe
                dataset as pandas Dataframe
        """

        self.pd_dataset = pd_dataset

        self.variable_names = pd_dataset.columns

        self.nb_predictor = len(self.pd_dataset.columns)

        self.nb_observation = len(self.pd_dataset)

        self.io_size = 0

        self.arch = [] # build architecture
        for i, column in enumerate(self.pd_dataset):

            self.arch.append({})
            self.arch[-1]["name"] = column

            # weight of variable in loss function [0 : 1] 
            self.arch[-1]["lambda"] = 1 # for now TODO

            if self.pd_dataset.dtypes[i] == "float64" or self.pd_dataset.dtypes[i] == "int64":
                self.arch[-1]["size"] = 1
                self.arch[-1]["type"] = "regression"
                print("R1 ", end="")
                

            else:                
                self.arch[-1]["size"] = self.pd_dataset[column].nunique()
                self.arch[-1]["type"] = "classification"
                print("C%d " %self.arch[-1]["size"], end="")
            
            self.arch[-1]["position"] = self.io_size
            self.io_size += self.arch[-1]["size"]

        self.type_mask = torch.zeros((self.io_size))
        for i, variable in enumerate(self.arch):
            if variable["type"] == "regression":
                self.type_mask[variable["position"]:variable["position"]+variable["size"]] = 1

        # vectorize dataset
        self.map = {}
        self.data = np.empty(shape=(self.nb_observation, self.io_size))
        for i in range(self.nb_observation):
            
            for variable in self.arch:

                if variable["type"] == "classification":

                    if variable["name"] not in self.map:
                        self.map[variable["name"]] = {}
                        self.map[variable["name"]]["COUNT"] = 0

                    if self.pd_dataset[variable["name"]][i] not in self.map[variable["name"]]:

                        self.map[variable["name"]][self.pd_dataset[variable["name"]][i]] = self.map[variable["name"]]["COUNT"]

                        self.map[variable["name"]]["COUNT"] += 1

                    int_label = self.map[variable["name"]][self.pd_dataset[variable["name"]][i]]

                    ohe_variable = self._categorical_to_OHE(
                        label=int_label,
                        max=variable["size"])

                    self.data[i][variable["position"]:variable["position"]+variable["size"]] = ohe_variable

                else: # regression
                    self.data[i][variable["position"]:variable["position"]+variable["size"]] = self.pd_dataset[variable["name"]][i]

        self.data = torch.Tensor(self.data)


    def __len__(self):

        return self.nb_observation


    def __getitem__(self, idx):

        return self.data[idx], idx


    def _categorical_to_OHE(self, label, max):
        """
        Convert label as integer to one hot encoding vector
        input
            label: int
                label as integer
            max : int
                maximum label value
        output
            output : np.array()
                one hot encoding
        """

        output = np.zeros(max)

        output[label] = 1

        return output

    
    def to(self, device):
        """
        send to inner data to device
        input
            device : torch.device
        """

        self.data = self.data.to(device)
        self.type_mask = self.type_mask.to(device)

    
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