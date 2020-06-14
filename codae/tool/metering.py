# tools related to mesuring stuff

import os, sys
import math
import json
import operator

import torch
import numpy as np
import matplotlib.pyplot as plt


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



def get_ranking_loss(prediction, dataset, corrupt_embedding, idx, sub_index):
    """
    Compute item ranking loss
    input 
        prediction : torch.Tensor
            the predicted observation
        corrupt_embedding : list(list)
            batch size list of list of corrupted indices
        dataset : torch.data.Dataset
            the dataset of original embedding
    output
        ranking_loss : 0 < float < 1
            the ranking loss between 0 and 1
    """

    ranking_loss = 0

    batch_index = 0
    for i in idx: # for each observation in batch

        # extract predicted missing embedding
        predicted_missing_embedding = prediction[batch_index][corrupt_embedding[batch_index]*dataset.embedding_size : (corrupt_embedding[batch_index]+1)*dataset.embedding_size]

        # compute similarity in corresponding inventory
        s = torch.matmul(
            predicted_missing_embedding,
            dataset.data_per_category[corrupt_embedding[batch_index]]).tolist()

        batch_index += 1

        rank = 0
        for j in sub_index:
            if s[i] > s[j]:
                rank += 1
        ranking_loss += rank/len(sub_index)

    return ranking_loss/len(idx)



class CombinedCriterion:
    """
    Combine multiple losses together according
    to the architecture of the input.
    """

    def __init__(self, arch, k_max, device):

        if k_max < 0 | k_max >= len(arch):
            raise Exception("Error: maximum number of corrupted index [k_max] must be (> 0) && (< len(arch)).")

        self.arch = arch
        self.k_max = k_max
        self.device = device

        self.book = {}

        self.MSE_criterion = torch.nn.MSELoss(reduction="sum")
        self.CE_criterion = torch.nn.CrossEntropyLoss(reduction="sum")


    def __call__(self, x, y, masks):
        """
        Default method => compute the loss
        input
            x : torch.tensor
            y : torch.Tensor
            masks : list(torch.Tensor(io_size X batch_size))
                list of binary masks tensors
        """

        return self._compute_loss(x, y, masks)


    def _compute_loss(self, x, y, masks):
        """
        Compute the loss of each variable and return a list.
        input 
            x : torch.Tensor
            y : torch.Tensor
            masks : list(torch.Tensor(io_size X batch_size))
                list of binary masks tensors
        """

        # row = variable, column = nb_corrupt_index
        losses = torch.zeros((self.k_max, len(self.arch)), device=self.device)
        ones = torch.ones((x.size()[1], x.size()[1]), device=self.device)

        for i, mask in enumerate(masks): # for each k number of missing variable
            
            m = mask.matmul(ones).type(torch.cuda.ByteTensor)
            a = x * m
            b = y * m

            for j, variable in enumerate(self.arch): # for each input variable

                if variable["type"] == "regression":

                    losses[i,j:j+1] = torch.sqrt(self.MSE_criterion(
                            input=a[:,variable["position"]:variable["position"]+variable["size"]],
                            target=b[:,variable["position"]:variable["position"]+variable["size"]]))

                else: # is classification

                    losses[i,j:j+1] = self.CE_criterion(
                        input=a[:,variable["position"]:variable["position"]+variable["size"]],
                        target=b[:,variable["position"]:variable["position"]+variable["size"]].max(1)[1])
        #print(losses)

        return losses


class BookLoss:

        def __init__(self, name, shape, device):

            self.name = name
            self.shape = shape
            self.device = device

            self.loss = torch.zeros((shape[0], shape[1]), device=self.device)

        def add(self, x):

            self.loss += x

            return self

        def mean(self):

            self.loss = torch.mean(self.loss)

            return self

        def divide(self, x):
            """
            Return loss divided by scalar.
            """

            self.loss /= x

            return self

        def purge(self):
            """
            Set losses to zero.
            """

            self.loss = torch.zeros((self.shape[0], self.shape[1]), device=self.device)

            return self

        def sum(self, dim=None):
            """
            Get sum over specific dimension.
            """

            if dim == None:
                self.loss = torch.sum(self.loss)
            else:
                self.loss = torch.sum(self.loss, dim=dim)

            return self

        def get(self):

            return self.loss.clone().detach().cpu().numpy()


class LossManager:

    def __init__(self, device):
        """
        Keep track of loss per variable & per missing number of variable.
        Provide methods to get whole layer loss per number of missing variable ([1]x[k_max] list), and whole layer loss over all number of missing variables ([nb_variable]x[k_max] 2D list). Allow for sum or mean reduction scheme.
        input
            device
        """

        self.device = device

        self.book = {}

        self.log = {}

    
    def add_book(self, name, shape):

        self.book[name] = BookLoss(
            name=name,
            shape=shape,
            device=self.device)

        return self


    def get_book(self, name):

        return self.book[name]


    def copy_cook(self, origin, destination):

        self.book[destination] = self.book[origin]

        return self


    def log_book(self, name):
        """
        Store book as a numpy array of losses.
        input
            name : str
        """

        if not name in self.log:
            self.log[name] = []

        self.log[name].append( self.book[name].get() )

        return self


    def get_log(self, name):

        return self.log[name]


    def get_mean(self, x):

        return torch.mean(torch.stack([i for j in x for i in j]))