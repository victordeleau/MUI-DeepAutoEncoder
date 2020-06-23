# tools related to mesuring stuff

import os, sys
import math
import json
import operator

import torch
import numpy as np
import matplotlib.pyplot as plt

from codae.tool import get_mask_transformation


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

    def __init__(self, arch, k_max, device, observation_mask):

        if k_max < 0 | k_max >= len(arch):
            raise Exception("Error: maximum number of corrupted index [k_max] must be (> 0) && (< len(arch)).")

        self.arch = arch
        self.k_max = k_max
        self.device = device

        self.observation_mask = observation_mask
        self.io_size = len(self.observation_mask)

        self.loss_mask = []
        for i, variable in enumerate(arch):
            if variable["type"] == "continuous":
                self.loss_mask.append(1)
            else:
                self.loss_mask.append(0)

        self.mask_transformation = get_mask_transformation(
            observation_mask=self.observation_mask,
            loss_mask=self.loss_mask).to(self.device).cpu().numpy()

        self.MSE_criterion = torch.nn.MSELoss(reduction="none")
        self.CE_criterion = torch.nn.CrossEntropyLoss(reduction="none")


    def __call__(self, x, y):
        """
        Default method => compute the loss
        input
            x : torch.tensor
            y : torch.Tensor
        """

        return self._compute_loss(x, y)


    def _compute_loss(self, x, y):
        """
        Compute the loss of each variable and return a list.
        input 
            x : torch.Tensor
            y : torch.Tensor
        """

        loss = torch.zeros((x.size()[0], len(self.arch)), device=self.device)        

        eps = 1e-6

        for j, variable in enumerate(self.arch): # for each input variable

            if variable["type"] == "regression":

                loss[:,j:j+1] += torch.sqrt(self.MSE_criterion(
                        input=x[:,variable["position"]:variable["position"]+variable["size"]],
                        target=y[:,variable["position"]:variable["position"]+variable["size"]]) + eps)

            else: # is classification

                loss[:,j] += self.CE_criterion(
                    input=y[:,variable["position"]:variable["position"]+variable["size"]],
                    target=x[:,variable["position"]:variable["position"]+variable["size"]].max(1)[1])

        return loss
        

    ############################################################################
    # loss masking #############################################################

    def get_per_k(self, loss, masks):

        losses_per_k = np.zeros((self.k_max, len(self.arch)))
        
        for i, mask in enumerate(masks): # for each k number of missing variable

            m = np.matmul(mask.cpu().numpy(),np.ones((self.io_size,self.io_size)))
            m[m > 1] = 1
            losses_per_k[i, :] = np.sum( np.matmul(m, self.mask_transformation) * loss, axis=0)

        return losses_per_k


    def get_partial(self, loss, mask):

        partial_loss = np.zeros((loss.shape[0], len(self.arch)))

        #print(loss)
        #print(mask)
        
        partial_loss = (1-np.matmul(mask.cpu().numpy(), self.mask_transformation)) * loss

        #print(partial_loss)
        #print()

        return partial_loss



class BookLoss:

        def __init__(self, name, shape, device, loss=None):

            self.name = name
            self.shape = shape
            self.device = device

            if loss == None:
                self.loss = np.zeros((shape[0], shape[1]))
            else:
                self.loss = loss

        def add(self, x):

            self.loss[:len(x)] += x

            return self

        def mean(self):

            self.loss = np.mean(self.loss)

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

            self.loss = np.zeros((self.shape[0], self.shape[1]))

            return self

        def sum(self, dim=None):
            """
            Get sum over specific dimension.
            """

            if dim == None:
                self.loss = np.sum(self.loss)
            else:
                self.loss = np.sum(self.loss, axis=dim)

            return self

        def get(self):

            return self.loss

        def copy(self):

            return BookLoss(
                name=self.name,
                shape=self.shape,
                device=self.device,
                loss=self.loss)


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


    def copy_book(self, origin, destination):

        self.add_book(
            name=destination,
            shape=self.book[origin].shape)

        self.book[destination].loss = self.book[origin].loss

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

        return np.mean(x)        