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


class RankingLoss:

    def __init__(self, dataset, validation_indices, device):

        self.dataset = dataset
        self.device = device

        self.category_getter = torch.zeros((self.dataset.nb_predictor), device=self.device)

        for i in range(self.dataset.nb_used_category):
            self.category_getter[i*self.dataset.embedding_size] = i

        #self.cosim = torch.nn.CosineSimilarity(dim=1)

        self.validation_indices = validation_indices


    def get(self, prediction, fmask, indices):

        ranking_loss = 0

        fmask_reverse = 1 - fmask

        # for each observation in batch
        for i, idx in enumerate(indices):

            # retrieve category of item
            c = int(torch.dot(fmask_reverse[i], self.category_getter).item())

            # compute similarity of item with corresponding inventory
            """
            print(type(c))
            print(self.dataset.data_per_category[c].size())
            print(prediction[i].size())
            print(prediction[i][c*self.dataset.embedding_size:(c+1)*self.dataset.embedding_size].size())
            print(prediction[i][c*self.dataset.embedding_size:(c+1)*self.dataset.embedding_size].reshape(1, -1).size())
            """

            s = torch.nn.functional.cosine_similarity(
                self.dataset.data_per_category[c],
                prediction[i][c*self.dataset.embedding_size:(c+1)*self.dataset.embedding_size].reshape(1, -1))

            # get rank
            rank = 0
            for j in self.validation_indices:
                if s[idx] > s[j]:
                    rank += 1
            #print("=>" + str(1 - (rank/len(self.validation_indices))))
            ranking_loss += 1 - (rank/(len(self.validation_indices)-1))

        return ranking_loss


class CombinedCriterion:
    """
    Combine multiple losses together according
    to the architecture of the input.
    """

    def __init__(self, arch, k_max, device, observation_mask, weight=None, reduction="none"):

        if k_max < 0 | k_max >= len(arch):
            raise Exception("Error: maximum number of corrupted index [k_max] must be (> 0) && (< len(arch)).")

        self.arch = arch
        self.k_max = k_max
        self.device = device
        self.reduction = reduction

        if weight == None:
            self.weight = torch.ones((1,len(self.arch)))
        else:
            self.weight = torch.Tensor(weight)

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

        self.MSE_criterion = torch.nn.MSELoss(reduction=self.reduction)
        self.CE_criterion = torch.nn.NLLLoss(reduction=self.reduction)


    def __call__(self, x, y, as_numpy=False):

        if self.reduction == "mean":
            return self._mean_loss(x, y, as_numpy)
        elif self.reduction == "none":
            return self._full_loss(x, y, as_numpy)
        else:
            raise Exception("Unknown reduction type.")


    def _full_loss(self, x, y, as_numpy=False):

        loss = torch.zeros((x.size()[0], len(self.arch)))

        for i, variable in enumerate(self.arch):

            if variable["type"] == "regression":

                loss[:, i:i+1] = self.MSE_criterion(
                    input=x[:,variable["position"]:variable["position"]+variable["size"]],
                    target=y[:,variable["position"]:variable["position"]+variable["size"]])

            else: # is classification

                loss[:, i] = self.CE_criterion(
                    input=torch.log_softmax(y[:,variable["position"]:variable["position"]+variable["size"]], dim=1),
                    target=x[:,variable["position"]:variable["position"]+variable["size"]].max(dim=1)[1])

        if as_numpy:
            return loss.clone().cpu().detach().numpy()

        return loss


    def _mean_loss(self, x, y, as_numpy=False):

        loss = [0 for x in range(len(self.arch))]

        for i, variable in enumerate(self.arch):

            if variable["type"] == "regression":

                loss[i] += torch.sqrt(self.MSE_criterion(
                    input=x[:,variable["position"]:variable["position"]+variable["size"]],
                    target=y[:,variable["position"]:variable["position"]+variable["size"]]))

            else: # is classification

                loss[i] += self.CE_criterion(
                    input=torch.log_softmax(y[:,variable["position"]:variable["position"]+variable["size"]], dim=1),
                    target=x[:,variable["position"]:variable["position"]+variable["size"]].max(dim=1)[1])

        if as_numpy:
            loss = sum(loss)/len(self.arch)
            return loss.clone().cpu().detach().numpy()

        loss = [ loss[i]*self.weight[i] for i in range(len(loss)) ]
        #print(loss)

        return sum(loss)/len(self.arch)

        

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
        
        partial_loss = (1-np.matmul(mask.cpu().numpy(), self.mask_transformation)) * loss

        return partial_loss