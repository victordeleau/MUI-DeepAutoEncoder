# load and prepare dataset

import pandas as pd
import sys
import math
import logging
from sklearn.model_selection import train_test_split
import torchvision
import torch
import scipy.sparse as sparse
import numpy as np
import random
from torch.utils.data.dataset import Dataset as PytorchDataset

# TODO remove first row and column from data (nan, possibly text or something)
#np.set_printoptions(threshold=sys.maxsize)

torch.multiprocessing.set_sharing_strategy('file_system')


"""
    load provided dataset inside object, provide useful methods and information
    input
        ...
"""
class RatingDataset(PytorchDataset):

    # pass normalization data to sub-dataset

    def __init__(self, df_data, name, view="user"):

        assert( isinstance(df_data, pd.DataFrame) )

        self.name = name

        self._view = view

        self.iterator_count = 0

        self._has_been_normalized = False

        self.df_data = df_data

        self.csr_data = sparse.csr_matrix(
            (
                df_data.values[:, 2],
                (df_data.values[:, 0].astype(int),
                df_data.values[:, 1].astype(int))
            )
        )               

        self.index_user, self.index_item = self.df_data.userId.unique(), self.df_data.itemId.unique()
        self.nb_user, self.nb_item = len( self.index_user )-1, len( self.index_item )-1

        self._randomize()
        self._map_index_to_monotonic()

        self._io_size = (self.nb_item+1 if self._view == "user" else self.nb_user+1 )
        


    """
        just return the dataset's view
    """
    def get_view(self):

        return self._view


    """
        just return io_size (depends on view)
    """
    def get_io_size(self):

        return self._io_size


    def __iter__(self):

        self.count = 0

        return self


    def __next__(self):

        
        if self._view == "item":

            self.iterator_count += 1

            if self.iterator_count > self.nb_item:
                raise StopIteration

            return self.__getitem__(self.iterator_count-1)


        elif self._view == "user":

            self.iterator_count += 1

            if self.iterator_count > self.nb_user:
                raise StopIteration

            return self.__getitem__(self.iterator_count-1)

        else:
            return 0


    """
        return size of the dataset (overide required from abstract parent class)
    """
    def __len__(self):
        
        if self._view == "item":
            return self.nb_item

        elif self._view == "user":
            return self.nb_user

        else:
            return 0


    """
        allow index access of the dataset (overide required from abstract parent class)
        input
            idx: index of the vector we want to grab
    """
    def __getitem__(self, idx):

        print(idx)

        if self._view == "item":
            
            if idx < 0 or idx > self.nb_item:

                return 0

            elif self._has_been_normalized:

                swap_idx = self.item_index_swap[idx]

                data = self.csr_data[:, swap_idx].todense()[1:]
                bias = self.gm + self.um + self.im[swap_idx]

                unbiased = np.ravel( data ) - bias
                centered = unbiased - unbiased.mean()

                return centered

            else:

                swap_idx = self.item_index_swap[idx]

                output = np.ravel(self.csr_data[:, swap_idx].todense())[1:]

                return output

        elif self._view == "user":
            
            if idx < 0 or idx > self.nb_user:

                return 0

            elif self._has_been_normalized:

                swap_idx = self.user_index_swap[idx]

                data = np.ravel(self.csr_data[swap_idx, :].todense())[1:]
                bias = self.gm + self.um[swap_idx] + self.im

                unbiased = np.ravel( data - bias )
                centered = unbiased - unbiased.mean()

                return centered

            else:

                swap_idx = self.user_index_swap[idx]

                return np.ravel( self.csr_data[swap_idx, :].todense() )[1:]

        else:
            return 0


    """
        remove mean, user and/or item biases and return new Dataset object
        input
            mean:
            user:
            item:
        output:
            dataset: new Dataset object with modified ratings
    """
    def normalize(self, global_mean=True, user_mean=True, item_mean=True):

        self.gm, self.um, self.im = 0, [], []

        if global_mean == True:

            self.gm = self.csr_data.sum() / self.csr_data.getnnz()

        if user_mean == True:

            um_sum = np.ravel( np.transpose( self.csr_data.sum(axis=1) ) )[1:]
            um_nnz = self.csr_data.getnnz(axis=1)[1:]
            self.um = np.divide( um_sum.astype(float), um_nnz, out=np.zeros_like(um_sum), where=um_nnz!=0 ) - self.gm
        
        if item_mean == True:

            im_sum = np.ravel( self.csr_data.sum(axis=0) )[1:]
            im_nnz = self.csr_data.getnnz(axis=0)[1:]
            self.im = np.ravel( np.divide( im_sum.astype(float), im_nnz, out=np.zeros_like(im_sum), where=im_nnz!=0 ) ) - self.gm

        self._has_been_normalized = True

        return self


    """
        create random swap of row and column indices
        don't randomize and return 0 if self.has_been_randomized==True
        typically if current dataset is a subset of another dataset that has
        been previously randomized.
    """
    def _randomize(self):

        self.user_index_swap = sorted(np.arange(self.nb_user), key=lambda k: random.random())
        self.item_index_swap = sorted(np.arange(self.nb_item), key=lambda k: random.random())

        self._has_been_randomized = True


    """
        map row and column name to monotonic index and save in dict (as string)
        input
            df_data: pandas.DataFrame to process
    """
    def _map_index_to_monotonic(self):

        temp_dict = {}
        c = 1
        for i in self.df_data.userId.unique():
            temp_dict[ i ] = c
            c+= 1

        self.df_data['userId'].replace( temp_dict, inplace = True )

        self.userId_map = dict((str(k), v) for k, v in temp_dict.items())
        
        temp_dict = {}
        c = 1
        for i in self.df_data.itemId.unique():
            temp_dict[ i ] = c
            c+= 1
        
        self.df_data['itemId'].replace( temp_dict, inplace = True )

        self.itemId_map = dict((str(k), v) for k, v in temp_dict.items())

        return self
