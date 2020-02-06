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
np.set_printoptions(threshold=sys.maxsize)


"""
    load provided dataset inside object, provide useful methods and information
    input
        ...
"""
class RatingDataset(PytorchDataset):

    def __init__(self, data, name, view="user", index_user=[], index_item=[]):

        self.name = name

        self._view = view

        self.iterator_count = 0

        if isinstance(data, pd.DataFrame):

            self._has_been_mean_normalized = False

            data = self._map_index_to_monotonic(data)

            self.csr_data = sparse.csr_matrix(
                (
                    data.values[:, 2],
                    (data.values[:, 0].astype(int),
                    data.values[:, 1].astype(int))
                )
            )

            self.index_user, self.index_item = data.userId.unique(), data.itemId.unique()

        elif isinstance(data, sparse.csr_matrix):

            assert( len(index_item) != 0 and len(index_user) != 0 )

            self._has_been_mean_normalized = False

            self.csr_data = data

            self.index_user, self.index_item = index_user, index_item

        else:

            raise Exception("Error: provided data not valid. ")
        
        self.nb_user, self.nb_item = len( self.index_user )-1, len( self.index_item )-1

        self._randomize()
        
        self.io_size = ( self.nb_item+1 if self._view == "user" else self.nb_user+1 )
        


    """
        just return the dataset's view
    """
    def get_view(self):

        return self._view


    """
        just return io_size (depends on view)
    """
    def get_io_size(self):

        return self.io_size


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

        if self._view == "item":
            
            if idx < 0 or idx > self.nb_item:

                return 0

            elif self._has_been_mean_normalized:

                swap_idx = self.item_index_swap[idx]

                data = self.csr_data[:, swap_idx].todense()[1:]

                unbiased = np.ravel( data ) - self.gm - self.um - self.im[swap_idx]

                return unbiased

            else:

                swap_idx = self.item_index_swap[idx]

                return np.ravel(self.csr_data[:, swap_idx].todense())[1:]

        elif self._view == "user":
            
            if idx < 0 or idx > self.nb_user:

                return 0

            elif self._has_been_mean_normalized:

                swap_idx = self.user_index_swap[idx]

                data = np.ravel(self.csr_data[swap_idx, :].todense())[1:]

                unbiased = np.ravel( data - self.gm - self.um[swap_idx] - self.im )

                return unbiased

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
    def mean_normalize(self, global_mean=True, user_mean=True, item_mean=True):

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

        self._has_been_mean_normalized = True

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
    def _map_index_to_monotonic(self, df_data):

        temp_dict = {}
        c = 1
        for i in df_data.userId.unique():
            temp_dict[ i ] = c
            c+= 1

        df_data['userId'].replace( temp_dict, inplace = True )

        self.userId_map = dict((str(k), v) for k, v in temp_dict.items())
        
        temp_dict = {}
        c = 1
        for i in df_data.itemId.unique():
            temp_dict[ i ] = c
            c+= 1
        
        df_data['itemId'].replace( temp_dict, inplace = True )

        self.itemId_map = dict((str(k), v) for k, v in temp_dict.items())

        return df_data


    """
        get a list of split factors to apply iteratively from a full list of slip factors
        ex: [0.8, 0.1, 0.1] => [ [0.8, 0.2], [0.5, 0.5] ]
    """
    def _full_split_to_iterative_split(self):

        iterative_split = []
        iterative_split.append( [ self.split[0], sum(self.split[1:]) ] )

        for i in range(1, len(self.split)-1):
            iterative_split.append( [ self.split[i]/sum(self.split[i:]), 1-(self.split[i]/sum(self.split[i:])) ] )

        return iterative_split

    
    """
        create a set of random matrix with provided split probability
        and apply successively to obtain a set of split dataset.
    """
    def split(self, split_factor):

        class CustomRandomState(np.random.RandomState):
            def randint(self, k):
                i = np.random.randint(k)
                return i - i % 2
        np.random.seed(12345)
        self.random_seed = CustomRandomState()

        mask_matrix = sparse.random(self.nb_user+2, self.nb_item+2, density=split_factor, random_state=self.random_seed)

        # check stat
        #nb = mask_matrix.shape[0] * mask_matrix.shape[1]
        #proportion = mask_matrix.getnnz() / nb
        #print(nb, proportion)

        one = self.csr_data.multiply(mask_matrix)

        # check nb value per vec
        #print( one.getnnz(0) )
        #print()
        #print( one.getnnz(1) )

        return (
            RatingDataset(one,"split"+str(split_factor), view=self._view, index_user=self.index_user, index_item=self.index_item),
            RatingDataset(self.csr_data - one,"split"+str(1-split_factor), view=self._view, index_user=self.index_user, index_item=self.index_item)
        )


    """
        find a suitable set of mask matrix to apply successively, such that
        every column AND row has at least one rating in each of split dataset
        resulting from those masks
    """
    def _smart_split(self):

        # first remove every 
        pass