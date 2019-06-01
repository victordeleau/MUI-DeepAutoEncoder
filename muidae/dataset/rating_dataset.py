# load and prepare dataset

import pandas as pd
import math
import logging
from sklearn.model_selection import train_test_split
import torchvision
import torch
import scipy.sparse as sparse
import numpy as np
from random import shuffle
from torch.utils.data.dataset import Dataset as PytorchDataset


"""
    load provided dataset inside object, provide useful methods and information
    input
        ...
"""
class RatingDataset(PytorchDataset):

    def __init__(self, data, name, view=None, has_been_randomized=None, is_sub_dataset=False, normalize=True, is_normalized=False,
            user_index_swap=None, item_index_swap=None,
            userId_map=None, itemId_map=None,
            index_user=None, index_item=None,
            nb_user=None, nb_item=None):

        self.name = (name if name != None else name)

        self.has_been_randomized = (has_been_randomized if has_been_randomized!=None else False)
        
        self._is_sub_dataset = (is_sub_dataset if is_sub_dataset!=None else False)

        self._view = (view if view != None else "user_view")

        self.iterator_count = 0
        self.column_id = None
        self.row_id = None 

        if not is_sub_dataset:

            assert( isinstance(data, pd.DataFrame) )
        
            self.userId_map = None
            self.itemId_map = None

            data = self._map_index_to_monotonic(data)

            self.data = sparse.csr_matrix(
                (
                    data.values[:, 2],
                    (data.values[:, 0].astype(int),
                    data.values[:, 1].astype(int))
                )
            )

            self.index_user = data.userId.unique()
            self.index_item = data.itemId.unique()

            self.nb_user = len( self.index_user )
            self.nb_item = len( self.index_item )

            self.user_index_swap = (np.arange(self.nb_user) if user_index_swap==None else user_index_swap)
            self.item_index_swap = (np.arange(self.nb_item) if item_index_swap==None else item_index_swap)

            if normalize and not is_normalized:
                self.normalize()
                self.is_normalized = True

        else:

                assert isinstance(data, sparse.csr_matrix)

                self.userId_map = userId_map
                self.itemId_map = itemId_map

                self.index_user = index_user
                self.index_item = index_item
                
                self.user_index_swap = user_index_swap
                self.item_index_swap = item_index_swap

                self.nb_user = nb_user
                self.nb_item = nb_item

                self.data = data

        self._io_size = (self.nb_item+1 if self._view == "user_view" else self.nb_user+1 )


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

        
        if self._view == "item_view":

            self.iterator_count += 1

            if self.iterator_count > self.nb_item:
                raise StopIteration

            return self.__getitem__(self.iterator_count-1)


        elif self._view == "user_view":

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
        
        if self._view == "item_view":
            return self.nb_item

        elif self._view == "user_view":
            return self.nb_user

        else:
            return 0


    """
        allow index access of the dataset (overide required from abstract parent class)
        input
            idx: index of the vector we want to grab
    """
    def __getitem__(self, idx):

        if self._view == "item_view":
            
            if idx < 0 or idx > self.nb_item:
                return 0
            else:
                swap_idx = self.item_index_swap[idx]
                return np.ravel( self.data[:,swap_idx].todense() )

        elif self._view == "user_view":
            
            if idx < 0 or idx > self.nb_user:
                return 0
            else:
                swap_idx = self.user_index_swap[idx]
                return np.ravel( self.data[swap_idx,:].todense() )

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

        gm, um, im = 0, 0, 0

        if global_mean == True:
            gm = self.data.mean()

        if user_mean == True:
            um = self.data.mean(axis=0).tolist()[0]
        
        if item_mean == True:
            im = self.data.mean(axis=1).T.tolist()[0]

        non_zero = self.data.nonzero()
        for i,j in zip(*non_zero):
            self.data[i, j] -= gm + um[j] + im[i]


    """
        create random swap of row and column indices
        don't randomize and return 0 if self.has_been_randomized==True
        typically if current dataset is a subset of another dataset that has
        been previously randomized.
    """
    def randomize(self):

        if not self.has_been_randomized:

            self.user_index_swap = shuffle(np.arrange(self.nb_user))
            self.item_index_swap = shuffle(np.arrange(self.nb_item))
        
            return self.user_index_swap, self.item_index_swap

        return 0


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
        split the dataset into two sub datasets, to create training/validation/testing sets
        input
            split_factor: between 0 and 1, default to 0.8
        output
            tuple of two RatingDataset, subset of this
    """
    def get_split_sets(self, split_factor=0.8, view=None):

        if split_factor >= 1 or split_factor <= 0:
            return 0

        first_dataset, second_dataset = train_test_split(self.data, test_size=1-split_factor)
        first_nb_user, first_nb_item = math.floor( self.nb_user * split_factor), math.floor( self.nb_item * split_factor )
        second_nb_user, second_nb_item = math.floor( self.nb_user * (1-split_factor)), math.floor( self.nb_item * (1-split_factor) )

        return (
                RatingDataset(
                    first_dataset,
                    name=self.name+"_subset",
                    has_been_randomized=True,
                    is_sub_dataset=True,
                    user_index_swap=self.user_index_swap, item_index_swap=self.item_index_swap,
                    userId_map=self.userId_map, itemId_map=self.itemId_map,
                    index_user=self.index_user, index_item=self.index_item,
                    nb_user=first_nb_user, nb_item=first_nb_item,
                    view=self._view),
                    
                RatingDataset(
                    second_dataset,
                    name=self.name+"_subset",
                    has_been_randomized=True,
                    is_sub_dataset=True,
                    user_index_swap=self.user_index_swap, item_index_swap=self.item_index_swap,
                    userId_map=self.userId_map, itemId_map=self.itemId_map,
                    index_user=self.index_user, index_item=self.index_item,
                    nb_user=second_nb_user, nb_item=second_nb_item,
                    view=self._view)
                )