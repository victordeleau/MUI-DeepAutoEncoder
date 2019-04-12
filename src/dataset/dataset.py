# load and prepare dataset

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import torchvision
import torch
import scipy.sparse as sparse
import numpy as np


"""
    load provided dataset inside object, provide useful methods and information
"""
class RatingDataset(torch.utils.data.Dataset):

    def __init__(self, df_data, name="no_name", view="item_view"):

        self.name = name

        if not isinstance(df_data, pd.DataFrame):
            logging.error("Error: provided data is not a pandas.DataFrame object")
            raise Exception("Error: provided data is not a pandas.DataFrame object")

        df_data = self._map_index(df_data)

        self.count = 0 # for iterator

        # store in csr format
        self.data = sparse.csr_matrix(
            ( df_data.values[:, 2], (df_data.values[:, 0].astype(int), df_data.values[:, 1].astype(int)) )
        )

        self.index_user = df_data.userId.unique()
        self.index_item = df_data.itemId.unique()

        self.nb_user = len( self.index_user )
        self.nb_item = len( self.index_item )

        self.view = view

        self.is_mapped = False
        self.column_id = None
        self.row_id = None


    """
        return size of the dataset (overide required from abstract parent class)
    """
    def __len__(self):
        
        if self.view == "item_view":
            return self.nb_item

        elif self.view == "user_view":
            return self.nb_user

        else:
            return 0

    """
        allow index access of the dataset (overide required from abstract parent class)
    """
    def __getitem__(self, idx):

        if self.view == "item_view":
            
            if idx < 0 or idx > self.nb_item:
                return 0
            else: 
                return self.data[:,idx].todense()

        elif self.view == "user_view":
            
            if idx < 0 or idx > self.nb_user:
                return 0
            else:
                return self.data[idx,:].todense()

        else:
            return 0


    """
        set dataset view to either "item_view" or "user_view"
        in practise, return user or item vector
        input
            new_view: "item_view" or "user_view" string
    """
    def set_view(self, new_view):

        if new_view == "item_view":
            self.view = "item_view"
            self.count = 0
            return True

        elif new_view == "user_view":
            self.view = "user_view"
            self.count = 0
            return True

        else:
            return False


    """
        split the dataset in two according to split factor
        input
            factor: 0>factor>1 float value
        output
            p1: Dataset object
            p2: Dataset object
    """
    def split(self, factor=0.8):

        train_df, test_df = train_test_split(self.data, test_size=1-factor)

        return Dataset(train_df), Dataset(test_df)


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
            gm = self.data.mean() # compute global mean score

        if user_mean == True:
            um = self.data.mean(axis=0).tolist()[0]  # compute user mean score
        
        if item_mean == True:
            im = self.data.mean(axis=1).T.tolist()[0] # compute item mean score

        non_zero = self.data.nonzero()
        for i,j in zip(*non_zero):
            self.data[i, j] -= gm + um[j] + im[i]


    """
        map row and column name to monotonic index and save in dict (as string)
        input
            df_data: pandas.DataFrame to process
    """
    def _map_index(self, df_data):

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
    

    def __iter__(self):

        self.count = 0
        return self


    def __next__(self):

        
        if self.view == "item_view":

            if self.count > self.nb_item:
                raise StopIteration
            
            self.count += 1
            return self.__getitem__(self.count-1)


        elif self.view == "user_view":

            if self.count > self.nb_user:
                raise StopIteration
            
            self.count += 1
            return self.__getitem__(self.count-1)

        else:
            return 0
    




"""

ml_option = DatasetOption(g_root_dir='ds_movielens', g_save_dir=dataset, d_ds_name=dataset)


ml_ds_train = Movielens(ml_option)


ml_ds_train.download_and_process_data()


ml_ds_test = Movielens(ml_option, train=False)

"""