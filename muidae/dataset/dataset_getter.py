# download and prepare movielens 100k dataset

import os, sys, io
import requests
import pandas as pd
import logging
import pickle
from zipfile import ZipFile
from io import StringIO
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import math

sys.path.insert(0,os.path.pardir) 

from muidae.tool.dictionnary import Dict
from muidae.dataset.rating_dataset import RatingDataset


"""
    download and return specified dataset into a Dataset object
"""
class DatasetGetter(object):

    def __init__(self):

        self._dataset_name = ['100k', '1m', '10m', '20m', '26m', 'serendipity', '100k_old']

        self._dataset_info = Dict({
            self._dataset_name[0]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
                'filename': 'ml-latest-small.zip',
                'year': 2016,
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ['userId','itemId','rating'],
                'rename_column': {'movieId': 'itemId'}
            }),
            self._dataset_name[1]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                'filename': 'ml-1m.zip',
                'year': 2003,
                'delimiter': '::',
                'rating_file': 'ratings.dat',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[2]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
                'filename': 'ml-10m.zip',
                'year': 2009,
                'delimiter': '::',
                'rating_file': 'ratings.dat',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[3]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
                'filename': 'ml-20m.zip',
                'year': 2016,
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[4]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-latest.zip',
                'filename': 'ml-latest.zip',
                'year': 2017,
                'youtube': 'http://files.grouplens.org/datasets/movielens/ml-20m-youtube.zip',
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[5]: Dict({
                'url': 'http://files.grouplens.org/datasets/serendipity-sac2018/serendipity-sac2018.zip',
                'filename': 'serendipity-sac2018.zip',
                'year': 2018,
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[6]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                'filename': 'ml-100k.zip',
                'year': 1998,
                'delimiter': '\t',
                'rating_file': 'u.data',    
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            })
        })
    
    """
        download specified dataset from the network
    """
    def download_dataset(self, dataset_name="100k", dataset_location="data/" ):

        cnt = 0
        downloaded = False
        while cnt != 3 and not downloaded:
            try:
                res = requests.get( self._dataset_info[dataset_name].url )
                downloaded = True
            except:
                cnt += 1
        
        if not downloaded:
            raise Exception("Was not able to download the dataset after 3 tentatives.")

        if not os.path.exists(dataset_location):
            os.mkdir(dataset_location)

        with open(dataset_location + dataset_name + ".zip", 'wb') as f:
            f.write(res.content)


    """
        return a list of available dataset to download
        output
            list of available dataset as string
    """
    def get_available_dataset(self):

        return self.ds_name


    """
        search locally for a saved dataset object to pickle
    """
    def local_dataset_found(self, dataset_name='100k', dataset_location="data/"):

        if os.path.exists(dataset_location + dataset_name + ".zip"):
            return True
        
        return False

    
    """
        load local dataset using pickle
        input
            dataset_location: where to look for a dataset to pickle
        output
            a RatingDataset object containing the specified dataset
    """
    def load_local_dataset(self, dataset_name='100k', dataset_location="data/", view="item_view", store_as_binary=True, try_load_binary=True):

        if os.path.exists(dataset_location + dataset_name + "_" + view + "_norm.bin" ) and try_load_binary:

            return self._load_binary_rating_dataset(
                dataset_location + dataset_name + "_" + view + "_norm.bin"
            )

        else:
            path = dataset_location + dataset_name + ".zip"

            try:
                os.path.exists(path)
            except:
                raise Exception("Local dataset not found in folder " + dataset_location)

            zip_file = None
            try:
                zip_file = ZipFile( path, 'r' )
            except Exception:
                raise Exception("File is not a Zip file.")

            extracted_files = {name: zip_file.read(name) for name in zip_file.namelist()}

            rating_file = None
            try:
                rating_file = extracted_files[ self._dataset_info[dataset_name].filename.split('.')[0] + '/' + self._dataset_info[dataset_name].rating_file ]
            except Exception:
                raise Exception("Data file not found in Zip archive.")

            df_data = None
            try:
                df_data = pd.read_csv(
                    StringIO( str(rating_file,'utf-8') ),
                    sep=self._dataset_info[dataset_name].delimiter
                )
            except Exception:
                raise Exception("Data file is not a .csv file.")

            if self._dataset_info[dataset_name].rename_column != None:
                try:
                    df_data = df_data.rename(index=str, columns=self._dataset_info[dataset_name].rename_column)
                except Exception:
                    raise Exception("Provided column to rename not found.")

            if self._dataset_info[dataset_name].keep_column != None:
                try:
                    df_data = df_data[["userId","itemId","rating"]]
                except:
                    raise Exception("One or more provided column name to keep not found.")

            dataset = RatingDataset(df_data, name=dataset_name, view=view)

            if store_as_binary:
                self._store_rating_dataset_as_binary(
                    dataset,
                    dataset_location + dataset_name + "_" + view + "_norm.bin"
                )    

            return dataset


    """
        Take RatingDataset as input and return a DataLoader for it
        input
            dataset: a RatingDataset
            redux: percentage of data to use [0:1]
            batch_size
            nb_worker: for parralel data preparation
            shuffle: boolean, the data 
        output
            a DataLoader object

    """
    def get_dataset_loader(self, dataset, batch_size=1, nb_worker=4, shuffle=True):

        if not isinstance(dataset, RatingDataset):
            raise Exception("Provided dataset is not a RatingDataset object")

        dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nb_worker,)

        return dataset_loader

    
    def _store_rating_dataset_as_binary(self, dataset, path):

        with open(path, 'wb') as f:
            pickle.dump(dataset, f)


    def _load_binary_rating_dataset(self, path):

        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        return dataset