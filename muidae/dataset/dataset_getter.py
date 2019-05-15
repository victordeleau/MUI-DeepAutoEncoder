# download and prepare movielens 100k dataset

import os, sys
import requests
import pandas as pd
import logging
import pickle
from zipfile import ZipFile
from io import StringIO

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
                'year': 2016,
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ['userId','itemId','rating'],
                'rename_column': {'movieId': 'itemId'}
            }),
            self._dataset_name[1]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                'year': 2003,
                'delimiter': '::',
                'rating_file': 'ratings.dat',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[2]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
                'year': 2009,
                'delimiter': '::',
                'rating_file': 'ratings.dat',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[3]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
                'year': 2016,
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[4]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-latest.zip',
                'year': 2017,
                'youtube': 'http://files.grouplens.org/datasets/movielens/ml-20m-youtube.zip',
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[5]: Dict({
                'url': 'http://files.grouplens.org/datasets/serendipity-sac2018/serendipity-sac2018.zip',
                'year': 2018,
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._dataset_name[6]: Dict({
                'url': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                'year': 1998,
                'delimiter': '\t',
                'rating_file': 'u.data',    
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            })
        })


    """
        download specified dataset locally and return a Dataset object
    """
    def get(self, dataset_name='100k', dataset_location="../data/", export_to_disk=False):

        if dataset_name not in self._dataset_name:
            raise Exception("Error: provided dataset name is not available ('100k', '1m', '10m', '20m', '26m', 'serendipity', '100k_old')")

        dataset_zipfile = self._download_dataset( self._dataset_info[dataset_name].url )

        file_name = self._dataset_info[dataset_name].file.split('/')[-1]

        zip_file = ZipFile( dataset_zipfile )

        extracted_files = {name: zip_file.read(name) for name in zip_file.namelist()}

        rating_file = extracted_files[ self._dataset_info[dataset_name].rating_file ]

        df_data = pd.read_csv(
            StringIO( str(rating_file,'utf-8') ),
            sep=self._dataset_info[dataset_name].delimiter
        )

        if self._dataset_info[dataset_name].rename_column != None:
            df_data = df_data.rename(index=str, columns=self._dataset_info[dataset_name].rename_column)

        if self._dataset_info[dataset_name].keep_column != None:
            df_data = df_data[["userId","itemId","rating"]]

        dataset = RatingDataset(df_data, name=dataset_name)

        if export_to_disk:

            if not os.path.exists(dataset_location):
                os.makedirs(dataset_location)

            self.save_dataset_object_to_disk(dataset, file_name, dataset_location)

        return dataset

    
    """
        download specified dataset from the network
    """
    def _download_dataset(self, dataset_url):

        cnt = 0
        downloaded = False
        while cnt != 3 and not downloaded:
            try:
                dataset = requests.get( dataset_url )
                downloaded = True
            except:
                cnt += 1
        
        if not downloaded:
            raise Exception("Was not able to download the dataset after 3 tentatives.")

        return dataset


    """
        save provided dataset to disk
    """
    def save_dataset_to_disk(self, dataset, file_name, dataset_location):

        with open(dataset_location + file_name, 'wb') as f:
            f.write(dataset.content)
        
        

        with zipfile.ZipFile(dataset_location + file_name) as out_f:
            out_f.extractall(dataset_location)

        os.remove(dataset_location + file_name)


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
    def local_dataset_found(self, dataset_name='100k', dataset_location="data/pickle/"):

        if os.path.exists(dataset_location + dataset_name + ".dump"):  # if local dataset object exist, load it
            return True
        
        return None

    
    """
        load local dataset using pickle
        input
            dataset_location: where to look for a dataset to pickle
    """
    def load_local_dataset_object(self, dataset_name='100k', dataset_location="data/pickle/"):

        try:
            with open(dataset_location + dataset_name + ".dump", 'rb') as f:
                return pickle.load( f )

        except Exception as e: 
            sys.out.println("Error while unpickling local dataset ...")
            raise e


    """
        export dataset object to disk
        input
            dataset: object of class Dataset to export
            dataset_location: where to export the dataset
    """
    def export_dataset_object_to_disk(self, dataset, dataset_location="data/pickle/"):

        if not os.path.exists("data"):
            os.makedirs("data/")
            
        if not os.path.exists("data/pickle"):
            os.makedirs("data/pickle")

        with open(dataset_location + dataset.name + ".dump", 'wb') as f:

            pickle.dump( dataset, f )





