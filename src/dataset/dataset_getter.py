# download and prepare movielens 100k dataset

import os, sys
import requests
import pandas as pd
import logging
import pickle

sys.path.insert(0,os.path.pardir) 

from tool.dictionnary import Dict
from dataset.dataset import RatingDataset


"""
    download and return specified dataset into a Dataset object
"""
class dataset_getter(object):

    def __init__(self):

        self._ds_name = ['100k', '1m', '10m', '20m', '26m', 'serendipity', '100k_old']

        self._ds_info = Dict({
            self._ds_name[0]: Dict({
                'file': 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
                'year': 2016,
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ['userId','itemId','rating'],
                'rename_column': {'movieId': 'itemId'}
            }),
            self._ds_name[1]: Dict({
                'file': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                'year': 2003,
                'delimiter': '::',
                'rating_file': 'ratings.dat',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._ds_name[2]: Dict({
                'file': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
                'year': 2009,
                'delimiter': '::',
                'rating_file': 'ratings.dat',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._ds_name[3]: Dict({
                'file': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
                'year': 2016,
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._ds_name[4]: Dict({
                'file': 'http://files.grouplens.org/datasets/movielens/ml-latest.zip',
                'year': 2017,
                'youtube': 'http://files.grouplens.org/datasets/movielens/ml-20m-youtube.zip',
                'delimiter': ',',
                'rating_file': 'ratings.csv',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._ds_name[5]: Dict({
                'file': 'http://files.grouplens.org/datasets/serendipity-sac2018/serendipity-sac2018.zip',
                'year': 2018,
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            }),
            self._ds_name[6]: Dict({
                'file': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                'year': 1998,
                'delimiter': '\t',
                'rating_file': 'u.data',
                'keep_column': ["userId","itemId","rating"],
                'rename_column': {"movieId": "itemId"}
            })
        })


    """
        download specified dataset lcoally and return a Dataset object
    """
    def get(self, dataset_name='100k', dataset_location="../data/"):

        ############################################################################################
        # download locally #########################################################################

        if dataset_name not in self._ds_name:

            logging.error("Error: provided dataset name is not available ('100k', '1m', '10m', '20m', '26m', 'serendipity', '100k_old')")

            raise Exception("Error: provided dataset name is not available ('100k', '1m', '10m', '20m', '26m', 'serendipity', '100k_old')")

        file_name = self._ds_info[dataset_name].file.split('/')[-1]

        if not os.path.exists(dataset_location): # create 'data' folder if doesn't exist
            os.makedirs(dataset_location)

        # download specified dataset if necessary
        if not os.path.exists(dataset_location + file_name):

            logging.info("Retrieving %s dataset ..." %(dataset_name))
            
            data = requests.get(self._ds_info[dataset_name].file)
            with open(dataset_location + file_name, 'wb') as f:
                f.write(data.content)
            
            import zipfile

            with zipfile.ZipFile(dataset_location + file_name) as out_f:
                out_f.extractall(dataset_location)

            os.remove(dataset_location + file_name)

        else:
            logging.info("Provided dataset name already in folder %s" %(dataset_location))
        
        # load local dataset into pandas.Dataframe
        df_data = pd.read_csv(dataset_location + file_name.split(".")[0] + "/" + self._ds_info[dataset_name].rating_file, sep=self._ds_info[dataset_name].delimiter)

        # rename columns if necessary
        if self._ds_info[dataset_name].rename_column != None:
            df_data = df_data.rename(index=str, columns=self._ds_info[dataset_name].rename_column)

        # keep necesary columns if necessary
        if self._ds_info[dataset_name].keep_column != None:
            df_data = df_data[["userId","itemId","rating"]]
            pass

        return RatingDataset(df_data, dataset_name)


    """
        return a list of available dataset to download
        output
            list of available dataset as string
    """
    def available(self):

        return self.ds_name

    """
        search locally for a saved dataset object to pickle
    """
    def local_dataset_found(self, dataset_name='100k', dataset_location="data/pickle/"):

        if os.path.exists(dataset_location + dataset_name + ".dump"):  # if local dataset object exist, load it

            with open(dataset_location + dataset_name + ".dump", 'rb') as f:

                return pickle.load( f )
        
        return None

    
    """
        load local dataset using pickle
        input
            dataset_location: where to look for a dataset to pickle
    """
    def load_local_dataset(self, dataset_name='100k', dataset_location="data/pickle/"):

        try:
            with open(dataset_location + dataset_name + ".dump", 'rb') as f:

                return pickle.load( f )

        except Exception as e: 
            System.out.println("Error while unpicling local dataset ...")
            sys.exit(0)


    """
        export dataset object to disk
        input
            dataset: object of class Dataset to export
            dataset_location: where to export the dataset
    """
    def export_dataset(self, dataset, dataset_location="data/pickle/"):

        # create folder if doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data/")
            
        if not os.path.exists("data/pickle"):
            os.makedirs("data/pickle")

        with open(dataset_location + dataset.name + ".dump", 'wb') as f:

            pickle.dump( dataset, f )





