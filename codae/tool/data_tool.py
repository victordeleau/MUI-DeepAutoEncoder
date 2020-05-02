#

import random
import hashlib
import glob
import os, sys
import pickle
import json

import torch
import numpy as np

from codae.dataset import ConcatenatedEmbeddingDataset


def my_collate(batch):
    """
    merge list of torch.Tensor into single Tensor by stacking them
    """

    batch, indices = zip(*batch)

    return torch.stack(batch), indices


def load_dataset_of_embeddings(embedding_path, config, cache_dir="tmp/"):

    using_cache = False
    dataset_cache = glob.glob(os.path.join(cache_dir, "*_dataset.bin"))

    if len(dataset_cache) > 0: # check for cached dataset

        dataset_cache_path = dataset_cache[0]
        dataset_hash = dataset_cache_path.split("/")[-1].split("_")[0]

        # load cached dataset if corresponding embedding file hasn't changed
        if dataset_hash == hashlib.sha1( str( os.stat(
            embedding_path)[9]).encode('utf-8') ).hexdigest():

            #args.log.info("Using cached dataset.")
            using_cache = True

            try:
                with open(dataset_cache_path, 'rb') as f:
                    dataset = pickle.load(f)
            except:
                raise Exception("Error while reading embedding json file.")

        else: # delete old cached dataset
            os.remove(dataset_cache_path)

    if not using_cache: # then load from json of embeddings
        
        try:
            with open(embedding_path, 'r') as f:
                embeddings = json.load(f)
        except:
            raise Exception("Error while reading embedding json file.")

        dataset = ConcatenatedEmbeddingDataset(
            embeddings=embeddings,
            used_category=config["DATASET"]["USED_CATEGORY"])

        # write new cache to disk
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, "new_dataset_tmp.bin"), "wb") as f:
            pickle.dump( dataset, f )
        dataset_cache_name = os.path.join(cache_dir, hashlib.sha1( str(\
            os.stat(embedding_path)[9])\
            .encode('utf-8') ).hexdigest() + "_dataset.bin")
        os.rename(os.path.join(cache_dir, "new_dataset_tmp.bin"), dataset_cache_name)

    return dataset