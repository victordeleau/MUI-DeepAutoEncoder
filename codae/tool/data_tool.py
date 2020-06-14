
import random
import hashlib
import glob
import os, sys
import pickle
import json
import itertools

import torch
import numpy as np

from codae.dataset import ConcatenatedEmbeddingDataset


class Normalizer:

    def __init__(self, normalization_type="min_max"):
        """
        input
            normalization_type : str
                how to normalize (min_max/...)
        """

        self.normalization_type = normalization_type

        self.min = None
        self.max = None
        self.scale = None
        self.is_fitted = False


    def fit(self, data, mask=None):
        """
        fit normalization model to data
        input
            data : (np.ndarray/torch.Tensor)
                data to normalize
            mask : (np.ndarray/torch.Tensor)
                optionnal mask to apply (to avoid normalizing nominal variables for ex)
        """   

        if self.normalization_type == "min_max":

            if isinstance(data, np.ndarray):

                if mask == None:
                    self.mask = torch.ones((self.data.shape()[0]))
                else:
                    self.mask = mask

                self.min = data.min(0)[0] * self.mask
                self.max = data.max(0)[0] * self.mask
                self.scale = self.max - self.min
                self.scale[self.scale == 0] = 1

            elif isinstance(data, torch.Tensor):

                if mask == None:
                    self.mask = torch.ones((self.data.size()[0]))
                else:
                    self.mask = mask

                self.min = data.min(0, keepdim=True)[0] * self.mask
                self.max = data.max(0, keepdim=True)[0] * self.mask
                self.scale = self.max - self.min
                self.scale[self.scale == 0] = 1

            else:
                raise Exception("Error: dataset type not suppored (np.Array/torchTensor).")

        self.is_fitted = True

        return self

    
    def normalize(self, data):
        """
        normalize data using previously fitted model
        input 
            data : (np.ndarray/torch.Tensor)
                data to normalize
        """

        if not self.is_fitted:
            raise Exception("Error: normalizer is not fitted.")

        return (data - self.min) / self.scale
        

    def denormalize(self, data):
        """
        denormalize data using previously fitted model
        input 
            data : (np.ndarray/torch.Tensor)
                data to denormalize
        """

        if not self.is_fitted:
            raise Exception("Error: normalizer is not fitted.")

        return ((data * self.scale) + self.min)




def collate_embedding(batch):
    """
    unzip and merge list of torch.Tensor into single Tensor by stacking them
    """

    batch, indices = zip(*batch)

    return torch.stack(batch), indices


def simple_collate(batch):
    """
    merge list of torch.Tensor into single Tensor by stacking them
    """

    return torch.stack(batch)


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


class Corrupter:
    """
    Create, handle, and keep track of corruption masks.
    """

    def __init__(self, nb_observation, arch, k_max, device):
        """
        input
            nb_observation : int
            arch : int
            k_max : int
            device : torch.device
        """

        self.nb_observation = nb_observation
        self.arch = arch
        self.k_max = k_max
        self.device = device

        if (k_max < 0) | (k_max > len(self.arch)-1):
            raise Exception("Invalid k_max number. k_max > 0 && k_max < nb_predictor - 1")

        self.io_size = sum([v["size"] for v in self.arch])
        self.nb_predictor = len(self.arch)

        # build binary mask tensors ############################################

        # get binomial coefficent indexes
        indices = [i for i in range(self.nb_predictor)]
        binomial_coef_indices = []
        self.nb_corruption_per_k = [0 for x in range(0, k_max)]
        for k in range(0, k_max):
            binomial_coef_indices.append( torch.LongTensor(list(itertools.combinations(indices, k+1))) )
            self.nb_corruption_per_k[k] = len(binomial_coef_indices[-1])
        self.nb_run = sum(self.nb_corruption_per_k)
        #print(self.nb_corruption_per_k)

        # dictionnary of stacked binary masks tensors
        binary_masks = []
        for k, bci in enumerate(binomial_coef_indices): # for k missing variable
            for subset in bci: # for subset of indices of size k
                tmp = torch.ones((self.io_size))
                for idx in subset: # for each corrupted variable
                    tmp[self.arch[idx]["position"]:self.arch[idx]["position"]+self.arch[idx]["size"]] = 0
                binary_masks.append(tmp)
        self.binary_masks = torch.stack(binary_masks)

        # build randomized index of mask to use per observation ################

        # nb missing variable to pick at each augmentation run
        self.nb_missing_per_run = [0 for x in range(self.nb_run)]
        j = 0
        for k in range(k_max):
            for i in range(self.nb_corruption_per_k[k]):
                self.nb_missing_per_run[j] = k+1
                j += 1

        # shuffle nb of missing variable to pick at each run per observation
        mask_to_use = []
        self.corrupted_index = [x for x in range(self.nb_run)]
        for i in range(nb_observation):
            mask_to_use.append( torch.LongTensor(random.sample(self.corrupted_index, self.nb_run)) )
        self.mask_to_use = torch.stack(mask_to_use)



    def get_masks(self, batch_indices, run):
        """
        Return list of [k_max] corruption masks as boolean tensors.
        input
            batch_indices : list(int)
                list of observation index
            run : int
                current augmentation run
        output
            masks : list(torch.Tensor)
                list of binary mask tensor for each [nb_missing_variable]
        """

        masks = [torch.zeros((len(batch_indices), self.io_size), device=self.device) for x in range(self.k_max)]

        for i, idx in enumerate(batch_indices): # for each observation in batch

            # get nb missing variable
            k = self.nb_missing_per_run[self.mask_to_use[idx][run]]-1

            # extract masks
            masks[k][i, :] = self.binary_masks[self.mask_to_use[idx][run]]

        return masks, torch.sum(torch.stack(masks, dim=0), dim=0)