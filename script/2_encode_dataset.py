
import sys, os
import json

import torchvision.models as models


class DatasetEncoder(object):
    """
    A class to extract embeddings out of a dataset using an autoencoder, or any other feature extractor
    """

    def __init__(self, output_path, model="resnet18"):
        """
        input
            output_path: str
                path to output folder on disk
            model: str
                name of the pretrained model to use
        """

        self.output_path = output_path
        self.model = model

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if model == "resnet18":
            self.model = models.resnet18()
        elif model == "resnet50":
            self.model = models.resnet50()
        else:
            raise Exception("Model name not found (resnet18/resnet50).")


    def to_json(self, image_path, sub_dir=True):
        """
        encode all image in path using pretrained model
        input
            image_path: str
                path to image dir on disk
            sub_dir: boolean
                look at sub folder for image or not (default: True)
        """

        # for all image in dir (and sub dir if sub_dir=True)

        # encode image

        # write to json


    def to_hdf5(self, image_path):

        pass