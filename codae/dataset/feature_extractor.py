
import os, sys

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image


class FeatureExtractor(object):

    def __init__(self, model="resnet50"):

        self.model_name = model

        if self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        else:
            raise Exception("Unknown model name.")


        # extract avgpool layer
        self.layer = self.model._modules.get('avgpool')

        # to avoid dropouts layers are inactive
        self.model.eval() 

        # extract image preprocessing functions
        self._scale = transforms.Scale((224, 224))
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._to_tensor = transforms.ToTensor()


    def _copy_last_layer(self, i, o):
        """
        A hook to extract data from the last layer of the model
        """
        
        output_tensor.copy_(o.data)


    def encode(self, image):
        """
        extract vector of feature from image
        input
            image: PIL Image
        output
            A numpy 1D array of lenght 512
        """

        # make sure image is in PIL format
        assert(isinstance(image, Image))

        input_tensor = Variable( # image preprocessing
            self._normalize(
                self._to_tensor(
                    self._scale(image))).unsqueeze(0))

        # create output vector
        output_tensor = torch.zeros(512)

        # register hook to access last layer data
        hook = self.layer.register_forward_hook(self._copy_last_layer)

        # encode image
        self.model(input_tensor)

        # detach hook
        hook.remove()

        return output_tensor.data.numpy()