
import os, sys

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image


class FeatureExtractor(object):

    def __init__(self, model="resnet18"):

        self.model_name = model

        if self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        else:
            raise Exception("Unknown model name.")

        if torch.cuda.is_available():
            self.model.cuda()

        # create output vector
        self.output_tensor = torch.zeros(512)

        def _copy_last_layer(m, i, o):
            """
            A hook to extract data from the last layer of the model
            """
            
            self.output_tensor.copy_(o.data.squeeze())

        # extract avgpool layer
        self.layer = self.model._modules.get('avgpool')

        # register hook to access last layer data
        hook = self.layer.register_forward_hook(_copy_last_layer)

        # to avoid dropouts layers are inactive
        self.model.eval() 

        # image preprocessing functions
        self._scale = transforms.Resize((224, 224))
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._to_tensor = transforms.ToTensor()


    def encode(self, image):
        """
        extract vector of feature from image
        input
            image: PIL Image
        output
            A numpy 1D array of lenght 512
        """

        # make sure image is in PIL format
        assert(isinstance(image, Image.Image))

        # if too small, resize
        size = image.size
        if size[0] < 224 or size[1] < 224:
            image = self._scale(image)

        # image preprocessing
        input_tensor = Variable( 
            self._normalize(
                self._to_tensor(image))
                    .unsqueeze(0)).cuda()

        self.model(input_tensor) # encode image

        return self.output_tensor.data.numpy()