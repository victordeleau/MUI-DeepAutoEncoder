# autoencoder base class

import math
import random

import torch
import numpy as np

class EmbeddingDenoisingAutoencoder(torch.nn.Module):
    
    def __init__(self, io_size, z_size, embedding_size, nb_input_layer=2, nb_output_layer=2, steep_layer_size=True, activation=torch.nn.ReLU):
        """input
            io_size : int
                size of the input and output layer
                default : 512
            z_size : int
                size of the embedding layer
                default : 128
            nb_input_layer : int
                number of layer for the encoder
                default : 2
            nb_output_layer : int
                number of layer for the decoder
                default : 2
            steep_layer_size : boolean
                if whether or not layer size should decrease/increase up to z_size 
            activation : torch.nn.Module
                the activation function to use
        """

        super(EmbeddingDenoisingAutoencoder, self).__init__()

        if io_size % embedding_size != 0:
            raise Exception("Error: io_size must be a multiple of embedding_size")

        self.embedding_size = embedding_size
        self.nb_category = io_size / embedding_size
        self.io_size = io_size
        self.z_size = z_size
        
        self.nb_input_layer = nb_input_layer
        self.nb_output_layer = nb_output_layer
        self.steep_layer_size = steep_layer_size
        self.activation = activation

        # default to train mode 0 (1 for validation mode, 2 for test mode)
        self.mode = 0 

        input_layer_increment = 0
        output_layer_increment = 0
        if not self.steep_layer_size: # set layers steepiness
            delta = self.io_size - self.z_size
            input_layer_increment = math.floor( delta / self.nb_input_layer )
            output_layer_increment = math.floor( delta / self.nb_output_layer )

        ########################################################################
        # define encoder #######################################################

        input_layer = []

        # first layer, always there
        #next_layer_input_size = io_size-input_layer_increment
        input_layer.append( torch.nn.Linear(io_size, io_size) )
        input_layer.append( activation(True) )
        
        # middle encoding layers
        for i in range(1, nb_input_layer):

            if self.steep_layer_size:
                input_layer.append( torch.nn.Linear(io_size, io_size) )
                input_layer.append( activation(True) )

            else:
                next_layer_input_size = max(io_size-((i-1)*input_layer_increment), z_size)

                next_layer_output_size = max(io_size-(i*input_layer_increment), z_size)

                input_layer.append(
                    torch.nn.Linear(
                        next_layer_input_size,
                        next_layer_output_size ) )

                input_layer.append( activation(True) )

        # embedding layer, always there
        if self.steep_layer_size:
            input_layer.append( torch.nn.Linear( io_size, z_size) )
        else:
            input_layer.append( torch.nn.Linear( next_layer_output_size, z_size) )

        self.input_layer = torch.nn.Sequential( *input_layer ) # join encoder layers

        # initialize layer's weights and biases
        self.input_layer.apply(self.init_weight_general_rule)
        self.input_layer.apply(self.init_bias_zero)


        ########################################################################
        # define decoder #######################################################

        output_layer = []

        for i in range(nb_output_layer):
            if self.steep_layer_size:
                if i == 0:
                    output_layer.append( torch.nn.Linear(z_size, io_size ) )
                    output_layer.append( activation(True) )
                else:
                    output_layer.append( torch.nn.Linear(io_size, io_size ) )
                    output_layer.append( activation(True) )
            else:
                next_layer_input_size = min(z_size+(i*output_layer_increment), io_size)

                next_layer_output_size = min(z_size+((i+1)*output_layer_increment), io_size)
                
                output_layer.append(
                    torch.nn.Linear(
                        next_layer_input_size,
                        next_layer_output_size) )

                output_layer.append( activation(True) )

        # make sure last layer has correct size
        
        # last layer, always there
        output_layer.append(torch.nn.Linear(next_layer_output_size,io_size) )

        # join decoder layers
        self.output_layer = torch.nn.Sequential( *output_layer ) 

        # initialize layer's weights and biases
        self.output_layer.apply(self.init_weight_general_rule)
        self.output_layer.apply(self.init_bias_zero)


    
    def forward(self, x):
        """
        apply forward pass to input vector x
        input
            x : torch.Tensor
                input vector of size self.io_size
        output
            y : torch.Tensor
                output vector of size self.io_size
        """

        z = self.encode(x)
        y = self.decode(z)

        return y


    
    def encode(self, x):
        """
        take input vector x and encode to z
        input
            x : torch.Tensor
                input vector of size self.io_size to encode
        output
            z : torch.Tensor
                embedding vector of size self.z_size
        """

        z = self.input_layer(x)

        return z
        

    def decode(self, z):
        """
        take compressed vector z and decode to y
        input
            z : torch.Tensor
                embedding vector of size self.z_size to decode
        output
            y : torch.Tensor
                output vector of size self.io_size

        """
        
        y = self.output_layer(z)

        return y


    def init_weight_general_rule(self, m):
        """
        initialize the auto encoder's weight using general rule
        input
            m : nn.Linear module to initialize
        """

        if m.__class__.__name__.find('Linear') != -1:

            torch.nn.init.xavier_uniform_(m.weight)
        
        
    def init_bias_zero(self, m):
        """
        initialize the auto encoder's biases with zeros
        input
            m : nn.Linear module to initialize
        """

        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            
            m.bias.data.fill_(0)


    def to(self, *args, **kwargs):
        """
        override .to() to make sure custom layers are sent to the correct device
        """

        self = super().to(*args, **kwargs) 
        self.input_layer = self.input_layer.to(*args, **kwargs)
        self.output_layer = self.output_layer.to(*args, **kwargs) 
        
        return self


    def corrupt(self, input_data, mask):
        """
        corrupt input_data using requested corruption type
        input
            input_data : torch.Tensor
                the piece of data to corrupt
            mask : torch.Tensor
                list of embedding index to corrupt
        output
            corrupted_input : torch.Tensor
                the corrupted input data
        """

        return input_data.clone() * mask