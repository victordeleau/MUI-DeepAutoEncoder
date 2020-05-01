# autoencoder base class

import math
import random

import torch
import numpy as np

class DenoisingAutoencoder(torch.nn.Module):
    
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

        super(DenoisingAutoencoder, self).__init__()

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
        for i in range(nb_input_layer-1):

            if self.steep_layer_size:
                input_layer.append( torch.nn.Linear(io_size, io_size) )
                input_layer.append( activation(True) )

            else:
                next_layer_input_size = io_size-(i*input_layer_increment)
                next_layer_output_size = io_size-((i+1)*input_layer_increment)

                input_layer.append(
                    torch.nn.Linear(
                        next_layer_input_size,
                        next_layer_output_size ) )

                input_layer.append( activation(True) )

        if self.steep_layer_size:
            input_layer.append( torch.nn.Linear( io_size, z_size) )
        else:
            input_layer.append( torch.nn.Linear(
                io_size-((self.nb_input_layer-1)*input_layer_increment), z_size) )
        input_layer.append( activation(True) )

        self.input_layer = torch.nn.Sequential( *input_layer ) # join encoder layers

        # initialize layer's weights and biases
        self.input_layer.apply(self.init_weight_general_rule)
        self.input_layer.apply(self.init_bias_zero)


        ########################################################################
        # define decoder #######################################################

        output_layer = []

        for i in range(nb_output_layer-1):

            if self.steep_layer_size:
                output_layer.append( torch.nn.Linear(io_size, io_size) )
                output_layer.append( activation(True) )

            else:
                next_layer_input_size = z_size+(i*output_layer_increment)
                next_layer_output_size = z_size+((i+1)*output_layer_increment)

                output_layer.append(
                    torch.nn.Linear(
                        next_layer_input_size,
                        next_layer_output_size) )

                output_layer.append( activation(True) )

        # make sure last layer has correct size
        output_layer.append(
            torch.nn.Linear(
                z_size+((nb_output_layer-1)*output_layer_increment),
                io_size) )
        output_layer.append( activation(True) )
        #
            
        self.output_layer = torch.nn.Sequential( *output_layer ) # join decoder layers

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
            
            #y = 1.0/math.sqrt( self.input_layer )
            #m.weight.data.uniform_(-y, y)

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


    def get_mmse_loss(self, input_data, output_data):
        """
        compute mask mean square error (mmse) and return loss
        use a mask if data is not normalized
        input
            input : torch.Tensor
            output : torch.Tensor
        """

        mmse_criterion = torch.nn.MSELoss(reduction='sum')

        # get index of values that aren't set to zero
        mask = input_data != 0.0

        # get number of uncorrupted input
        nb_rating = torch.sum( mask )

        loss = mmse_criterion(
            input_data,
            output_data * mask.float() ) 

        return loss / nb_rating.float()


    def to(self, *args, **kwargs):
        """
        override .to() to make sure custom layers are sent to the correct device
        """

        self = super().to(*args, **kwargs) 
        self.input_layer = self.input_layer.to(*args, **kwargs)
        self.output_layer = self.output_layer.to(*args, **kwargs) 
        
        return self


    def corrupt(self, input_data, device, nb_corrupted=1, corruption_type="zero_continuous"):
        """
        corrupt input_data using requested corruption type
        input
            input_data : torch.Tensor
                the piece of data to corrupt
            nb_corrupted : 0 < int < self.nb_category-1
                number of embeddings to corrupt (for zero_continuous corruption type only)
            corruption_type : str
                type of corruption to apply
        output
            corrupted_input : torch.Tensor
                the corrupted input data
            corrupted_indices : list(int)
                indices of corrupted categories
        """

        if corruption_type == "zero_continuous":
            return self._corrupt_zero_continuous(
                input_data=input_data,
                device=device,
                nb_corrupted=1)
        else:
            raise Exception("Error: invalid corruption type requested (zero_continuous).")


    def _corrupt_zero_continuous(self, input_data, device, nb_corrupted=1):
        """
        randomly set one of the input embeddings and set all values to zero.
        input
            input_data : torch.Tensor
                the piece of data to corrupt
            nb_corrupted : 0 < int < self.nb_category-1
                number of embeddings to corrupt
        output
            c_input : torch.Tensor
                the corrupted input data
            c_indices : list(int)
                indices of corrupted categories
        """

        if nb_corrupted >= self.nb_category:
            raise Exception("Error: too many corrupted embeddings requested (0 < nb_corrupted < self.nb_category-1).")

        c_indices = []
        c_input = input_data
        c_mask = torch.empty(
            input_data.size(),
            device=device)

        # for batch size 
        for i in range( input_data.size()[0] ): 

            c_indices.append(
                random.sample(
                    range(int(self.nb_category)),
                    nb_corrupted))

            # for nb_corrupted embedding requested
            for j in c_indices[-1]: 
                c_input[i][j*self.embedding_size:(j+1)*self.embedding_size]=0.0

        c_mask = ( c_input == 0 )

        return c_input, c_mask, c_indices