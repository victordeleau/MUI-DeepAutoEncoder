# autoencoder base class

from torch import nn
import torch
import numpy as np


class Autoencoder(nn.Module):

    def __init__(self, io_size, z_size, nb_input_layer=2, nb_output_layer=2, steep_layer_size=True, activation=nn.ReLU):
        """
        input
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

        super(BaseDAE, self).__init__()

        self.io_size = io_size
        self.z_size = z_size
        self.nb_input_layer = nb_input_layer
        self.nb_output_layer = nb_output_layer
        self.steep_layer_size = steep_layer_size
        self.activation = activation

        self.mode = 0 # default to train mode (1 for validation mode, 2 for test mode)

        # set the steepiness of the layers if necessary

        if not self.steep_layer_size:
            delta = self.io_size - self.z_size
            input_layer_increment = math.floor( delta / self.nb_input_layer )
            output_layer_increment = math.floor( delta / self.nb_output_layer )

        # set the encoder ##############################################################

        input_layer = []
        for i in range(nb_input_layer-1):

            if self.steep_layer_size:
                input_layer.append( nn.Linear(io_size, io_size) )
                input_layer.append( activation(True) )
            else:
                next_layer_input_size = io_size-(i*input_layer_increment)
                next_layer_output_size = io_size-((i+1)*input_layer_increment))
                input_layer.append( nn.Linear(next_layer_input_size, next_layer_output_size )
                input_layer.append( activation(True) )

        input_layer.append( nn.Linear(z_size+input_layer_increment, z_size) )
        input_layer.append( activation(True) )

        # join encoder layers
        self.input_layer = nn.Sequential( *input_layer )

        # initialize encoder layer weights and biases
        self.input_layer.apply(self.init_weight_general_rule)
        self.input_layer.apply(self.init_bias_zero)


        # set the decoder ##############################################################

        output_layer = []
        output_layer.append( nn.Linear(z_size, io_size) )
        output_layer.append( activation(True) )

        for i in range(nb_output_layer-1):

            if self.steep_layer_size:
                output_layer.append( nn.Linear(io_size, io_size) )
                output_layer.append( activation(True) )
            else:
                next_layer_input_size = io_size-((nb_output_layer-i)*output_layer_increment)
                next_layer_output_size = io_size-((nb_output_layer-i+1)*output_layer_increment)
                output_layer.append( nn.Linear(next_layer_input_size, next_layer_output_size) )
                output_layer.append( activation(True) )
            
        # join decoder layers
        self.output_layer = nn.Sequential( *output_layer )

        # initialize decoder layer weights and biases
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

        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            
            n = m.in_features # get nb of neural input
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
        
        
    def init_bias_zero(self, m):
        """
        initialize the auto encoder's biases with zeros
        input
            m : nn.Linear module to initialize
        """

        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            
            m.bias.data.fill_(0)


    def get_mmse_loss(self, input, output):
        """
        compute mask mean square error (mmse) and return loss
        use a mask if data is not normalized
        input
            input : torch.Tensor
            output : torch.Tensor
        """

        mmse_criterion = nn.MSELoss(reduction='sum')

        mask = input_data != 0.0
        nb_rating = torch.sum( mask )
        loss = mmse_criterion( input_data, output_data * mask.float() ) / nb_rating.float()

        return loss