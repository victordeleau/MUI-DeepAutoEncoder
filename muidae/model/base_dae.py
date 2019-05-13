# autoencoder base class

from torch import nn
import torch
import numpy as np


class BaseDAE(nn.Module):

    def __init__(self, io_size, z_size, nb_input_layer=1, nb_output_layer=1, activation=nn.ReLU ):

        super(BaseDAE, self).__init__()

        self.io_size = io_size
        self.z_size = z_size
        self.activation = activation

        self.mode = 0 # default to train mode (1 for validation mode, 2 for test mode)

        # variable length input layer #############################################################

        input_layer = []
        for i in range(nb_input_layer-1):
            input_layer.append( nn.Linear(io_size, io_size) )
            input_layer.append( activation(True) )

        input_layer.append( nn.Linear(io_size, z_size) )
        input_layer.append( activation(True) )

        self.input_layer = nn.Sequential( *input_layer )


        # variable length output layer #############################################################

        output_layer = []
        output_layer.append( nn.Linear(z_size, io_size) )
        output_layer.append( activation(True) )

        for i in range(nb_output_layer-1):
            output_layer.append( nn.Linear(io_size, io_size) )
            output_layer.append( activation(True) )
            
        self.output_layer = nn.Sequential( *output_layer )


        # initialize weights and biases ###########################################################

        # recursive application of the provided function to any nn.Linear layer

        self.input_layer.apply(self.init_weight_general_rule)
        self.input_layer.apply(self.init_bias_zero)
        
        self.output_layer.apply(self.init_weight_general_rule)
        self.output_layer.apply(self.init_bias_zero)

        

    """
        apply forward pass to input vector x
        input
            input vector of size self.io_size
        output
            output vector of size self.io_size
    """
    def forward(self, x):

        z = self.encode(x)

        y = self.decode(z)

        return y


    """
        take input vector x and encode to z
        input
            x: input vector to encode
        output
            z: encoded output vector
    """
    def encode(self, x):

        z = self.input_layer(x)

        return z
        

    """
        take compressed vector z and decode to y
        input
            z: compressed vector to decode
        output
            y: decoded output vector
    """
    def decode(self, z):
        
        y = self.output_layer(z)

        return y


    """
        initialize the auto encoder's weight using general rule
        input
            m: nn.Linear layer
    """
    def init_weight_general_rule(self, m):

        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            
            n = m.in_features # get nb of neural input
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
        
        
    """
        initialize the auto encoder's biases with zeros
        input
            m: nn.Linear layer
    """
    def init_bias_zero(self, m):

        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            
            m.bias.data.fill_(0)


    """def activation(self, input, kind):
        if kind == 'selu':
            return nn.functional.selu(input)
        elif kind == 'relu':
            return nn.functional.relu(input)
        elif kind == 'relu6':
            return nn.functional.relu6(input)
        elif kind == 'sigmoid':
            return nn.functional.sigmoid(input)
        elif kind == 'tanh':
            return nn.functional.tanh(input)
        elif kind == 'elu':
            return nn.functional.elu(input)
        elif kind == 'lrelu':
            return nn.functional.leaky_relu(input)
        elif kind == 'swish':
            return input * nn.functional.sigmoid(input)
        elif kind == 'none':
            return input
        else:
            raise ValueError('Unknown activation function')"""



