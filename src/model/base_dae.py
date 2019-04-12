# autoencoder base class

from torch import nn
import torch


class BaseDAE(nn.Module):

    def __init__(self, io_size, z_size, nb_input_layer=1, nb_output_layer=1, activation=nn.ReLU ):

        super().__init__()

        self.io_size = io_size
        self.z_size = z_size
        self.activation = activation

        # variable length input layer #############################################################

        input_layer = []
        for i in range(nb_input_layer-1):
            input_layer.append( nn.Linear(io_size, io_size) )
            input_layer.append( activation(True) )

        input_layer.append( nn.Linear(io_size, z_size) )
        input_layer.append( activation(True) )

        self.encoder = nn.Sequential( *input_layer )


        # variable length output layer #############################################################

        output_layer = []
        output_layer.append( nn.Linear(z_size, io_size) )
        output_layer.append( activation(True) )

        for i in range(nb_output_layer-1):
            output_layer.append( nn.Linear(io_size, io_size) )
            output_layer.append( activation(True) )
            
        self.decoder = nn.Sequential( *output_layer )

    """
        apply forward pass to input
        input
            input vector of size self.io_size
        output
            output vector of size self.io_size
    """
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


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



