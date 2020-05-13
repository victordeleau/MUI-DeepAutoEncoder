# A mixed variable denoising autoencoder model

import math

import torch
import numpy as np

class MixedVariableDenoisingAutoencoder(torch.nn.Module):
    
    def __init__(self, input_arch, z_size, nb_input_layer=2, nb_output_layer=2, steep_layer_size=True, activation=torch.nn.ReLU):
        """input
            input_arch : dict
                list of dict {"name": name, "type": "regression"/"classification", "size": int, "lambda": 1, "position": int}
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

        super(MixedVariableDenoisingAutoencoder, self).__init__()

        self.io_size = d["position"] + d["size"]

        self.input_arch = input_arch
        self.z_size = z_size
        
        self.nb_input_layer = nb_input_layer
        self.nb_output_layer = nb_output_layer
        self.steep_layer_size = steep_layer_size
        self.activation = activation

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

        # join encoder layers
        self.input_layer = torch.nn.Sequential( *input_layer ) 

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


    def get_loss(self, input_data, output_data):
        """
        compute mask mean square error (mmse) and return loss
        use a mask if data is not normalized
        input
            input : torch.Tensor
            output : torch.Tensor
        """

        mse_criterion = torch.nn.MSELoss(reduction='sum')
        ce_criterion = torch.nn.cross

        mask = torch.Tensor(output_data.size())

        # compute mask
        for variable in self.input_arch:
            if output_data[variable["position"]:variable["size"]].mean() == 0:
                mask[variable["position"]:variable["size"]] = True

        loss, nb = 0, 0
        for variable in self.input_arch:

            if output_data[variable["position"]:variable["size"]].mean() == 0:
                continue

            if variable["type"] == "regression":

                loss += mse_criterion(
                    input_data[variable["position"]: variable["position"]+variable["size"]],
                    output_data[variable["position"]: variable["position"]+variable["size"]])
                nb += variable["size"]

            elif variable["type"] == "classification":

                loss += ce_criterion(
                    input_data[variable["position"]: variable["position"]+variable["size"]],
                    output_data[variable["position"]: variable["position"]+1])
                nb += 1

            else:
                raise Exception("Error: invalid loss type requested.")

        return loss / nb


    def to(self, *args, **kwargs):
        """
        override .to() to make sure custom layers are sent to the correct device
        """

        self = super().to(*args, **kwargs) 
        self.input_layer = self.input_layer.to(*args, **kwargs)
        self.output_layer = self.output_layer.to(*args, **kwargs) 
        
        return self


    def corrupt(self, input_data, device, indices, corruption_type="zero_continuous"):
        """
        corrupt input_data using requested corruption type
        input
            input_data : torch.Tensor
                the piece of data to corrupt
            indices : list(list(int))
                list of embedding index to corrupt
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
                indices=indices)
        else:
            raise Exception("Error: invalid corruption type requested (zero_continuous).")


    def _corrupt_zero_continuous(self, input_data, device, indices):
        """
        randomly set one of the input embeddings and set all values to zero.
        input
            input_data : torch.Tensor
                the piece of data to corrupt
            indices : list(list(int))
                list of embedding index to corrupt
        output
            c_input : torch.Tensor
                the corrupted input data
            c_indices : list(int)
                indices of corrupted categories
        """

        c_input = input_data
        c_mask = torch.empty(
            input_data.size(),
            device=device)

        for i in range( input_data.size()[0] ): # for batch size 
            
            start = indices[i]*self.input_layer[indices[i]]["position"]
            end = (indices[i]*self.input_layer[indices[i]]["position"]) + self.input_layer[indices[i]]["size"]

            c_input[i][start:end]=0.0
            c_mask[i][start:end]=True

        return c_input, c_mask