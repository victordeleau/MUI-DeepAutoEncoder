
from codae.model import Autoencoder

class DenoisingAutoencoder(Autoencoder):

    def __init__(self, io_size, z_size, nb_item, corruption_type="zero", nb_input_layer=2, nb_output_layer=2, steep_layer_size=True, activation=nn.ReLU):

        super(Autoencoder, self).__init__()

        self.nb_item = nb_item

        if corruption_type == "zero" or corruption_type == "gaussian":
            raise Exception("Error: invalid corruption type provided (zero/gaussian)")

        self.corruption_type = corruption_type


    def corrupt(self, input_data):
        """
        corrupt the input data with the corruption noise specified during the model's initialization.

        input
            input_data : torch.Tensor
                input vector of size self.io_size
        """

        pass
