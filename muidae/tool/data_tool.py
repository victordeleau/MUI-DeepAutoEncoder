import numpy as np
import random

"""
    allow to create multiple batch from a single batch, with the following constraints
    - output must be deterministic on the input once the random masks are created
    - cannot return a batch where one or more element (vector) are empty
    The trick is to create multiple potential masks for each element of the batch to return,
    and to try each one until conditions are met, i.e. there is no empty vector in the batch
"""
class BatchBuilder:

    def __init__(self, vector_size, split=[0.8, 0.1, 0.1], nb_potential_mask=16):

        if sum(split) != 1:
            raise Exception("Split percentage should sum up to 1.")

        self.vector_size = vector_size

        self.nb_potential_mask = nb_potential_mask

        self.split = split

        self.iterative_split = self._full_split_to_iterative_split( split )

        self.masks = []

        self.full_vector = np.ones(vector_size)

        for i in range(len(split)-1):

            self.masks.append([])

            for j in range(self.nb_potential_mask):

                random_mask = np.random.choice([0, 1], size=self.vector_size, p=self.iterative_split[i])

                self.masks[i].append( np.random.choice([0, 1], size=self.vector_size, p=self.iterative_split[i]) )


    """
        get a list of split factors to apply iteratively from a full list of slip factors
        ex: [0.8, 0.1, 0.1] => [ [0.8, 0.2], [0.5, 0.5] ]
    """
    def _full_split_to_iterative_split(self, split):

        iterative_split = []
        iterative_split.append( [ split[0], sum(split[1:]) ] )

        for i in range(1, len(split)-1):
            iterative_split.append( [ split[i]/sum(split[i:]), 1-(split[i]/sum(split[i:])) ] )

        return iterative_split


    """
        return N split batches from dataset
    """
    def get_batches(self, dataset, batch_size, nb_sample_to_process, i):   

        output_batches = []

        if batch_size == 1:
            remaining = 1
            main_batch = [ dataset[i] ]

        else:
            remaining = (
                batch_size if (i+1)*batch_size < nb_sample_to_process
                else nb_sample_to_process-(i*batch_size))
            
            main_batch = np.stack([dataset[i*batch_size+j] for j in range(remaining)])

        output_batches = [ [] for i in range(len(self.split)) ]

        # for each element in batch
        for element in main_batch:

            # for every iterative split
            for i in range(len(self.split)-1):

                # find a suitable mask
                good_mask_found = False
                mask_index = 0
                while not good_mask_found:

                    # apply the mask
                    recto = np.multiply( element, self.full_vector - self.masks[i][mask_index] )
                    verso = np.multiply( element, self.masks[i][mask_index] )

                    # check the mask validity
                    if recto.sum() != 0 and verso.sum() != 0:

                        good_mask_found = True
                        output_batches[i].append( recto )
                        element = verso

                    else:
                        mask_index += 1
                        if mask_index == self.nb_potential_mask:
                            raise Exception("Was not able to find a suitable mask ... increase parameter.")

            output_batches[-1].append( verso )

        for i in range(len(self.split)):
            output_batches[i] = np.stack(output_batches[i])

        return (*output_batches, remaining)



if __name__ == "__main__":

    batch_builder = BatchBuilder(100, [0.8, 0.1, 0.1], 16)

    dataset = np.ones(10000).reshape((100, 100))

    batch_1, batch_2, batch_3, remaining = batch_builder.get_batches( dataset, batch_size=2, nb_sample_to_process=1000, i=0 )

    print(batch_1)
    print( batch_1[0].sum() / batch_1[0].shape[0] )

    print(batch_2)
    print( batch_2[0].sum() / batch_2[0].shape[0] )

    print(batch_3)
    print( batch_3[0].sum() / batch_3[0].shape[0] )