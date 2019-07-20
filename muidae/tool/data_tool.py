import numpy as np

def get_batch(dataset, batch_size, nb_sample_to_process, i):

    if batch_size == 1:

        return dataset[i], 1

    else:

        remaining = (
            batch_size if (i+1)*batch_size < nb_sample_to_process
            else nb_sample_to_process-(i*batch_size))

        batch = np.stack([dataset[i*batch_size+j] for j in range(remaining)])

        return batch, remaining