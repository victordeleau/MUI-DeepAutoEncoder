# CODAE - Co-occurence Denoising Autoencoder

````
    git clone https://github.com/victordeleau/codae
    cd codae/
    conda create -f env.yml
    conda activate codae
    pip install -e ./
````


# Steps

4 steps are involved. They can be started independently, or all at once using

```python3 script/4_complementarity_inference```

## I - Segment dataset

```python3 script/1_segment_dataset.py```

## II - Encode dataset

```python3 script/2_encode_dataset.py```

## III - Train autoencoder

```python3 script/3_train_autoencoder.py```

## IV - Complementarity inference

```python3 script/4_complementarity_inference.py```