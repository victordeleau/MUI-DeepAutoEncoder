# MUIDAE - Mixed User Item Deep AutoEncoder

MUIDAE is a collaborative filtering recommendation system based on a Deep AutoEncoder Neural Network architecture.

To give it a shot, for example:

````
    cd muidae/
    conda create -f env.yml
    conda activate muidae
    clear && python3 main.py --batch_size 32 --learning_rate 0.00001 --zsize 128 --normalize
````