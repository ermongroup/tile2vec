# tile2vec
Implementation and examples for Tile2Vec

## Getting data

Follow the follow instructions for getting the necessary data for the provided examples.

### Getting pre-trained NAIP model

To get a Tile2Vec model that has been pre-trained on the NAIP dataset, download the weights provided [here](https://www.dropbox.com/s/bvzriiqlcof5lol/naip_trained.ckpt?dl=0) and save the file in the models directory: `tile2vec/models/naip_trained.ckpt`.

### Getting tiles
Unzip `data/tiles.zip` in the data directory, i.e., `tile2vec/data/tiles`. Within this folder, you should find 1000 tiles saved as Numpy arrays named `{1-1000}tile.npy` and 1 file named `y.npy` that contains the CDL labels corresponding to these 1000 tiles. 

### Getting triplets
Download tile triplet dataset for the training example [here](https://www.dropbox.com/s/afw3cbvo7sjerru/triplets.zip?dl=0) and then unzip in the data directory, i.e., `tile2vec/data/triplets`. This directory contains 100k tile triplets, with each triplet identified by its index and "anchor", "neighbor", or "distant".