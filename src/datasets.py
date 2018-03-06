from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import glob
import os
import numpy as np
from src.data_utils import clip_and_scale_image


class TileTripletsDataset(Dataset):

    def __init__():
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


### TRANSFORMS ###

class GetBands(object):
    raise NotImplementedError

class RandomFlipAndRotate(object):
    raise NotImplementedError

class ClipAndScale(object):
    raise NotImplementedError

class ToFloatTensor(object):
    raise NotImplementedError

### TRANSFORMS ###


def triplet_dataloader(args):
    raise NotImplementedError

    