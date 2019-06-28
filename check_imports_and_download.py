#!/usr/bin/env python
import os

import matplotlib as mpl
import numpy as np
import seaborn as sns
import sklearn
import torch
import torchvision
import tqdm

import ds3_support

torchvision.datasets.MNIST(root='data', download=True)
ds3_support.CombinedOmniglot(root='data', download=True)

assert os.path.exists('data/transcripts.tar.bz2')

print("Okay, you're good to go!")
