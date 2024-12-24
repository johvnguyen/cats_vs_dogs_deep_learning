# install conda environment with pytorch support
# - conda create -n torch python=3.7
# - conda activate torch
# - conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

#import numpy as np
#import pandas as pd
import os
import random
import time

import torch
#import torchvision
import torch.nn as nn
#import torchvision.datasets as datasets
#from torchvision import datasets, transforms
from torchvision import transforms
#from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import torch.nn.functional as F

#from sklearn.model_selection import train_test_split
from CatDogDataset import CatDogDataset
from TestCatDogDataset import TestCatDogDataset
from model_architectures.CatAndDogConvNet import CatAndDogConvNet
#from model_architectures.Wide3Layer import Wide3Layer
from Utils import save_training_stats

from PIL import Image
import matplotlib.pyplot as plt
import pprint

def calculate_accuracy(preds):
    correct = 0
    
    for fileid in preds.keys():
        fname = fileid.split('.')[0]
        correct += fname == preds[fileid]
        
    correct /= len(preds.keys())
    
    return correct

# image normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_files = os.listdir('data/split1/test/')
test_files = list(filter(lambda x: x != 'test', test_files))
def test_path(p): return f"data/split1/test/{p}"
test_files = list(map(test_path, test_files))

test_ds = TestCatDogDataset(test_files, transform)
test_dl = DataLoader(test_ds, batch_size=100)
print(len(test_ds), len(test_dl))

model = CatAndDogConvNet()
model.load_state_dict(torch.load('models/DefaultTest.pth'))
model.eval()

dog_probs = []

with torch.no_grad():
    for X, fileid in test_dl:
        preds = model(X)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), preds_list))
        
preds = {}

for (fileid, prob) in dog_probs:
    #fname = fileid.split('/')[-1].split('.')[0]
    fname = fileid.split('/')[-1]
    if prob >= 0.5:
        prediction = 'dog'
    else:
        prediction = 'cat'
        
    preds[fname] = prediction
        
accuracy = calculate_accuracy(preds)

print(f'Achieved accuracy of {accuracy} out of {len(preds.keys())} samples')