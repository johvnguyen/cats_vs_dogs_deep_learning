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
#from model_architectures.DefaultTest import DefaultTest
from model_architectures.Wide3Layer import Wide3Layer
#from model_architectures.SkinnyDeep import SkinnyDeep

from Utils import save_training_stats

from PIL import Image
import matplotlib.pyplot as plt

experiment_name = 'Wide3Layer'

# make sure all data is local and found
img_files = os.listdir('data/split1/train/')
img_files = list(filter(lambda x: x != 'train', img_files))
def train_path(p): return f"data/split1/train/{p}"
img_files = list(map(train_path, img_files))

validation_files = os.listdir('data/split1/validation/')
validation_files = list(filter(lambda x: x != 'validation', validation_files))
def validation_path(p): return f'data/split1/validation/{p}'
validation_files = list(map(validation_path, validation_files))

print("total training images", len(img_files))
print("First item", img_files[0])

# create train-test split for training
random.shuffle(img_files)

# Refactor "test" to use validation data
train = img_files
test = validation_files

print("train size", len(train))
print("test size", len(test))

# image normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# preprocessing of images


# create train dataset
train_ds = CatDogDataset(train, transform)
train_dl = DataLoader(train_ds, batch_size=100)
print(len(train_ds), len(train_dl))

# create test dataset
test_ds = CatDogDataset(test, transform)
test_dl = DataLoader(test_ds, batch_size=100)
print(len(test_ds), len(test_dl))



# Create instance of the model
model = Wide3Layer()

losses = []
accuracies = []

test_losses = []
test_accuracies = []

epoches = 7
start = time.time()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Model Training...
for epoch in range(epoches):

    epoch_loss = 0
    epoch_accuracy = 0

    for X, y in train_dl:

        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = ((preds.argmax(dim=1) == y).float().mean())
        epoch_accuracy += accuracy
        epoch_loss += loss
        print('.', end='', flush=True)

    epoch_accuracy = epoch_accuracy/len(train_dl)
    accuracies.append(epoch_accuracy.item())
    epoch_loss = epoch_loss / len(train_dl)
    losses.append(epoch_loss.item())

    print("\n --- Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))

    # test set accuracy
    with torch.no_grad():   # torch.no_grad() disables gradient descent tracking so test samples will not be counted when Tensor.backward() is called next

        test_epoch_loss = 0
        test_epoch_accuracy = 0

        for test_X, test_y in test_dl:

            test_preds = model(test_X)
            test_loss = loss_fn(test_preds, test_y)

            test_epoch_loss += test_loss            
            test_accuracy = ((test_preds.argmax(dim=1) == test_y).float().mean())
            test_epoch_accuracy += test_accuracy

        test_epoch_accuracy = test_epoch_accuracy/len(test_dl)
        test_epoch_loss = test_epoch_loss / len(test_dl)

        test_losses.append(test_epoch_loss.item())
        test_accuracies.append(test_epoch_accuracy.item())

        print("Epoch: {}, test loss: {:.4f}, test acc: {:.4f}, time: {}\n".format(epoch, test_epoch_loss, test_epoch_accuracy, time.time() - start))

torch.save(model.state_dict(), f'models/{experiment_name}.pth')
save_training_stats(experiment_name, losses, accuracies, test_losses, test_accuracies)

test_files = os.listdir('data/split1/test/')
test_files = list(filter(lambda x: x != 'test', test_files))
def test_path(p): return f"data/split1/test/{p}"
test_files = list(map(test_path, test_files))



test_ds = TestCatDogDataset(test_files, transform)
test_dl = DataLoader(test_ds, batch_size=100)
print(len(test_ds), len(test_dl))

dog_probs = []

with torch.no_grad():
    for X, fileid in test_dl:
        preds = model(X)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), preds_list))

# display some images
for img, probs in zip(test_files[:5], dog_probs[:5]):
    pil_im = Image.open(img, 'r')
    label = "dog" if probs[1] > 0.5 else "cat"
    title = "prob of dog: " + str(probs[1]) + " Classified as: " + label
    plt.figure()
    plt.imshow(pil_im)
    plt.suptitle(title)
    plt.show()
