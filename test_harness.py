import torch
import os
import torch.nn.functional as F


from torchvision import transforms
from torch.utils.data import DataLoader


from TestCatDogDataset import TestCatDogDataset
from CatAndDogConvNet import CatAndDogConvNet

from PIL import Image
import matplotlib.pyplot as plt


# image normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# This file will load the specified model and run it through test data.
# The purpose of this file is to support model evaluation

# Load the model
model = CatAndDogConvNet()
model.load_state_dict(torch.load('models/base_model.pth'))

# Load the test data
test_files = os.listdir('data/test/')
test_files = list(filter(lambda x: x != 'test', test_files))
def test_path(p): return f"data/test/{p}"
test_files = list(map(test_path, test_files))

# Load test data into dataloader
test_ds = TestCatDogDataset(test_files, transform)
test_dl = DataLoader(test_ds, batch_size=100)
print(len(test_ds), len(test_dl))

dog_probs = []

# TODO: Test data is unlabeled. How do I evaluate on them?

with torch.no_grad():
    for X, fileid in test_dl:
        preds = model(X)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), preds_list))

# Need to find a better way to evaluate rather than "display some images"
# display some images
for img, probs in zip(test_files[:5], dog_probs[:5]):
    pil_im = Image.open(img, 'r')
    label = "dog" if probs[1] > 0.5 else "cat"
    title = "prob of dog: " + str(probs[1]) + " Classified as: " + label
    plt.figure()
    plt.imshow(pil_im)
    plt.suptitle(title)
    plt.show()
