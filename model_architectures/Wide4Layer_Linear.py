import torch.nn as nn
import torch.nn.functional as F


# Pytorch Convolutional Neural Network Model Architecture
class Wide4Layer_Linear(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=(5, 5), stride=1, padding=1)
        self.fc1 = nn.Linear(in_features= 788544, out_features=500)
        self.fc2 = nn.Linear(in_features= 500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)
        


    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        #print(f'Shape of X: {X.shape}')
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X



