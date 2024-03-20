import torch.nn as nn


class CNN_2D(nn.Module):
    def __init__(self, in_chanel = 1):
        super(CNN_2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chanel, 20, 3, 1, 0),  
            nn.ReLU()
            )
        self.dropout1 = nn.Dropout(0.4)
        self.subsample1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 15, 3, 1, 0),  
            nn.ReLU()
            )
        self.dropout2 = nn.Dropout(0.3)
        self.subsample2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(15 * 4 * 4, 78)
        self.fc2 = nn.Linear(78, 2)

    def forward(self, x):
        x = self.subsample1(self.dropout1(self.conv1(x)))
        x = self.subsample2(self.dropout2(self.conv2(x)))
        x = self.fc1(x.view(x.size()[0], -1))
        out = self.fc2(x)
        return out
