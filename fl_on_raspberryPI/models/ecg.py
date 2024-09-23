import torch.nn as nn
import torch.nn.functional as F


class EcgConv1d(nn.Module):
    def __init__(self):
        super(EcgConv1d, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 7)
        self.conv2 = nn.Conv1d(16, 16, 5)
        self.conv3 = nn.Conv1d(16, 16, 5)
        self.conv4 = nn.Conv1d(16, 16, 5)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(25 * 16, 128)
        self.fc2 = nn.Linear(128, 5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 25 * 16)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
