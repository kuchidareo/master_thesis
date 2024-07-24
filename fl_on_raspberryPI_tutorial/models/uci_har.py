import torch.nn as nn
import torch.nn.functional as F


class HARNet(nn.Module):
    def __init__(self):
        super(HARNet, self).__init__()
        self.fc1 = nn.Linear(561, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
