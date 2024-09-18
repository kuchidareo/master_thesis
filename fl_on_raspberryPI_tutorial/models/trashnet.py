import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Model from FedAvg paper. 2-layer simple CNN."""

    def __init__(self) -> None:
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(3 * 300 * 300, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class TrashNet(nn.Module):
    """Model from vasantvohra TrashNet: CNN 80% ipynb."""
    def __init__(self):
        super(TrashNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.max_pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 37 * 37, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.max_pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x