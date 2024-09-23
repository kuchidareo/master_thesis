import torch
import torch.nn as nn

class MnistNet(nn.Module):
    """Model from FedAvg paper. 2-layer simple CNN."""

    def __init__(self) -> None:
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x