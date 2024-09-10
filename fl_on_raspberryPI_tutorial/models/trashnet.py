import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Model from FedAvg paper. 2-layer simple CNN."""

    def __init__(self) -> None:
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(224 * 224, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) # 224*224
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x