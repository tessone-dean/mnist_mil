import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),                     # 14×14
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),                     # 7×7
        )
        self.fc = nn.Linear(64*7*7, out_dim)

    def forward(self, x):           # x: (N,1,28,28)
        x = self.features(x)
        return self.fc(x.flatten(1))            # (N,out_dim)
