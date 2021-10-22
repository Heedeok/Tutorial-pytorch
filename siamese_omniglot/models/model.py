import torch
import torch.nn as nn

class My_siamese(nn.Module):
    def __init__(self):
        super(My_siamese, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),  # 128@42*42
            nn.MaxPool2d(2),  # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(inplace=True))  # 256@6*6))
        self.fc1 = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        
        # We use BCEWithLogitsLoss, so remove sigmoid layer
        self.head = nn.Linear(4096, 1)

    def forward_one(self,x):
        x = self.cnn1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x

    def forward(self, x1, x2):
        x1 = self.forward_one(x1)
        x2 = self.forward_one(x2)
        out = torch.abs(x1-x2)
        out = self.head(out)

        return out
