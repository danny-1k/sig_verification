import torch
import torch.nn as nn 

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=2,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=2,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(14400,512)
        )

        self.fc = nn.Sequential(
            nn.Linear(1024,200),
            nn.ReLU(),
            nn.Dropout(.4),
            nn.Linear(200,50),
            nn.ReLU(),
            nn.Linear(50,2),
        )

    
    def forward(self,x1,x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)

        x = torch.hstack([x1,x2])

        x = self.fc(x)

        return x

    def save_(self,f='model.pt'):
        torch.save(self.state_dict(),f)

    def load_(self,f='model.pt'):
        self.load_state_dict(torch.load(f))
        self.eval()