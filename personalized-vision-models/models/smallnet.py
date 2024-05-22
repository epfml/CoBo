import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class SmallNet(BaseModel):
    def __init__(self, args=None, client=None, config=None):
        """initializing a simple NN with 2 convolutional layers and 2 fully connected layers"""
        super(SmallNet, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.client = client


    def forward(self, x, targets=None, get_logits=False):
        """rewriting the forward pass"""

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(x, targets)
        else:
            loss = None

        logits = x if get_logits else None

        return {'logits': logits, 'loss': loss}

