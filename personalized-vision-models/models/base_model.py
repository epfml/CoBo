from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(ABC, nn.Module):
    def __init__(self, args=None):
        super(BaseModel, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss
        self.previous_momentum = None
        self.args = args

    def zero_grad(self):
        """set gradients of the model to zero"""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        return

    def get_gradients(self):
        """return the gradients of the model"""
        gradients = []
        for i, p in enumerate(self.parameters()):
            gradients.append(p.grad)
        return gradients

    def get_momentum(self, momentum_param=0.9):
        """return the momentum of the model"""
        gradients = self.get_gradients()
        # breakpoint()
        if self.previous_momentum is None:
            self.previous_momentum = []
            for i in range(len(gradients)):
                self.previous_momentum.append(gradients[i])
        else:
            for i, p in enumerate(self.parameters()):
                self.previous_momentum[i] = momentum_param*self.previous_momentum[i] + (1 - momentum_param)*gradients[i]
                # self.previous_momentum[i].mul_(momentum_param)
                # self.previous_momentum[i].add_(gradients[i], alpha=(1 - momentum_param))
                # momentum.append(momentum_param*self.previous_momentum[i] + (1 - momentum_param)*gradients[i])

        return self.previous_momentum

    def update(self, gradients, learning_rate=0.1):
        """update the model parameters with the gradients and the given learning rate"""
        for i, p in enumerate(self.parameters()):
            if gradients[i] is not None:
                p.data.add_(gradients[i].to(p.device), alpha=-learning_rate)

        return

    def set_params(self, params):
        """set the model parameters to the given params"""
        for i, p in enumerate(self.parameters()):
            p.data = params[i].data.clone().to(p.device)

        return

    def conv_block(self, in_channels, out_channels, pool=False):
        # print('in channels and out channels', in_channels, out_channels)
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)