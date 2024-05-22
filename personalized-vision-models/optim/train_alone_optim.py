from .base_optim import Optim
import torch.optim as optim
import torch

class TrainAloneOptim(optim.Optimizer, Optim):

    def __init__(self, params, clients, device, learning_rate):
        super().__init__(params, {})
        self.clients = clients
        self.device = device
        self.params = params

    def step(self, learning_rate):

        for i in range(len(self.clients)):
            client = self.clients[i]
            model = client.model
            # momentum = model.get_momentum()
            # with torch.no_grad():
            model.update(model.get_momentum(), learning_rate=learning_rate)
            # model.previous_momentum = momentum

