from .base_optim import Optim
import torch.optim as optim
from models.resnet import ResNet9
import random
from line_profiler import profile
import torch
import matplotlib.pyplot as plt
import wandb
from copy import deepcopy
import numpy as np

class IFCAOptim(optim.Optimizer, Optim):

    def __init__(self, params, clients, device, learning_rate, k, m, config):
        #k is the number of cluster centers
        #m is the number of clients that we sample each time
        super().__init__(params, {'lr': learning_rate})
        print('salam!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.clients = clients
        self.device = device
        self.k = k
        self.m = m
        self.centers = []
        gpus = config.gpus
        for _ in range(self.k):
            device = torch.device("cuda:" + str(_ % gpus) if torch.cuda.is_available() else "cpu")
            self.centers.append(ResNet9().to(device))
        # self.centers.append(clients[0].model)
        self.time_step = 0

    @profile
    def step(self, learning_rate):
        """We sample m clients from len(clients) each time. Then we evaluate each of the centers based on this client
        dataset and pick the lowest one. For each client center, we update the center based on the gradients of its clients"""

        # data, target = client.get_next_batch_train()
        # device = next(center.parameters()).device
        # center.train()
        # center.zero_grad()
        # output = center(data.to(device), targets=target.to(device))
        # loss = output["loss"]
        # loss.backward()
        # gradients = center.get_gradients()
        # center.update(gradients, learning_rate)

        # self.clients[0].model = deepcopy(center)

        # for param_client, param_center in zip(client.model.parameters(), client.parameters()):
        #     param_client.data = param_center.data.clone()

        indices = random.sample(range(len(self.clients)), self.m)
        chosen_centers = []
        # print('indices:', indices)

        for index in indices:
            client = self.clients[index]
            losses = []
            for center_model in self.centers:

                data, target = next(client.get_next_batch_train())
                device = next(center_model.parameters()).device
                center_model.train()
                center_model.zero_grad()
                output = center_model(data.to(device), targets=target.to(device))
                losses.append(output["loss"].item())

            print('agent', index, "chose center", losses.index(min(losses)))
            chosen_centers.append(losses.index(min(losses)))

        # print('done choosing!')
        for i in range(self.k):
            self.centers[i].zero_grad()
            self.centers[i].train()

        #
        # # print('done setting gradients to zero')
        #
        # # breakpoint()
        #
        for i in range(self.m):
            client = self.clients[indices[i]]
            center = self.centers[chosen_centers[i]]

            data, target = next(client.get_next_batch_train())
            device = next(center.parameters()).device
            # device = next(client.model.parameters()).device

            output = center(data.to(device), targets=target.to(device))
            loss = output["loss"]
            loss.backward()

            # gradients = center.get_gradients()
            # center.update(gradients, learning_rate)

            # breakpoint()
            # print('backprop for agent', indices[i], 'done!')
        #
        for i in range(self.k):
            gradients = self.centers[i].get_gradients()
            self.centers[i].update(gradients, learning_rate)
            self.centers[i].zero_grad()
        #
        # # print('centers are updated')
        #
        # # for i in range(self.m):
        # #     client = self.clients[indices[i]]
        # #     center = self.centers[chosen_centers[i]]
        # #     # cloned_params = [param.clone().detach() for param in center.parameters()]
        # #     client.model.set_params(list(center.parameters()))
        # # for param_client, param_center in zip(self.clients[0].model.parameters(), self.centers[0].parameters()):
        # #     param_client.data = param_center.data.clone().detach()
        #
        #     # print('updated the model of agent', indices[i])
        # self.clients[0].model.load_state_dict(self.centers[0].state_dict())

        self.time_step += 1

        for i in range(self.m):
            device = next(self.clients[indices[i]].model.parameters()).device
            self.clients[indices[i]].model = deepcopy(self.centers[chosen_centers[i]]).to(device)

        if self.time_step % 100 == 0 and self.time_step > 0:
            adj = torch.zeros((len(self.clients), len(self.clients)))
            for i, id1 in enumerate(indices):
                for j, id2 in enumerate(indices):
                    adj[id1, id2] = (chosen_centers[j] == chosen_centers[i])
                    adj[id2, id1] = adj[id1, id2]

            adj = adj.cpu().numpy()
            np.fill_diagonal(adj, np.nan)

            plt.imshow(adj, cmap='viridis', interpolation='none', vmin=0, vmax=1)
            wandb.log({"adjacency matrix": wandb.Image(plt)}, commit=False)