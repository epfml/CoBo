from .base_optim import Optim
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import wandb
class FederatedClusteringOptim(optim.Optimizer, Optim):

    def __init__(self, params, clients, device, grouping, learning_rate, percentile=0.2):
        super().__init__(params, {})
        self.grouping = grouping
        self.clients = clients
        self.device = device
        self.params = params
        self.centers = []
        self.percentile = percentile
        self.iteration = 0
        self.centers = None

    def _find_tau_percentile(self, dists):
        return torch.quantile(torch.tensor(dists), self.percentile)

    def _find_distance(self, model, centers):
        dist = 0
        for i, param in enumerate(model.parameters()):
            dist += torch.sum((param.grad - centers[i]) ** 2)
        return torch.sqrt(dist)

    def step(self, learning_rate):
        """For each of the clients, we send a batch of its training data to every other client and do a backward pass.
        then we collect their gradients. we find their distances to current client center and update the center
        accordingly."""
        print('iteration number', self.iteration)
        adjacency_matrix = torch.zeros((len(self.clients), len(self.clients)), device=self.device)

        self.centers = []

        for i in range(len(self.clients)):
            client = self.clients[i]
            data, target = next(client.get_next_batch_train())
            #
            # # print('len new_center is', len(new_center))
            for j in range(len(self.clients)):

                client_j = self.clients[j]
                model_j = client_j.model
                model_j.train()
                model_j.zero_grad()
                output_j = model_j(data.to(self.device), targets=target.to(self.device))
                loss_j = output_j['loss']
                loss_j.backward()

            center = []
            for param in client.model.parameters():
                center.append(param.grad.clone())
            self.centers.append(center)

            for l in range(10):
                dists = []
                new_center = []
                for j in range(len(self.centers[i])):
                    new_center.append(torch.zeros_like(self.centers[i][j]))

                for j in range(len(self.clients)):
                    dists.append(self._find_distance(self.clients[j].model, self.centers[i]))

                tau = self._find_tau_percentile(dists)
                #
                for j in range(len(self.clients)):
                    client_j = self.clients[j]
                    model_j = client_j.model
                    for k, param in enumerate(model_j.parameters()):
                        new_center[k] += (((dists[j] <= tau) * param.grad.clone() + (dists[j] > tau) * self.centers[i][k])
                                          / len(self.clients))
            #
                    adjacency_matrix[i, j] = (dists[j] <= tau)
                #
                self.centers[i] = new_center

            for j, param in enumerate(client.model.parameters()):
                # if j == 0:
                #     print('here grad is', param.grad[0])
                # breakpoint()
                param.data -= learning_rate * self.centers[i][j]

        self.iteration += 1

        if self.iteration % 100 == 0:
            print('adjacency matrix', adjacency_matrix)
            # print('sample tau', tau)

            plt.imshow(adjacency_matrix.cpu().numpy(), cmap='viridis', interpolation='none', vmin=0, vmax=1)
            wandb.log({'adjacency matrix': wandb.Image(plt)}, step=self.iteration // 100)




