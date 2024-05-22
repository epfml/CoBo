from .base_optim import Optim
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import wandb


class DittoOptim(optim.Optimizer, Optim):

    def __init__(self, params, clients, device, grouping, learning_rate, k=2, w_lambda=1):
        super().__init__(params, {})
        self.grouping = grouping
        self.clients = clients
        self.device = device
        self.params = params
        self.iteration = 0
        self.global_model = None
        self.k = k
        self.w_lambda = w_lambda
        print('k is', self.k)

    def step(self, learning_rate):
        """in the first iteration we set global model to be average of all the models. then, in each iteration we sample
        k agents, and update global model based on their average gradient which is already stored in param.grad."""

        sz = len(self.clients)
        if self.global_model is None:
            for client in self.clients:
                if self.global_model is None:
                    self.global_model = []
                    for p in client.model.parameters():
                        self.global_model.append(p.data.clone())
                else:
                    for i, p in enumerate(client.model.parameters()):
                        self.global_model[i] += p.data.clone().to(self.global_model[i].device)

            for i in range(len(self.global_model)):
                self.global_model[i] /= sz

        selected_clients = torch.randperm(sz)[:self.k]
        print('it gets here', selected_clients)

        for idx in selected_clients:
            client = self.clients[idx]
            client.model.zero_grad()
            client.model.train()

            # for microstep_idx in range(acc_steps):
            local_device = next(client.model.parameters()).device
            next_batch = next(client.get_next_batch_train())
            inputs, targets = next_batch[0].to(local_device), next_batch[1].to(local_device)
            output = client.model(inputs,
                                  targets=targets)  # TODO: make the output of vision models the same as this!
            # loss = output["loss"]/acc_steps
            loss = output["loss"]
            loss.backward()

        grads = []

        for client in selected_clients:
            for i, p in enumerate(self.clients[client].model.parameters()):
                try:
                    p.data -= learning_rate * (p.grad + self.w_lambda * (p.data - self.global_model[i].to(p.data.device)))
                except RuntimeError as e:
                    breakpoint()
                if len(grads) <= i:
                    grads.append(p.grad.clone())
                else:
                    grads[i] += p.grad.clone().to(grads[i].device)

        for i in range(len(self.global_model)):
            self.global_model[i] -= learning_rate * grads[i].to(self.global_model[i].device) / self.k


