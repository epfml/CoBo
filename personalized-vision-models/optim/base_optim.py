import torch
from abc import ABC, abstractmethod

class Optim(ABC):
    def __init__(self, train):
        self.train = train


    def zero_grad(self):
        """set gradients of the model to zero"""
        for client in self.train.clients:
            for p in client.model.parameters():
                if p.grad is not None:
                    p.grad = torch.zeros(p.grad.shape, device=self.train.device)
        return

    def calculate_grads(self):
        """calculate the gradients of the model for each client"""
        for client in self.train.clients:
            client.model.zero_grad()
            client.model.train()

            for microstep_idx in range(self.train.config.acc_steps):
                next_batch = next(client.get_next_batch_train())
                output = client.model(next_batch[0].to(self.train.device), targets=next_batch[1].to(
                    self.train.device))  # TODO: make the output of vision models the same as this!
                loss = output["loss"]/self.train.config.acc_steps
                # loss = output["loss"]
                loss.backward()

    @staticmethod
    def find_average_gradients(averaging_models):
        """finds the average of the model gradients for the models given in the input"""
        aggregated_grads = list()
        for i, model in enumerate(averaging_models):
            for j, grad in enumerate(model.get_gradients()):
                # if j == 0:
                #     print(grad[0,0,0,:3])
                if i == 0:
                    aggregated_grads.append(grad.clone())
                else:
                    aggregated_grads[j] += grad.clone().to(aggregated_grads[j].device)

        for i in range(len(aggregated_grads)):
            aggregated_grads[i] /= len(averaging_models)
            # if i == 0:
            #     print('averaged: ', aggregated_grads[i][0, 0, 0,:3])

        return aggregated_grads

    @staticmethod
    def find_average_momentum(averaging_models):
        """finds the average of the model momentum for the models given in the input"""
        aggregated_momentum = list()
        for i, model in enumerate(averaging_models):
            for j, momentum in enumerate(model.get_momentum()):
                # if j == 0:
                #     pisrint(grad[0,0,0,:3])
                if i == 0:
                    aggregated_momentum.append(momentum.clone())
                else:
                    aggregated_momentum[j] += momentum.clone()

        for i in range(len(aggregated_momentum)):
            aggregated_momentum[i] /= len(averaging_models)
            # if i == 0:
            #     print('averaged: ', aggregated_grads[i][0, 0, 0,:3])

        return aggregated_momentum

    @staticmethod
    def find_average_parameters(averaging_models):
        """finds the average of the model parameters for the models given in the input"""
        params = list()
        for i in range(len(averaging_models)):
            for j, param in enumerate(averaging_models[i].parameters()):
                if i == 0:
                    params.append(param.data.clone())
                else:
                    params[j] += param.data.clone().to(params[j].device)

        for i in range(len(params)):
            params[i] /= len(averaging_models)

        return params

    @staticmethod
    def average_layer_parameters(models):
        """
        instead of aggregating gradient, each model, updates itself based on local gradient, and then we aggregate
        model parameters (we don't use shared layers here)
        """
        # for model in models:
        #     model.update(model.get_gradients(), learning_rate)

        average_params_all = Optim.find_average_parameters(models)

        for model in models:
            for j, param in enumerate(model.parameters()):
                param.data = average_params_all[j].clone().to(param.data.device)
