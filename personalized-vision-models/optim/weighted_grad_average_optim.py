from .base_optim import Optim
import torch.optim as optim


class WeightedGradAverageOptim(optim.Optimizer, Optim):
    def __init__(self, params, clients, device, groups, shared_layers, learning_rate):
        super().__init__(params, {'lr': learning_rate})
        self.shared_layers = shared_layers
        self.groups = groups
        self.clients = clients
        self.device = device
        self.made_models_similar = False

    def step(self, learning_rate):
        """
        We average the gradient of shared layers for all the models. For other layers we only average within their groups.
        The aggregated gradient of private layers is weighted by the number of clients in each group
        """
        if not self.made_models_similar:
            self.made_models_similar = True
            models = []
            for client in self.clients:
                models.append(client.model)

            self.average_layer_parameters(models)

        # models = []
        # for client in self.clients:
        #     models.append(client.model)

        # average_momentum_all = self.find_average_gradients(models)

        # for model in models:
        #     for i, param in enumerate(model.parameters()):
        #         if self.shared_layers[i] == 1:
        #             param.data -= learning_rate * average_momentum_all[i]

        # The rest should be uncommented!
        for group in self.groups:
            # group_rate = len(group.clients)/len(models)
            group_rate = 1
            models = list()
            for client in group.clients:
                models.append(client.model)

            print('group size:', len(group.clients))

            average_momentum_group = self.find_average_momentum(models)

            for j, client in enumerate(group.clients):
                for i, param in enumerate(client.model.parameters()):
                    if self.shared_layers[i] == 1:
                        param.data -= learning_rate * group_rate * average_momentum_group[i].to(param.data.device)