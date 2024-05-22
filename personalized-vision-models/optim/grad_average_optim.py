from .base_optim import Optim
import torch.optim as optim


class GradAverageOptim(optim.Optimizer, Optim):
    def __init__(self, params, clients, device, learning_rate):
        super().__init__(params, {'lr': learning_rate})
        self.clients = clients
        self.device = device
        self.made_models_similar = False


    def step(self, learning_rate):
        """
        the simplest federated learning aggregation. Just average the gradients of all models (for all layers)
        and update the models based on that (we don't use shared layers here)
        """

        if not self.made_models_similar:
            self.made_models_similar = True
            models = []
            for client in self.clients:
                models.append(client.model)

            self.average_layer_parameters(models)

        models = []
        for client in self.clients:
            models.append(client.model)

        aggregated_grads = self.find_average_momentum(models)

        for model in models:
            model.update(aggregated_grads, learning_rate)