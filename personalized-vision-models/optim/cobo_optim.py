from .base_optim import Optim
import torch.optim as optim
from line_profiler import profile
import torch
import copy
import concurrent.futures
class CoboOptim(optim.Optimizer, Optim):

    def __init__(self, params, clients, device, grouping, shared_layers, learning_rate):
        super().__init__(params, {})
        self.grouping = grouping
        self.shared_layers = shared_layers
        self.clients = clients
        self.device = device
        self.params = params
        self.cloned_models = list()

        for client in self.clients:
            cloned_model_parameters = [param.data.detach().clone() for param in client.model.parameters()]
            self.cloned_models.append(cloned_model_parameters)

    def convex_objective(self, learning_rate, p1_data, p2_data, adj):
        return learning_rate * self.grouping.rho * (p1_data - p2_data)/adj

    # @profile
    def general_fw_update(self, learning_rate, ind1):
        # with torch.no_grad():
        client1 = self.clients[ind1]

        # for ind2, client2 in enumerate(self.clients):
        #     if client1 != client2:
        #         less_ind = min(ind1, ind2)
        #         more_ind = max(ind1, ind2)
        #         coeff = learning_rate * self.grouping.rho * self.grouping.w_adjacency[less_ind, more_ind]

        momentum = client1.model.get_momentum()
        # momentum = client1.model.get_gradients()
        for param_ind, p1 in enumerate(client1.model.parameters()):
            diff = self.cloned_models[param_ind][ind1, None] - self.cloned_models[param_ind]
            w_expanded = self.grouping.w_adjacency[ind1]
            for i in range(diff.dim()-1):
                w_expanded = w_expanded.unsqueeze(i+1)
            # w_expanded = self.grouping.w_adjacency[ind1].unsqueeze(tuple(range(diff.dim() - 1)))
            diff *= w_expanded
            diff = torch.sum(diff, dim=0)
            # breakpoint()
            # if self.shared_layers[param_ind] == 1:
                # breakpoint()
                # try:
                # diff = p1_clone - p2_clone.to(p1_clone.device)
            p1.data -= diff.to(p1.device) * learning_rate * self.grouping.rho
                # except RuntimeError as e:
                #     breakpoint()
                # p1.data.add_(diff, alpha=-coeff)
            p1.data.add_(momentum[param_ind], alpha=-learning_rate)


    # def update_local_model(self, model, learning_rate, num_clients):
    #     # momentum = model.get_momentum()
    #     momentum = model.get_gradients()
    #     for i, p in enumerate(model.parameters()):
    #         p.data.add_(momentum[i], alpha=-learning_rate)

    # @profile
    def step(self, learning_rate):
        # print('just to make sure that we get here!')
        # breakpoint()
        num_clients = len(self.clients)

        individual_clones = list()
        self.cloned_models = list()
        for i, client in enumerate(self.clients):
            cloned_model = copy.deepcopy(client.model)
            cloned_model_parameters = [param.data for param in cloned_model.parameters()]
            individual_clones.append(cloned_model_parameters)

        for i in range(len(individual_clones[0])):
            params = list()
            for j in range(num_clients):
                params.append(individual_clones[j][i].to(self.device))
            self.cloned_models.append(torch.stack(params))


        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = []
        for ind1, client1 in enumerate(self.clients):
            self.general_fw_update(learning_rate, ind1)
            # futures.append(executor.submit(self.general_fw_update, learning_rate, ind1))


            # for ind2, client2 in enumerate(self.clients):
            #     if client1 != client2:
            #         less_ind = min(ind1, ind2)
            #         more_ind = max(ind1, ind2)
            #         # models_params = (client1.model.parameters(), client2.model.parameters())
            #
            #         for param_ind, p1 in enumerate(client1.model.parameters()):
            #             p1_clone = self.individual_clones[less_ind][param_ind]
            #             p2_clone = self.individual_clones[more_ind][param_ind]
            #             if self.shared_layers[param_ind] == 1:
            #                 update = self.general_fw_update(learning_rate, p1_clone.to(p1.device), p2_clone.to(p1.device), self.grouping.w_adjacency[less_ind, more_ind].to(p1.device))
            #                 p1.data.add_(update, alpha=-1)

                                # ((learning_rate * self.grouping.rho * (p1_clone.data - p2_clone.data))/
                                #         (self.grouping.w_adjacency[less_ind, more_ind]))

                            # sign = 1 if ind1 < ind2 else -1
                            # p1.data -= self.learning_rate * self.grouping.w_adjacency[less_ind, more_ind] * sign * unflatten_lambda[param_ind]

        # for i, client in enumerate(self.clients):
        #     for param_ind, (p1, cloned_p1, initial_p1) in enumerate(zip(client.model.parameters(), individual_clones[i].parameters(), self.initial_models[i].parameters())):
        #         if self.shared_layers[param_ind] == 1:
        #             p1.data -= 1 / num_clients * self.learning_rate * (cloned_p1.data - initial_p1.data)

            # concurrent.futures.wait(futures)
            # futures = []
            # cnt = 0
            # for client in self.clients:
            # #     self.update_local_model(client.model, learning_rate, num_clients)
            #     futures.append(executor.submit(self.update_local_model, client.model, learning_rate, num_clients))
            #
            # concurrent.futures.wait(futures)

        # for i, client in enumerate(self.clients):
            # breakpoint()
        # for client in self.clients:

        # print("I get here!")
        # for i, param_group in enumerate(self.param_groups):
        #     momentum = self.clients[i].model.get_momentum()
        #     for j, p in enumerate(self.clients[i].model.parameters()):
        #         # if p.grad is not None:
        #         p.data -= learning_rate * momentum[j]
        #
        #     self.clients[i].model.previous_momentum = self.clients[i].model.get_momentum()

