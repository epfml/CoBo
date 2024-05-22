import torch
import torch.nn.functional as F
import wandb
from optim.cobo_optim import CoboOptim
from optim.grad_average_optim import GradAverageOptim
from optim.weighted_grad_average_optim import WeightedGradAverageOptim
from optim.federated_clustering_optim import FederatedClusteringOptim
from optim.ditto_optim import DittoOptim
from optim.train_alone_optim import TrainAloneOptim
from optim.ifca_optim import IFCAOptim
import torch.optim as torch_optim
from line_profiler import profile
class Train:
    """
    This is a class that implements different training methods
    groups: We know the grouping of all clients, which means that they have the same dataset to train on
    shared_layer: this is a mask of layers that are shared between all the clients. Other layers are only shared within each group
    """

    def __init__(self, groups, learning_rate, known_grouping, master_process, shared_layers=None, grouping=None, config=None):
        self.test_grad = None
        self.groups = groups
        self.learning_rate = learning_rate
        self.known_grouping = known_grouping
        self.train_loaders, self.val_loaders, self.test_loaders = list(), list(), list()
        self.models, self.clients = list(), list()
        self.initial_models = list()
        self.grouping = grouping
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_losses = torch.zeros(len(self.models), device=self.device)
        self.master_process = master_process
        self.config = config
        for group in self.groups:
            for client in group.clients:
                self.clients.append(client)
                self.models.append(client.model)

                initial_model = client.group.model(config)
                initial_model.load_state_dict(client.model.state_dict())
                self.initial_models.append(initial_model.to(self.device))
                self.train_loaders.append(client.dataset.train_loader)
                self.test_loaders.append(client.dataset.test_loader)
                self.val_loaders.append(client.dataset.val_loader)

        if shared_layers is not None:
            self.shared_layers = shared_layers

    def get_optim(self, optim_type):
        params = []
        for model in self.models:
            params.append({"params": model.parameters()})

        optimizers = {'cobo': CoboOptim(params, self.clients, self.device, self.grouping,
                                               self.shared_layers, self.learning_rate),
                      'weighted_grad_average': WeightedGradAverageOptim(params, self.clients, self.device, self.groups,
                                                                        self.shared_layers, self.learning_rate),
                      'grad_average': GradAverageOptim(params, self.clients, self.device, self.learning_rate),
                      'federated_clustering': FederatedClusteringOptim(params, self.clients, self.device,
                                                                       self.learning_rate, self.grouping,
                                                                       percentile=self.config.fc_percentile),
                      'ditto': DittoOptim(params, self.clients, self.device, self.grouping, self.learning_rate,
                                          k=self.config.ditto_k, w_lambda=self.config.ditto_lambda),
                      'train_alone': TrainAloneOptim(params, self.clients, self.device, self.learning_rate),
                      'ifca': IFCAOptim(params, self.clients, self.device, self.learning_rate, self.config.ifca_k,
                                        self.config.ifca_m, self.config),
                      }
        return optimizers[optim_type]

    def one_client_evaluation(self, client, max_num_batches=2):
        """evaluating accuracy and loss based on the first client model and dataset"""
        model = client.model
        model.zero_grad()
        model.eval()
        test_loss = 0
        correct = 0
        loss_list_val, acc_list = [], []

        with torch.no_grad():
            for i in range(max_num_batches):
                data, target = client.get_next_batch_test()
                # print('data and target test shape are:', data.shape, target.shape)
                local_device = next(client.model.parameters()).device
                output = model(data.to(local_device), targets=target.to(local_device), get_logits=True)
                val_loss = output['loss']
                loss_list_val.append(val_loss)
                # breakpoint()
                acc_list.append((output['logits'].argmax(-1) == target.to(local_device)).float().mean())
                # test_loss += F.cross_entropy(output['loss'], target.to(self.device)).detach()
                # pred = output.data.max(1, keepdim=True)[1]
                # correct += pred.eq(target.to(self.device).data.view_as(pred)).sum()

        val_acc = torch.stack(acc_list).mean().item()
        val_loss = torch.stack(loss_list_val).mean().item()
        val_perplexity = 0
                # 2.71828 ** val_loss)



        return val_acc, val_loss, val_perplexity

        # test_loss /= test_num
        # test_acc = 100. * correct / test_num
        # print('size of test set:', test_num)
        # wandb.log({"accuracy": test_acc, "loss": test_loss})
        # print('accuracy and loss', test_acc, test_loss)
        # return test_acc, test_loss


    def neighbors_gradient_averaging(self):
        average_grad_all = self.find_average_gradients(self.models)

        for model in self.models:
            for i, param in enumerate(model.parameters()):
                if self.shared_layers[i] == 1:
                    param.data -= self.learning_rate * average_grad_all[i]

        for client in self.clients:
            average_grad_neighbors = self.find_average_gradients(client.neighbor_models)
            group_rate = len(client.neighbor_models) / len(self.models)
            for i, param in enumerate(client.model.parameters()):
                if self.shared_layers[i] == 0:
                    param.data -= self.learning_rate * group_rate * average_grad_neighbors[i]


    def shared_model_evaluation(self):
        """evaluating accuracy and loss based on average of accuracy and loss of all agents"""
        global_loss, global_acc, global_perplexity = 0, 0, 0
        for i, client in enumerate(self.clients):
            test_acc, test_loss, test_perplexity = self.one_client_evaluation(client)
            print('client ', i, "test accuracy and loss: ", test_acc, test_loss, test_perplexity)
            global_loss += test_loss
            global_acc += test_acc
            global_perplexity += test_perplexity

        print('global acc:', global_acc / len(self.models)*100, 'global loss: ', global_loss / len(self.models),
              'global perplexity: ', global_perplexity / len(self.models))
        wandb.log({"accuracy": global_acc / len(self.models)*100, "loss": global_loss / len(self.models),
                   "perplexity": global_perplexity / len(self.models)})

        return global_acc, global_loss, global_perplexity
    #
    def train(self, optim, evaluate, iterations, lr_scheduler=None, grouping_method=None,
              start_iteration_number=0, partitioning=None, run_id=None, acc_steps=1):
        """
        main training loop
        aggregator: one of the methods for aggregating gradients/parameters and updating the models
        evaluate: one of the methods for evaluating the models
        """

        optims = list()
        for client in self.clients:
            optims.append(torch_optim.SGD(client.model.parameters(), lr=0.01))

        for i in range(start_iteration_number, iterations):
            print('iteration number: ', i)
            for client in self.clients:
                client.model.zero_grad()
                client.model.train()

                for microstep_idx in range(acc_steps):
                    local_device = next(client.model.parameters()).device
                    next_batch = next(client.get_next_batch_train())
                    # next_batch = client.get_next_batch_train()
                    # breakpoint()
                    inputs, targets = next_batch[0].to(local_device), next_batch[1].to(local_device)
                    output = client.model(inputs, targets=targets)    # TODO: make the output of vision models the same as this!
                    # loss = output["loss"]/acc_steps
                    loss = output["loss"]
                    loss.backward()
            #
                    # loss = F.cross_entropy(output, next_batch[1].to(self.device))

            # groups_models = []
            # groups_momentums = []
            # for k in range(len(self.groups)):
            #     group_models = []
            #     group_momentums = []
            #     for j in range(len(self.groups[k].clients)):
            #         group_models.append(self.groups[k].clients[j].model.state_dict())
            #         group_momentums.append(self.groups[k].clients[j].model.previous_momentum)
            #     groups_models.append(group_models)
            #     groups_momentums.append(group_momentums)

            # for i, p in enumerate(self.clients[0].model.parameters()):
            #     if i == 0:
            #         print('grad is ', p.grad[0])

            # aggregator()
            # params = list(self.models[0].parameters())
            # print('the first set of params', params[0].data[0], self.models[0].get_momentum()[0][0])
            optim.step(self.learning_rate)
            if list(self.models[0].parameters())[0].data is None:
                breakpoint()
            # for i in range(len(self.clients)):
            #     optims[i].step()
            # print('after applying:', params[0].data[0])
            # breakpoint()

            if not self.known_grouping:
                print('finding the grouping')
                grouping_method(self.clients, self)
            # grads = self.models[0].get_gradients()
            # for i, p in enumerate(self.models[0].parameters()):
            #     p.data -= self.learning_rate * grads[i]

            if lr_scheduler is not None:
                lr_scheduler.step()
                self.learning_rate = lr_scheduler.get_last_lr()[0]

            if self.master_process and i % 100 == 0 and i > 0:
                print('iteration number: ', i)
                evaluate()
            #
            #     checkpoint_dict = {
            #                     "models": groups_models,
            #                     "w_adjacency": self.grouping.w_adjacency,
            #                     "partitioning": partitioning,
            #                     "learning_rate": self.learning_rate,
            #                     "momentum": groups_momentums,
            #                     "starting_iteration": i
            #     }
            #
            #
            #     print("file path:", file_path)
            #     # torch.save(checkpoint_dict, file_path + run_id + '_last.pt')
            #     wandb.save(file_path + run_id + '_last.pt')
            #
            #     # torch.save(checkpoint_dict, file_path + run_id + '_epoch' + str(i) + '.pt')
            #     wandb.save(file_path + run_id + '/epoch' + str(i) + '.pt')