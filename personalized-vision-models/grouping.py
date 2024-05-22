import copy

from models.resnet import ResNet9
from models.smallnet import SmallNet
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import math
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
# from line_profiler import profile
import copy
class Grouping:

    def __init__(self, n_clients, learning_rate, alpha=None, rho=None, w_adjacency=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_clients = n_clients
        if w_adjacency is None:
            self.w_adjacency = torch.ones((self.n_clients, self.n_clients), device=self.device)
            # self.w_adjacency = torch.zeros((self.n_clients, self.n_clients), device=self.device)
        else:
            self.w_adjacency = w_adjacency

        # self.w_adjacency = torch.ones((self.n_clients, self.n_clients), device=self.device)
        # self.w_adjacency = torch.eye(self.n_clients, device=self.device)
        # cnt = 0
        # for i in range(10, 5, -1):
        #     for _ in range(2):
        #         self.w_adjacency[cnt:cnt+i, cnt:cnt+i] = 1
        #         cnt += i

        # a = [1,1,0,0,0,0,0,0]
        # b = [0,0,1,1,0,0,0,0]
        # c = [0,0,0,0,1,1,0,0]
        # d = [0,0,0,0,0,0,1,1]
        # self.w_adjacency = torch.tensor([a, a, b, b, c, c, d, d], device=self.device, dtype=torch.float32)
        self.step = 0
        self.alpha = alpha
        self.rho = rho
        print('in the constructor the rho is:', self.rho)
        self.smooth_grads = torch.zeros(self.n_clients, device=self.device)
        self.patience = torch.zeros(self.n_clients, device=self.device)
        self.differences = None
        self.prv = 0
        self.norm_squared_differences = None
        self.alpha_encourage = torch.zeros(self.n_clients, device=self.device)
        self.beta = 0.1
        self.learning_rate = learning_rate
        self.exp_avg_grad_list = None
        self.mid_param_models = list()

    def get_flattened_grad_and_param(self, model, shared_layers):
        flattened_grad = list()
        flattened_param = list()
        for i, param in enumerate(model.parameters()):
            if shared_layers[i] == 1:
                flattened_grad.append(torch.reshape(param.grad.clone(), (-1,)))
                flattened_param.append(torch.reshape(param.data.clone(), (-1,)))

        return torch.cat(flattened_grad), torch.cat(flattened_param)

    def cobo(self, clients, train):
        """Calculate the matrix norm_differences. The (i,j)-th entry is the l2 norm of difference between i-th and j-th model shared layers"""
        flattened_params_list = list()
        flattened_grads_list = list()
        for client in clients:
            flattened_grad, flattened_model = self.get_flattened_grad_and_param(client.model, train.shared_layers)

            flattened_params_list.append(flattened_model)
            flattened_grads_list.append(flattened_grad)

        if self.exp_avg_grad_list is None:
            self.exp_avg_grad_list = flattened_grads_list
        else:
            for i in range(len(self.exp_avg_grad_list)):
                self.exp_avg_grad_list[i] = 0.99 * self.exp_avg_grad_list[i] + 0.01 * flattened_grads_list[i]
                # breakpoint()
                # self.exp_avg_grad_list[i][flattened_grads_list[i] > 4] = \
                #     (0.9 * self.exp_avg_grad_list[i] + 0.1 * flattened_grads_list[i])[flattened_grads_list[i] > 4]
                # self.exp_avg_grad_list[i][flattened_grads_list[i] < -4] = (
                #         0.9 * self.exp_avg_grad_list[i] + 0.1 * flattened_grads_list[i])[flattened_grads_list[i] < -4]

        grad_inner_product = torch.zeros((self.n_clients, self.n_clients), device=self.device)
        real_inner_product = torch.zeros((self.n_clients, self.n_clients), device=self.device)
        grad_difference_inner_product = torch.zeros((self.n_clients, self.n_clients), device=self.device)

        # for i in range(len(flattened_grads_list)):
        #     for j in range(len(flattened_grads_list)):
        #         if i != j:
        #             grad_inner_product[i, j] = torch.dot(self.exp_avg_grad_list[i], self.exp_avg_grad_list[j])
        #             real_inner_product[i, j] = torch.dot(flattened_grads_list[i], flattened_grads_list[j])
        #             grad_difference_inner_product[i, j] = torch.dot(flattened_grads_list[i],
        #                                                             flattened_params_list[i] - flattened_params_list[j])


        mid_param_grad_inner_product = self.mid_param_grad_inner_prod(clients, train.shared_layers, sampling=True)


        # global_avrg_param_grad_inner_product = self.global_avrg_param_grad_inner_prod(clients, train.shared_layers)

        # stacked_gradients = torch.stack(flattened_grads_list)
        # grad_normxs = torch.norm(stacked_gradients, p=2, dim=-1)
        # stacked_exp_avg_grads = torch.stack(self.exp_avg_grad_list)
        # exp_avg_grad_norms = torch.norm(stacked_exp_avg_grads, p=2, dim=-1)

        # stacked_params = torch.stack(flattened_params_list)
        # self.differences = stacked_params[:, None, :] - stacked_params[None, :, :]
        # self.next_step_differences = (
        #             (stacked_params[:, None, :] - train.learning_rate * stacked_gradients[:, None, :]) -
        #             (stacked_params[None, :, :] - train.learning_rate * stacked_gradients[None, :, :]))
        # self.norm_squared_differences = torch.norm(self.differences, p=2, dim=-1) ** 2
        # self.next_step_norm_squared_differences = torch.norm(self.next_step_differences, p=2, dim=-1) ** 2

        """update w_adjacency"""


        self.step += 1
        applied_alpha = self.alpha
        if self.step > 20000:
            applied_alpha = self.alpha / self.step

        # self.update_convex_objective(train.learning_rate, applied_alpha)
        self.grad_based_update(mid_param_grad_inner_product)


        """logging and printing"""
        neighbor_distance = torch.norm(flattened_params_list[0] - flattened_params_list[1], p=2)
        non_neighbor_distance = torch.norm(flattened_params_list[0] - flattened_params_list[-1], p=2)
        print('neighbor distance:', neighbor_distance)
        print('non-neighbor distance:', non_neighbor_distance)

        if train.master_process and self.step % 100 == 0 and self.step > 0:



            print('w:', self.w_adjacency)
            # print('differences:\n', self.rho * self.norm_squared_differences / 2)
            print('exp-avg-grad-inner-product', grad_inner_product)
            # print('exp-avg-grad-norm', exp_avg_grad_norms)
            # print('grad_norm', grad_norms)
            print('grad-inner-product', real_inner_product)
            # print('mid-param-grad-inner-product', mid_param_grad_inner_product)
            print('grad-difference-inner-product', grad_difference_inner_product)

            matrix_clone = self.w_adjacency.clone().cpu().numpy()
            np.fill_diagonal(matrix_clone, np.nan)

            plt.imshow(matrix_clone, cmap='viridis', interpolation='none', vmin=0, vmax=1)
            wandb.log({'w_sum': torch.sum(self.w_adjacency),
                       # 'alpha': applied_alpha,
                       'neighbor_distances': neighbor_distance,
                       'non-neighbor_distance': non_neighbor_distance,
                       'adjacency matrix': wandb.Image(plt)}, step=self.step//100)
            # matrix_clone = mid_param_grad_inner_product.clone().cpu().numpy()
            # np.fill_diagonal(matrix_clone, np.nan)
            # plt.imshow(matrix_clone, cmap='viridis', interpolation='none')
            # wandb.log({'mid_param_inner_product': wandb.Image(plt)}, step=self.step//100)

    def project_w_row_stochastic(self, applied_alpha):
        """update w_adjacency"""
        w_grad = self.norm_squared_differences * self.rho / 2 - applied_alpha
        self.w_adjacency -= self.learning_rate * w_grad

        for i in range(self.w_adjacency.shape[0]):
            row_sorted, indices = torch.sort(self.w_adjacency[i], descending=True)
            partial_sum = 0
            max_ind = 0
            sum_max_ind = 0
            for j in range(len(indices)):
                partial_sum += row_sorted[j]
                if row_sorted[j] + 1/(j+1)*(1 - partial_sum) > 0:
                    max_ind = j
                    sum_max_ind = partial_sum

            tao = 1/(max_ind + 1)*(1 - sum_max_ind)
            self.w_adjacency[i] += tao
            self.w_adjacency[i][self.w_adjacency[i] < 0] = 0

    def update_convex_objective(self, learning_rate, applied_alpha):
        self.w_adjacency -= learning_rate * (-(self.rho/2 * self.norm_squared_differences)/(self.w_adjacency ** 2) + applied_alpha)
        self.w_adjacency[self.w_adjacency < 1] = 1
        self.w_adjacency[self.w_adjacency > 10] = 10
        return


    def grad_based_update(self, grad_inner_product):

        # s = torch.ones((self.n_clients, self.n_clients), device=self.device)
        # s[grad_inner_product < 0] = 0
        # s[grad_inner_product == 0] = self.w_adjacency[grad_inner_product == 0]
        #
        # self.w_adjacency = 0.99 * self.w_adjacency + 0.01 * s

        self.w_adjacency += 0.01 * grad_inner_product
        self.w_adjacency[self.w_adjacency > 1] = 1
        self.w_adjacency[self.w_adjacency < 0] = 0

        # row_sums = torch.sum(self.w_adjacency, dim=1)
        # self.w_adjacency = self.w_adjacency / row_sums[:, None]
        # self.w_adjacency[grad_inner_product < -4] = (0.9 * self.w_adjacency + 0.1 * s)[grad_inner_product < -4]
        # self.w_adjacency[grad_inner_product > 4] = (0.9 * self.w_adjacency + 0.1 * s)[grad_inner_product > 4]

    def _choose_indices_upper_triangular(self, matrix_shape, prob):
        indices = np.triu_indices(matrix_shape, k=1)
        num_elements = len(indices[0])
        random_numbers = np.random.rand(num_elements)
        mask = random_numbers <= prob

        chosen_indices = (indices[0][mask], indices[1][mask])

        return chosen_indices

    def _calculate_mid_param_models(self, i, j, clients, ind, shared_layers):
        # if len(self.mid_param_models) <= ind:
        #     self.mid_param_models.append((ResNet9(), ResNet9()))
        # print('it comes inside the function')
        mid_param1 = self.mid_param_models[ind][0]

        device = next(clients[i].model.parameters()).device
        mid_param1 = copy.deepcopy(clients[i].model)
        mid_param1.to(device)
        mid_param1.zero_grad()
        mid_param1.train()
        for param, client1_param, client2_param in zip(mid_param1.parameters(), clients[i].model.parameters(),
                                                       clients[j].model.parameters()):
            param.data.mul_(0.5)
            param.data.add_(client2_param.data.to(device), alpha=0.5)

        data, target = next(clients[i].get_next_batch_train())
        output = mid_param1(data.to(device), targets=target.to(device), get_logits=True)
        loss = output['loss']
        loss.backward()
        # grad_i, _ = self.get_flattened_grad_and_param(mid_param1, shared_layers)

        # mid_param2 = ResNet9().to(device)
        mid_param2 = copy.deepcopy(mid_param1)
        # mid_param2 = self.mid_param_models[ind][1]
        mid_param2.to(device)
        mid_param2.zero_grad()
        mid_param2.train()
        data, target = next(clients[j].get_next_batch_train())
        output = mid_param2(data.to(device), targets=target.to(device), get_logits=True)
        loss = output['loss']
        loss.backward()
        print(i , 'and', j, 'are done backwarding!')
        # breakpoint()
        self.mid_param_models[ind] = (mid_param1, mid_param2)
        # grad_j, _ = self.get_flattened_grad_and_param(self.mid_param, shared_layers)

        # return torch.dot(grad_i, grad_j)
    def _calculate_inner_product(self, model1, model2, shared_layers):
        grad_i, _ = self.get_flattened_grad_and_param(model1, shared_layers)
        grad_j, _ = self.get_flattened_grad_and_param(model2, shared_layers)
        prod = torch.dot(grad_i, grad_j.to(grad_i[0].device))
        # if math.isnan(prod):
        #     breakpoint()
        return prod

    def mid_param_grad_inner_prod(self, clients, shared_layers, sampling=False):
        """"this function get models i and j, and builds a model with average parameters of these, then it calculates
        two gradients on this model: gradient of dataset i and gradient of dataset j. Then it calculates the
        inner product for the gradients
        If sampling is true, it only does it for sz pairs uniformly at random, from the upper triangular"""

        sz = len(clients)
        mid_param_grad_inner_product = torch.zeros((sz, sz), device=self.device)

        if sampling:
            # p_sample = min(1.0, 100/(self.step+1))
            # p_sample = 1/10
            p_sample = 2/sz
            if self.step > 800:
                p_sample = 1/math.sqrt(self.step)
            chosen_indices = self._choose_indices_upper_triangular(sz, p_sample)
            print('the pairs are ', chosen_indices)

            cnt = 0
            for _ in range(max(0,len(chosen_indices[0]) - len(self.mid_param_models))):
                self.mid_param_models.append((ResNet9(), ResNet9()))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                cnt = 0
                for i, j in zip(*chosen_indices):
                    # print('i and j are:', i, j)
                    # if cnt == 0:
                    futures.append(executor.submit(self._calculate_mid_param_models, i, j, clients, cnt, shared_layers))
                    cnt += 1
                # self._calculate_mid_param_models(i, j, clients, cnt, shared_layers)
                # cnt += 1
                concurrent.futures.wait(futures)

            for future in futures:
                try:
                    future.result()  # This will raise any exceptions caught in the worker threads
                except Exception as e:
                    print(f"An error occurred: {e}")
                    breakpoint()

            cnt = 0
            for i, j in zip(*chosen_indices):
                if list(self.mid_param_models[cnt][0].parameters())[0].grad is None or list(self.mid_param_models[cnt][1].parameters())[0].grad is None:
                    print('no gradient')
                    breakpoint()
                product = self._calculate_inner_product(self.mid_param_models[cnt][0], self.mid_param_models[cnt][1], shared_layers)
                print('product of ', i, j, 'is:', product)
                if math.isnan(product):
                    breakpoint()
                cnt += 1
                mid_param_grad_inner_product[i, j] = product
                mid_param_grad_inner_product[j, i] = product

            print('inner products are calculated')
            return mid_param_grad_inner_product

        for i in range(sz):
            for j in range(i+1, sz):
                product = self._calculate_mid_param_models(i, j, clients, shared_layers)
                mid_param_grad_inner_product[i, j] = product
                mid_param_grad_inner_product[j, i] = product

        return mid_param_grad_inner_product

    def global_avrg_param_grad_inner_prod(self, clients, shared_layers):
        """The same as mid_param_grad_inner_prod, but it builds a model with params averaged over all the models and
        evaluate it for each of them separately"""

        mid_param = ResNet9()
        mid_param.to(self.device)
        mid_param.zero_grad()
        mid_param.train()

        for param in mid_param.parameters():
            param.data = torch.zeros_like(param.data)

        for i in range(len(clients)):
            for param, client_param in zip(mid_param.parameters(), clients[i].model.parameters()):
                param.data += client_param.data / len(clients)

        grads = list()
        for i in range(len(clients)):
            data, target = clients[i].get_next_batch_train()
            output = mid_param(data.to(self.device), targets=target.to(self.device), get_logits=True)
            loss = output['loss']
            loss.backward()
            grad_i, _ = self.get_flattened_grad_and_param(mid_param, shared_layers)
            grads.append(grad_i)
            mid_param.zero_grad()

        global_avrg_param_grad_inner_product = torch.zeros((len(clients), len(clients)), device=self.device)

        for i in range(len(clients)):
            for j in range(i+1, len(clients)):
                global_avrg_param_grad_inner_product[i, j] = torch.dot(grads[i], grads[j])
                global_avrg_param_grad_inner_product[j, i] = global_avrg_param_grad_inner_product[i, j]

        return global_avrg_param_grad_inner_product

























