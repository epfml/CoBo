import json
import time
from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
import matplotlib.pyplot as plt
from .utils import eval, get_batch
# from src.models.utils import get_model

def train_lora(clients, data, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path,
               distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype)

    num_clients = len(clients)
    itr, substep, best_val_loss, text_table = [0] * num_clients, [0] * num_clients, [
        float('inf')] * num_clients, None  # best_val_loss not used atm, early stopping not recommended but possible

    stats = {'train_loss': [[] for _ in range(num_clients)], 'val_loss': [[] for _ in range(num_clients)],
             'val_pp': [[] for _ in range(num_clients)], 'val_acc': [[] for _ in range(num_clients)]}

    num_substeps_per_epoch = []
    for i in range(num_clients):
        num_substeps_per_epoch.append(len(data['train'][i]) // (batch_size * sequence_length))

    if not extra_args.no_compile:
        print(f'Compiling model ...')
        for i in range(num_clients):
            clients[i][0] = torch.compile(clients[i][0], dynamic=True)  # requires pytorch 2.0+

    for i in range(num_clients):
        clients[i][0].train()

    trust_weights = (torch.ones((len(clients), len(clients))) - torch.eye(len(clients)))/(len(clients) - 1)

    global_model = None
    # trust_weights = torch.eye(len(clients))
    # for i in range(0, len(clients), 2):
    #     trust_weights[i, i+1] = 1
    #     trust_weights[i+1, i] = 1

    # print("I get here!")
    # breakpoint()
    t0 = time.time()
    while itr[-1] < iterations:
        for i in range(num_clients):
            print(f'\r{i} {itr[i]}', end='')
            model, opt, scheduler = clients[i]

            for microstep_idx in range(acc_steps):  # gradient accumulation
                x, y = get_batch(data['train'][i], sequence_length, batch_size, device=extra_args.device)
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx,
                                                                               gradient_accumulation_steps=acc_steps):
                        outputs = model(x, targets=y)

                loss = outputs['loss'] / acc_steps
                loss.backward()
                substep[i] += 1

                # breakpoint()
            if extra_args.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

            opt.step()
            scheduler.step()
            itr[i] += 1

        if extra_args.trust == 'cobo' and itr[-1] >= extra_args.pretraining_rounds - 1:
            __objective_function_update(clients, trust_weights)
        # if itr[-1] == 5:
        #     breakpoint()
        # distribute gradient
        if itr[-1] % extra_args.trust_freq == 0 and itr[-1] >= extra_args.pretraining_rounds - 1:
            if extra_args.trust == 'none':
                pass
            elif extra_args.trust == 'ditto':
                # breakpoint()
                global_model = ditto(clients, data, batch_size, sequence_length, extra_args, type_ctx, distributed_backend,
                                     global_model)
            elif extra_args.trust == 'cobo':
                # print('iterator:', itr[-1])
                # if itr[-1] == 5:
                #     print('yes!')
                #     for i in range(100):
                #         print(i)
                trust_weights = __cobo(clients, trust_weights, data, batch_size, sequence_length, extra_args,
                                       type_ctx, distributed_backend)
                # else:
                #     print('no!')
            elif extra_args.trust == 'naive':
                __average(clients)
            elif extra_args.trust == 'static':
                __average_static(clients, extra_args.dataset)
            elif extra_args.trust == 'dynamic':
                __average_dynamic(clients)
            elif extra_args.trust == 'dynamic-thresh':
                __average_dynamic_threshold(clients)
            elif extra_args.trust == 'dynamic-top-k':
                __average_dynamic_top_k(clients, extra_args.k)
            elif 'ref' in extra_args.trust:
                res = torch.zeros((num_clients, num_clients))
                for model, _, _ in clients:
                    model.eval()
                for id1 in range(len(clients)):
                    for id2, (model, _, _) in enumerate(clients):
                        model.eval()
                        _, _, val_perplexity = eval(model, data['val'][id1], sequence_length, batch_size,
                                                    extra_args.device, max_num_batches=12, ctx=type_ctx)
                        res[id1, id2] = val_perplexity
                        model.train()
                for model, _, _ in clients:
                    model.train()
                res = -res
                print(res)
                if extra_args.trust == 'dynamic-ref':
                    __average_dynamic_ref(clients, res)
                elif extra_args.trust == 'dynamic-thresh-ref':
                    __average_dynamic_threshold_ref(clients, res)
                elif extra_args.trust == 'dynamic-top-k-ref':
                    __average_dynamic_top_k_ref(clients, res, extra_args.k)
            elif 'token' in extra_args.trust:
                logits = [[] for _ in range(len(clients))]
                for model, _, _ in clients:
                    model.eval()
                for j in range(4):
                    print(f'\r{j} batch ref', end='')
                    x, y = get_batch(data['ref'], sequence_length, batch_size, extra_args.device)
                    for id, (model, _, _) in enumerate(clients):
                        with type_ctx:
                            outputs = model(x, get_logits=True)
                        logits_out = outputs['logits'].detach()
                        v, _ = torch.topk(logits_out, 100)
                        logits_out[logits_out < v[:, :, [-1]]] = 0
                        logits_out = logits_out.to_sparse_coo()
                        logits[id].append(logits_out)
                for model, _, _ in clients:
                    model.train()

                res = torch.zeros((num_clients, num_clients))
                for id1 in range(len(clients)):
                    for id2 in range(len(clients)):
                        sim = 0
                        for j in range(4):
                            sim += torch.sum(torch.abs(logits[id1][j] - logits[id2][j])).item()
                        res[id1, id2] = sim / (4 * batch_size)

                res = F.normalize(res, p=1, dim=1)
                res = -res * 10
                print(res)
                if extra_args.trust == 'dynamic-token':
                    __average_dynamic_token(clients, res)
                elif extra_args.trust == 'dynamic-thresh-token':
                    __average_dynamic_threshold_token(clients, res)
                elif extra_args.trust == 'dynamic-top-k-token':
                    __average_dynamic_top_k_token(clients, res, extra_args.k)

        # from here it's only evaluation code, all the training is above
        t1 = time.time()
        dt = (t1 - t0) / num_clients
        for i in range(num_clients):
            model, opt, scheduler = clients[i]
            opt.zero_grad(set_to_none=True)

            if itr[i] % eval_freq == 0 or itr[i] == iterations:
                if distributed_backend.is_master_process():
                    epoch = substep[i] // num_substeps_per_epoch[i]

                    model.eval()
                    train_loss = loss.detach().cpu().item() * acc_steps
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                    val_acc, val_loss, val_perplexity = eval(model, data['val'][i], sequence_length, batch_size,
                                                             extra_args.device, max_num_batches=12, ctx=type_ctx)

                    print_string = f"{i}: {epoch}/{itr[i]} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                    print_string += f" [time per itr] {dt * 1000 / eval_freq:.2f}ms"
                    if scheduler is not None:
                        print_string += f" [lr] {current_lr:.5f}"
                    print(f'\r{print_string}')

                    stats['train_loss'][i].append(train_loss)
                    stats['val_loss'][i].append(val_loss)
                    stats['val_pp'][i].append(val_perplexity)
                    stats['val_acc'][i].append(val_acc)

                    if extra_args.wandb:
                        if i == (num_clients - 1):
                            wandb.log({
                                f"train/loss_mean": np.mean([stats['train_loss'][i][-1] for i in range(num_clients)]),
                                f"val/loss_mean": np.mean([stats['val_loss'][i][-1] for i in range(num_clients)]),
                                f"val/perplexity_mean": np.mean([stats['val_pp'][i][-1] for i in range(num_clients)]),
                                f"val/acc_mean": np.mean([stats['val_acc'][i][-1] for i in range(num_clients)]),
                            }, commit=False)
                        wandb.log({
                            f"iter_{i}": itr[i],
                            f"train/loss_{i}": train_loss,
                            f"val/loss_{i}": val_loss,
                            f"val/perplexity_{i}": val_perplexity,
                            f"val/acc_{i}": val_acc,
                            f"lr_{i}": current_lr,
                        }, commit=(i == (num_clients - 1)))

                    model.train()
        if itr[-1] % eval_freq == 0 or itr[-1] == iterations:
            for idx, c in enumerate(clients):
                model, _, _ = c
                torch.save(model.state_dict(), f'{idx}_{itr[-1]}')
        t0 = time.time()

    return stats

def ditto(clients, data, batch_size, sequence_length, extra_args, type_ctx, distributed_backend, global_model=None):
    sz = len(clients)
    print('k is:', extra_args.k)
    if global_model is None:
        for client in clients:
            model, _, _ = client
            cnt = 0
            if global_model is None:
                global_model = []
                for p in model.parameters():
                    if p.requires_grad:
                        global_model.append(p.data.clone())
            else:
                for i, p in enumerate(model.parameters()):
                    if p.requires_grad:
                        global_model[cnt] += p.data.clone().to(global_model[cnt].device)
                        cnt += 1

        for i in range(len(global_model)):
            global_model[i] /= sz

    selected_clients = torch.randperm(sz)[:extra_args.k]
    print('it gets here', selected_clients)

    for idx in selected_clients:
        client = clients[idx]
        model, _, _ = client

        model.zero_grad()
        model.train()

        # for microstep_idx in range(acc_steps):
        # local_device = next(model.parameters()).device

        x, y = get_batch(data['train'][idx], sequence_length, batch_size, device=extra_args.device)
        with type_ctx:
            with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=1,
                                                                       gradient_accumulation_steps=1):
                outputs = model(x, targets=y)

        loss = outputs["loss"]
        loss.backward()

    grads = []
    learning_rate = None
    for client in selected_clients:
        model, _, scheduler = clients[client]
        learning_rate = scheduler.get_last_lr()[0]
        cnt = 0
        for i, p in enumerate(model.parameters()):
            if p.requires_grad:
                try:
                    p.data -= learning_rate * (p.grad + extra_args.w_lambda * (p.data - global_model[cnt].to(p.data.device)))
                except RuntimeError as e:
                    breakpoint()
                if len(grads) <= cnt:
                    grads.append(p.grad.clone())
                else:
                    grads[cnt] += p.grad.clone().to(grads[cnt].device)
                cnt += 1

    for i in range(len(global_model)):

        global_model[i] -= learning_rate * grads[i].to(global_model[i].device) / extra_args.k

    return global_model

def __weighted_average(clients, trust_weights) -> None:
    print(type(trust_weights), np.array(trust_weights), type(np.array(trust_weights)))
    wandb.log({'Trust weights': json.dumps(np.array(trust_weights).tolist())}, commit=False)
    np_trust_weights = trust_weights.clone().cpu().numpy()
    np.fill_diagonal(np_trust_weights, np.nan)
    plt.imshow(np_trust_weights, cmap='viridis', interpolation='none')
    wandb.log({'mid_param_inner_product': wandb.Image(plt)}, commit=False)

    # old
    weights = {}
    for id, client in enumerate(clients):
        for name, param in client[0].named_parameters():
            if param.requires_grad:
                if name in weights:
                    weights[name][id] = param.data.clone()
                else:
                    weights[name] = {}
                    weights[name][id] = param.data.clone()

    for idx, client in enumerate(clients):
        model, _, _ = client

        for name, param in model.named_parameters():
            if param.requires_grad:
                val = torch.zeros_like(param)
                for i in range(len(clients)):
                    val += trust_weights[idx, i] * weights[name][i]
                param.data = val

    del weights


def __objective_function_update(clients, trust_weights) -> None:
    # print(type(trust_weights), np.array(trust_weights), type(np.array(trust_weights)))
    # wandb.log({'Trust weights': json.dumps(np.array(trust_weights).tolist())}, commit=False)
    np_trust_weights = trust_weights.clone().cpu().numpy()
    np.fill_diagonal(np_trust_weights, np.nan)
    plt.imshow(np_trust_weights, cmap='viridis', interpolation='none')
    wandb.log({'adjacency_matrix': wandb.Image(plt)}, commit=False)
    wandb.log({'ca-es': trust_weights[0, 1], 'ca-de': trust_weights[0, 2], 'ca-nl': trust_weights[0, 3]}, commit=False)
    rho = 0.001
    # old
    weights = {}
    for id, client in enumerate(clients):
        for name, param in client[0].named_parameters():
            if param.requires_grad:
                if name in weights:
                    weights[name][id] = param.data.clone()
                else:
                    weights[name] = {}
                    weights[name][id] = param.data.clone()

    for idx, client in enumerate(clients):
        model, _, _ = client

        for name, param in model.named_parameters():
            if param.requires_grad:
                val = torch.zeros_like(param)
                for i in range(len(clients)):
                    val += rho * trust_weights[idx, i] * (weights[name][idx] - weights[name][i])
                param.data -= val


def _choose_indices_upper_triangular(matrix_shape, prob):
    indices = np.triu_indices(matrix_shape, k=1)
    num_elements = len(indices[0])
    random_numbers = np.random.rand(num_elements)
    mask = random_numbers <= prob

    chosen_indices = (indices[0][mask], indices[1][mask])

    return chosen_indices


def _calculate_mid_param_models(i, j, clients, data, batch_size, sequence_length, extra_args, type_ctx, distributed_backend):

    clients_i_model, _, _  = clients[i]
    clients_j_model, _, _ = clients[j]

    # breakpoint()
    mid_param1 = type(clients_i_model).from_pretrained(extra_args.use_pretrained, extra_args)
    # mid_param1 = get_model(extra_args).to(extra_args.device)
    mid_param1 = distributed_backend.transform_model(mid_param1)

    mid_param2 = type(clients_j_model).from_pretrained(extra_args.use_pretrained, extra_args)
    # mid_param2 = get_model(extra_args).to(extra_args.device)
    mid_param2 = distributed_backend.transform_model(mid_param2)

    device = next(clients_i_model.parameters()).device
    mid_param1.load_state_dict(clients_i_model.state_dict())
    mid_param1.to(device)
    mid_param1.zero_grad()
    mid_param1.train()
    for param, client1_param, client2_param in zip(mid_param1.parameters(), clients_j_model.parameters(),
                                                   clients_j_model.parameters()):
        param.data.mul_(0.5)
        param.data.add_(client2_param.data.to(device), alpha=0.5)

    # data, target = clients[i].get_next_batch_train()
    x, y = get_batch(data['train'][i], sequence_length, batch_size, device=extra_args.device)
    with type_ctx:
        with distributed_backend.get_context_for_microstep_forward(model=mid_param1, microstep_idx=1,
                                                                   gradient_accumulation_steps=1):
            outputs = mid_param1(x, targets=y)

    loss = outputs['loss']
    loss.backward()
    # grad_i, _ = self.get_flattened_grad_and_param(mid_param1, shared_layers)

    # mid_param2 = ResNet9().to(device)
    mid_param2.load_state_dict(mid_param1.state_dict())
    mid_param2.to(device)
    mid_param2.zero_grad()
    mid_param2.train()
    # data, target = clients[j].get_next_batch_train()
    x, y = get_batch(data['train'][i], sequence_length, batch_size, device=extra_args.device)
    with type_ctx:
        with distributed_backend.get_context_for_microstep_forward(model=mid_param2, microstep_idx=1,
                                                                   gradient_accumulation_steps=1):
            outputs = mid_param2(x, targets=y)

    loss = outputs['loss']
    loss.backward()
    # grad_j, _ = self.get_flattened_grad_and_param(self.mid_param, shared_layers)

    return mid_param1, mid_param2


def _calculate_inner_product(model1, model2):
    sum = 0
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        # if name1 != name2:
        #     raise NameError(f'Should be the same: {name1} != {name2}')
        assert name1 == name2, f'Should be the same: {name1} != {name2}'
        if param1.requires_grad:
            sum += torch.sum(torch.dot(torch.flatten(param1.grad), torch.flatten(param2.grad))).detach().item()

    return sum


def __update_trust_weights_frank_wolfe(trust_weights, mid_param_grad_inner_product):

    # row_sums = torch.sum(mid_param_grad_inner_product, dim=1)
    # nomalized_updates = mid_param_grad_inner_product / row_sums[:, None]

    # trust_weights = 0.9*trust_weights + 0.1*nomalized_updates

    trust_weights += 0.05 * mid_param_grad_inner_product


    # trust_weights[trust_weights > 1] = 1
    # trust_weights[trust_weights < 0] = 0

    for i in range(trust_weights.shape[0]):
        row_sorted, indices = torch.sort(trust_weights[i], descending=True)
        partial_sum = 0
        max_ind = 0
        sum_max_ind = 0
        for j in range(len(indices)):
            partial_sum += row_sorted[j]
            if row_sorted[j] + 1 / (j + 1) * (1 - partial_sum) > 0:
                max_ind = j
                sum_max_ind = partial_sum

        tao = 1 / (max_ind + 1) * (1 - sum_max_ind)
        trust_weights[i] += tao
        trust_weights[i][trust_weights[i] < 0] = 0
    #


    # row_sums = torch.sum(trust_weights, dim=1)
    # trust_weights = trust_weights / row_sums[:, None]
    # print('trust weights:', trust_weights)

    return trust_weights


def __cobo(clients, trust_weights, data, batch_size, sequence_length, extra_args, type_ctx, distributed_backend):
    sz = len(clients)
    mid_param_grad_inner_product = torch.zeros((sz, sz),)

    # p_sample = min(1.0, 100/(self.step+1))
    p_sample = 1
    # if self.step > 2000:
    #     p_sample = 1/math.sqrt(self.step)
    chosen_indices = _choose_indices_upper_triangular(sz, p_sample)
    print('the pairs are ', chosen_indices)

    cnt = 0
    for i, j in zip(*chosen_indices):
        models = _calculate_mid_param_models(i, j, clients, data, batch_size, sequence_length, extra_args, type_ctx, distributed_backend)
        product = _calculate_inner_product(models[0], models[1])
        print(f'inner product {i} and {j} is {product}')
        cnt += 1
        mid_param_grad_inner_product[i, j] = product
        mid_param_grad_inner_product[j, i] = product

    print('inner products are calculated')

    trust_weights = __update_trust_weights_frank_wolfe(trust_weights, mid_param_grad_inner_product)
    # np_trust_weights = trust_weights.clone().cpu().numpy()
    # np.fill_diagonal(np_trust_weights, np.nan)
    # plt.imshow(np_trust_weights, cmap='viridis', interpolation='none')
    # wandb.log({'adjacency_matrix': wandb.Image(plt)}, commit=True)

    # __objective_function_update(clients, trust_weights)

    return trust_weights


def __average(clients) -> None:
    trust_weights = torch.zeros((len(clients), len(clients)))
    trust_weights = torch.fill(trust_weights, 1 / len(clients))
    __weighted_average(clients, trust_weights)


def __average_static(clients, dataset) -> None:
    trust_weights = torch.zeros((len(clients), len(clients)))
    for id_1 in range(len(clients)):
        for id_2 in range(len(clients)):
            if id_2 <= id_1:
                score = 0
                if dataset == 'agnews_mixed':
                    if id_1 == id_2:
                        score = 3 / 4
                    elif (id_2 // 2) == (id_1 // 2):
                        score = 1 / 4
                if dataset == 'agnews_specific':
                    if (id_2 // 2) == (id_1 // 2):
                        score = 4 / len(clients)
                if dataset == 'three_multi_specific':
                    if (id_2 % 3) == (id_1 % 3):
                        score = 3 / len(clients)
                if dataset == 'three_multi_mixed':
                    if (id_1 % 3) == 0:
                        if id_2 % 3 == 0:
                            score = 5 / (len(clients) * 3)
                        elif id_2 % 3 == 1:
                            score = 3 / (len(clients) * 3)
                        else:
                            score = 1 / (len(clients) * 3)
                    elif (id_1 % 3) == 1:
                        if id_2 % 3 == 0:
                            score = 1 / (len(clients) * 3)
                        elif id_2 % 3 == 1:
                            score = 5 / (len(clients) * 3)
                        else:
                            score = 3 / (len(clients) * 3)
                    else:
                        if id_2 % 3 == 0:
                            score = 3 / (len(clients) * 3)
                        elif id_2 % 3 == 1:
                            score = 1 / (len(clients) * 3)
                        else:
                            score = 5 / (len(clients) * 3)
                if dataset == 'github_wiki_specific':
                    if (id_2 % 2) == (id_1 % 2):
                        score = 2 / len(clients)
                if dataset == 'github_wiki_mixed':
                    if (id_1 % 2) == (id_2 % 2):
                        score = 3 / (len(clients) * 2)
                    else:
                        score = 1 / (len(clients) * 2)

                trust_weights[id_1, id_2] = score
                trust_weights[id_2, id_1] = score
    __weighted_average(clients, trust_weights)


def similarity_weights(client1, client2,
                       similarity: Callable[[Tensor, Tensor], Tensor] = F.cosine_similarity):
    score = 0
    total_size = 0
    for (name1, param1), (name2, param2) in zip(client1.named_parameters(), client2.named_parameters()):
        if name1 != name2:
            raise NameError(f'Should be the same: {name1} != {name2}')
        if param1.requires_grad:
            sim = similarity(param1, param2)
            total_size += sim.size(0)
            score += torch.sum(sim).detach().item()

    return score / total_size


def clients_similarity(clients,
                       sim_func,
                       similarity: Callable[[Tensor, Tensor], Tensor] = F.cosine_similarity) -> Tensor:
    trust_weight = torch.zeros((len(clients), len(clients)))
    for idx1, (model1, _, _) in enumerate(clients):
        for idx2, (model2, _, _) in enumerate(clients):
            if idx2 <= idx1:
                score = sim_func(model1, model2, similarity)
                trust_weight[idx1, idx2] = score
                trust_weight[idx2, idx1] = score
    return trust_weight


def __average_dynamic(clients) -> None:
    trust_weights = clients_similarity(clients, similarity_weights)
    trust_weights = F.softmax(trust_weights, dim=1)
    __weighted_average(clients, trust_weights)


def __average_dynamic_threshold(clients) -> None:
    trust_weights = clients_similarity(clients, similarity_weights)
    topk_values, topk_indices = torch.topk(trust_weights, 2, dim=-1)
    trust_weights[trust_weights <= 0.5] = -1e9
    trust_weights.scatter_(-1, topk_indices, topk_values)
    trust_weights = F.softmax(trust_weights, dim=1)
    __weighted_average(clients, trust_weights)


def __average_dynamic_top_k(clients, k) -> None:
    trust_weights = clients_similarity(clients, similarity_weights)
    topk_values, topk_indices = torch.topk(trust_weights, k, dim=-1)
    mask = torch.zeros_like(trust_weights)
    mask = torch.fill(mask, -1e9)
    mask.scatter_(-1, topk_indices, topk_values)
    trust_weights = F.softmax(mask, dim=1)
    __weighted_average(clients, trust_weights)


def __average_dynamic_ref(clients, trust_weights) -> None:
    trust_weights = F.softmax(trust_weights, dim=1)
    __weighted_average(clients, trust_weights)


def __average_dynamic_threshold_ref(clients, trust_weights) -> None:
    topk_values, topk_indices = torch.topk(trust_weights, 2, dim=-1)
    trust_weights[trust_weights <= -30] = -1e9
    trust_weights.scatter_(-1, topk_indices, topk_values)
    trust_weights = F.softmax(trust_weights, dim=1)
    __weighted_average(clients, trust_weights)


def __average_dynamic_top_k_ref(clients, trust_weights, k) -> None:
    topk_values, topk_indices = torch.topk(trust_weights, k, dim=-1)
    mask = torch.zeros_like(trust_weights)
    mask = torch.fill(mask, -1e9)
    mask.scatter_(-1, topk_indices, topk_values)
    trust_weights = F.softmax(mask, dim=1)
    __weighted_average(clients, trust_weights)


def __average_dynamic_token(clients, trust_weights) -> None:
    trust_weights = F.softmax(trust_weights, dim=1)
    __weighted_average(clients, trust_weights)


def __average_dynamic_threshold_token(clients, trust_weights) -> None:
    topk_values, topk_indices = torch.topk(trust_weights, 2, dim=-1)
    trust_weights[trust_weights <= -50] = -1e9
    trust_weights.scatter_(-1, topk_indices, topk_values)
    trust_weights = F.softmax(trust_weights, dim=1)
    __weighted_average(clients, trust_weights)


def __average_dynamic_top_k_token(clients, trust_weights, k) -> None:
    topk_values, topk_indices = torch.topk(trust_weights, k, dim=-1)
    mask = torch.zeros_like(trust_weights)
    mask = torch.fill(mask, -1e9)
    mask.scatter_(-1, topk_indices, topk_values)
    trust_weights = F.softmax(mask, dim=1)
    __weighted_average(clients, trust_weights)
