# import subprocess
# command = "pip install transformers"
# subprocess.call(command, shell=True)
# command = "pip install tiktoken"
# subprocess.call(command, shell=True)

import wandb
from clients import ClientGroup
from models.smallnet import SmallNet
from train import Train
from data.datasets import Dataset
import argparse
from grouping import Grouping
from models.resnet import ResNet9
from torch.utils.data import random_split
import torch
import os
from torch.distributed import init_process_group
import numpy as np
from torch.utils.data import DataLoader, Subset
# from line_profiler import LineProfiler

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs_train', type=int, default=1)
    parser.add_argument('--bs_test', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_lambda', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=50)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--workers', type=int, nargs='+', default=[2, -1])
    parser.add_argument('--shared_layers', type=int, nargs='+',
                        default=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--train_method', type=str, default='cobo')
    parser.add_argument('--eval_method', type=str, default='shared_model_evaluation')
    parser.add_argument('--grouping_method', type=str, default='cobo')
    parser.add_argument('--known_grouping', type=bool, default=True)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--wandb_key', type=str, default='772bb501917cdea510fc4f46258987769788e1c3')
    parser.add_argument('--dataset', type=str, default='fashion_MNIST')
    parser.add_argument('--partitioned', type=bool, default=True)
    parser.add_argument('--identical_partition', type=bool, default=True)
    parser.add_argument('--wandb_id', type=str, default=wandb.util.generate_id())
    parser.add_argument('--acc_steps', type=int, default=1)
    parser.add_argument('--langs', type=str, nargs='+', default=['en'])
    parser.add_argument('--run_name', type=str, default='untitled run')
    parser.add_argument('--task_type', type=str, default='vision')
    parser.add_argument('--ditto_k', type=int, default=2)
    parser.add_argument('--ditto_lambda', type=float, default=1)
    parser.add_argument('--fc_percentile', type=float, default=0.2)
    # LLM config arguments
    parser.add_argument('--sequence_length', type=int, default=256)
    parser.add_argument('--use_pretrained', default="none", type=str)
    # 'none', 'gpt-2' or a path to the pretrained model
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--n_head', default=12, type=int)
    parser.add_argument('--n_layer', default=12, type=int)  # depths in att + ff blocks
    parser.add_argument('--n_embd', default=768, type=int)  # embedding size / hidden size ...
    parser.add_argument('--dtype', default=torch.bfloat16, type=torch.dtype)
    parser.add_argument('--bias', default=False, type=bool)
    parser.add_argument('--no_compile', action='store_true')
    parser.add_argument('--vocab_size', default=52000, type=int)
    parser.add_argument('--scheduler', default='none', choices=['linear', 'cos', 'none'])
    parser.add_argument('--warmup_percent', default=0.02, type=float)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cluster_partition', default=False, type=bool)
    parser.add_argument('--ifca_k', type=int, default=2)
    parser.add_argument('--ifca_m', type=int, default=2)
    args = parser.parse_args()

    batch_size_train = args.bs_train
    batch_size_test = args.bs_test
    learning_rate = args.lr
    lr_lambda = args.lr_lambda
    iterations = args.iterations
    workers = args.workers
    shared_layers = args.shared_layers
    wandb_run = args.wandb
    train_method = args.train_method
    eval_method = args.eval_method
    known_grouping = args.known_grouping
    grouping_method = args.grouping_method
    alpha = args.alpha
    rho = args.rho
    wandb_key = args.wandb_key
    dataset_name = args.dataset
    partitioned = args.partitioned
    identical_partition = args.identical_partition
    wandb_id = args.wandb_id
    acc_steps = args.acc_steps
    langs = args.langs
    run_name = args.run_name
    task_type = args.task_type
    gpus = args.gpus
    cluster_partition = args.cluster_partition
    # LLM config arguments
    sequence_length = args.sequence_length
    # use_pretrained = args.use_pretrained
    # dropout = args.dropout
    # n_head = args.n_head
    # n_layers = args.n_layers
    # n_embed = args.n_embed
    print('identical partition', identical_partition, 'cluster partition', cluster_partition)
    print('languages are:', langs)


    shared_layers = [0 for i in range(62)]
    for i in range(62):
        shared_layers[i] = 1

    split_set = [0.5, 0.5]

    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend='nccl')  # Should I change it?
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}' #ddp_local_rank
        print('ddp_rank', ddp_rank, '\nddp_local_rank', ddp_local_rank)
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0

    print('wandb ID is:', wandb_id)

    if wandb_run and master_process:
        if wandb_key is not None:
            wandb.login(key=wandb_key)

        wandb.init(
            project="personalization",
            name=run_name,
            id=wandb_id,
            config={
                "learning_rate": learning_rate,
                "iterations": iterations,
                "batch_size": batch_size_train,
                "workers": workers,
                "shared_layers": shared_layers,
                "alpha": alpha,
                "rho": rho,
                "ditto_k": args.ditto_k,
                "ditto_lambda": args.ditto_lambda,
                "fc_percentile": args.fc_percentile,
                "split_set": split_set
            }
        )

    worker_groups = list()
    used_net = SmallNet
    if dataset_name == 'Cifar100':
        used_net = ResNet9

    cnt_clients = 0

    print('used net: ', used_net)

    # try:
    #     checkpoint = torch.load(dir_checkpoint)
    # except FileNotFoundError:
    #     print('checkpoint file does not found')

    identical_dataset = Dataset(dataset_name, batch_size_train, batch_size_test)
    partitioning = None
    # if identical_partition:
    #     if wandb.run is not None and wandb.run.resumed:
    #         partitioning = checkpoint["partitioning"]
    #     else:
    #         train_size = len(identical_dataset.train_loader.dataset)
    #         partitioning = random_split(range(train_size), [int(train_size * x) for x in split_set])

    for i in range(len(workers)):

        if task_type == "language":
            print('in run.py, language is:', langs[i])
            dataset = Dataset(dataset_name, batch_size_train, batch_size_test, lang="ca", sequence_length=sequence_length)
        else:
            flipped = workers[i] < 0
            dataset = Dataset(dataset_name, batch_size_train, batch_size_test, flipped=flipped, seed=i * 11)
            if partitioned:
                print('dataset is partitioned')
                if identical_partition:
                    print('partitioning is identical', dataset)
                    dataset = Dataset.partition_train(dataset, abs(workers[i]), indices=partitioning)
                    print('number of batches:', len(dataset.train_loader[0]))
                elif cluster_partition:
                    train_targets = dataset.train_loader.dataset.targets.numpy()
                    indices = np.where((train_targets >= i*10) & (train_targets < (i+1)*10))[0]
                    indices_size = len(indices)
                    partition_size = indices_size//abs(workers[i])
                    partition_remainder = indices_size % abs(workers[i])
                    dataset.train_loader.dataset.targets[indices] = dataset.train_loader.dataset.targets[indices] % 10

                    partitioning = random_split(indices, [partition_size + (x < partition_remainder) for x in range(abs(workers[i]))])
                    dataset = Dataset.partition_train(dataset, abs(workers[i]), indices=partitioning)

                    test_targets = dataset.test_loader.dataset.targets.numpy()
                    test_indices = np.where((test_targets >= i * 10) & (test_targets < (i + 1) * 10))[0]
                    dataset.test_loader.dataset.targets[test_indices] = dataset.test_loader.dataset.targets[test_indices] % 10
                    dataset.test_loader = DataLoader(Subset(dataset.test_loader.dataset, test_indices),
                                                     batch_size=dataset.batch_size_test, shuffle=True)

                else:
                    print('dataset is not partitioned')
                    dataset = Dataset.partition_train(dataset, abs(workers[0]))

        # breakpoint()
        worker_groups.append(ClientGroup(abs(workers[i]), used_net, batch_size_train, batch_size_test, num_gpus=gpus,
                                         num_previous_agents=cnt_clients, dataset=dataset, task_type=task_type, args=args))
        cnt_clients += abs(workers[i])

    grouping = Grouping(cnt_clients, learning_rate, alpha=alpha, rho=rho)
    starting_iteration = 0

    # breakpoint()

    # if wandb.run is not None and wandb.run.resumed:
    #     print('Resuming the training')
    #     # wandb.restore("last.pt")
    #     for i in range(len(workers)):
    #         for j in range(abs(workers[i])):
    #             worker_groups[i].clients[j].model.load_state_dict(checkpoint['models'][i][j])
    #
    #             worker_groups[i].clients[j].model.previous_momentum = checkpoint['momentum'][i][j]
    #
    #     # alpha = checkpoint['alpha']
    #     grouping = Grouping(cnt_clients, learning_rate, alpha=alpha, rho=rho, w_adjacency=checkpoint['w_adjacency'], ddp=ddp)
    #     # starting_iteration = checkpoint["starting_iteration"]
    #     learning_rate = checkpoint["learning_rate"]


    #   TODO: gradient clipping, weight decay

    train = Train(worker_groups, learning_rate, (train_method != "cobo"), master_process, shared_layers=shared_layers, grouping=grouping, config=args)

    # breakpoint()
    optim = train.get_optim(train_method)
    print(optim)
    scheduler = None
    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            print('args.lr is', args.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optim, max_lr=args.lr, total_steps=iterations,
                                                            pct_start=args.warmup_percent,
                                                            anneal_strategy=args.scheduler,
                                                            cycle_momentum=False, div_factor=1e2, final_div_factor=.05)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    # profiler = LineProfiler()
    # profiler.add_function(train.train)


    # iterations
    train.train(optim, getattr(train, eval_method), iterations, grouping_method=getattr(grouping, grouping_method),
                lr_scheduler=scheduler, start_iteration_number=starting_iteration, partitioning=partitioning, run_id=wandb_id)

    # profiler.print_stats()
