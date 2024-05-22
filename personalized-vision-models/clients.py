from data.datasets import Dataset
import torch
import numpy as np
import itertools
class ClientGroup:
    def __init__(self, clients_num, model, batch_size_train, batch_size_test, dataset, task_type, num_gpus=1,
                 num_previous_agents=0, args=None, ddp=False):
        """
        builds a group of clients with the same dataset
        clients_num: number of clients in the group
        model: the model that the clients will use
        dataset: the dataset that the clients will use
        """

        self.clients_num = clients_num
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.dataset = dataset
        # print('dataset is:', self.dataset)
        # print('train loader type is:', type(self.dataset.train_loader))
        # print('number of batches in client group:', len(self.dataset.train_loader[0]))
        self.clients = []
        self.model = model
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.task_type = task_type
        self.args = args
        self.ddp = ddp
        self.gpus = num_gpus
        for i in range(self.clients_num):
            self.clients.append(Client(self, model, i, num_previous_agents + i))



class Client:
    def __init__(self, group, model, local_index, general_index):
        """
        initializing a client
        group: the group that the client belongs to
        model: the model that the client uses
        """
        self.neighbor_models = list()
        self.neighbor_inds = list()
        self.group = group
        self.model = model(self.group.args, client=self).to("cuda:" + str(general_index % self.group.gpus))
        # if self.group.ddp:
        #     self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[int(os.environ['LOCAL_RANK'])])

        group_dataset = self.group.dataset
        print(group_dataset.flipped)
        self.dataset = Dataset(group_dataset.name, group_dataset.batch_size_train,
                               group_dataset.batch_size_test, flipped=group_dataset.flipped, seed=group_dataset.seed,
                               lang=group_dataset.lang, sequence_length=group_dataset.sequence_length)

        if isinstance(self.group.dataset.train_loader, list):
            self.dataset.train_loader = self.group.dataset.train_loader[local_index]
        self.dataset.test_loader = self.group.dataset.test_loader

        # self.dataset.train_loader.dataset.targets %= 10
        # self.dataset.test_loader.dataset.targets %= 10

        self.train_loader_cycle = itertools.cycle(self.dataset.train_loader)
        self.test_loader_cycle = itertools.cycle(self.dataset.test_loader)

    def _generate_batches(self, loader, batch_size):
        seq_length = self.dataset.sequence_length

        while True:
            ix = torch.randint(len(loader) - seq_length, (batch_size,))
            x = torch.stack([torch.from_numpy((loader[i:i + seq_length]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((loader[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
            if "cuda" in torch.device(self.group.device).type:
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x = x.pin_memory().to(self.group.device, non_blocking=True)
                y = y.pin_memory().to(self.group.device, non_blocking=True)
            yield x, y

    def get_next_batch_train(self):
        if self.group.task_type == 'vision':
            # print('haha')
            # return next(self.train_loader_cycle)
            for i in range(1000):
                for data in self.dataset.train_loader:
            #         # breakpoint()
                    yield data
            # breakpoint()
            # return self.iter_train_loader
        else:
            #TODO: it was yield before. Not compatible with return yet (also for test loader)
            for x, y in self._generate_batches(self.dataset.train_loader, self.dataset.batch_size_train):
                yield x, y

    def get_next_batch_test(self):
        if self.group.task_type == 'vision':
            return next(self.test_loader_cycle)
        else:
            for x, y in self._generate_batches(self.dataset.test_loader, self.dataset.batch_size_test):
                return x, y
