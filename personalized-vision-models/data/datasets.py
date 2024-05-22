import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as tt
import sys
# print('path:', sys.path)
import copy
import os
class Dataset:

    def __init__(self, name, batch_size_train, batch_size_test, lang='en', sequence_length=512, flipped=False, seed=0):
        """
        initializing a dataset
        name: name of the dataset (we call its function with this name): fashion_MNIST, MNIST
        flipped: if the labels should be flipped
        seed: we shift the labels by this seed, e.g. if seed = 1 then all the labels increase by 1 except for 9 which becomes 0
        """
        self.original_dataset = None
        self.name = name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.flipped = flipped
        self.seed = seed
        print('in dataset constructor, the language is:', lang)
        self.lang = lang
        self.sequence_length = sequence_length
        dataset_to_use = getattr(self, self.name)
        self.train_loader, self.val_loader, self.test_loader = dataset_to_use()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    @staticmethod
    def flip_labels(train_loader, test_loader, seed, num_labels=100):
        """flips the labels of the dataset by using seed"""
        # print('type is: ', type(train_loader.targets))
        if isinstance(train_loader, list):
            print('flipping labels. number of samples in underlying dataset is ', len(train_loader[0].dataset))
            breakpoint()
            train_loader[0].dataset.targets = (train_loader[0].dataset.targets + seed) % num_labels
            # for i in range(len(train_loader)):
            #     for _, labels in train_loader[i]:
            #         labels.add_(seed).remainder_(num_labels)
        else:
            print('labels are changed')
            train_loader.targets = (train_loader.targets + seed) % num_labels

        test_loader.dataset.targets = (test_loader.dataset.targets + seed) % num_labels
        # print(train_loader.dataset.targets)
        return train_loader, test_loader


    @staticmethod
    def partition_train(dataset, num_clients, indices=None):
        """partitions the dataset to num_clients partitions"""
        train_data = dataset.train_loader.dataset
        train_size = len(train_data)
        print('len train data is:', train_size)
        if indices is None:
            indices = random_split(range(train_size), [int(train_size/num_clients) for _ in range(num_clients)])
        partitions = []
        for i in range(num_clients):
            partitions.append(DataLoader(Subset(train_data, indices[i]), batch_size=dataset.batch_size_train,
                                         shuffle=True, pin_memory=True))
            print('indices are:', indices[i])

        dataset.train_loader = partitions
        # breakpoint()
        return dataset

    # @staticmethod
    # def clone_trainset(dataset):
    #     if isinstance(dataset.train_loader, list):
    #         print('trainloader is actually a list')
    #         for i in range(len(dataset.train_loader)):
    #             dataset.train_loader[i] = DataLoader(dataset=dataset.train_loader[i].dataset,
    #                                                  batch_size=dataset.batch_size_train, shuffle=False)
    #     else:
    #         dataset.train_loader = DataLoader(dataset=dataset.train_loader.dataset,
    #                                                  batch_size=dataset.batch_size_train, shuffle=True)
    #     print('after cloning, dataset is:', dataset)
    #     return dataset


    def split_train(self, train_data):
        #TODO: fix validation!
        # val_size = int(0.2*len(train_data))
        # # val_size = 1
        # train_size = len(train_data) - val_size
        # train_data, val_data = random_split(train_data, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=self.batch_size_train)
        # , pin_memory = True, num_workers = 4
        val_loader = DataLoader(train_data, batch_size=self.batch_size_test, shuffle=True)
        return train_loader, val_loader


    def fashion_MNIST(self):
        """loads fashion MNIST"""
        train_data = torchvision.datasets.FashionMNIST(root=os.getcwd(),train=True, download=True,
                                              transform=tt.Compose([
                                                  tt.ToTensor(),
                                                  tt.Normalize(
                                                      (0.5,), (0.5,))
                                              ]))


        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root=os.getcwd(),train=False, download=True,
                                              transform=tt.Compose([
                                                  tt.ToTensor(),
                                                  tt.Normalize(
                                                      (0.5,), (0.5,))
                                              ])),
            batch_size=self.batch_size_test, shuffle=True)

        # print(type(train_data.dataset), type(test_loader.dataset))

        if self.flipped:
            train_loader, test_loader = Dataset.flip_labels(train_data, test_loader, self.seed, num_labels=10)

        train_loader, val_loader = self.split_train(train_data)

        return train_loader, val_loader, test_loader


    def MNIST(self):
        """loads MNIST"""
        train_data = torchvision.datasets.MNIST(root=os.getcwd(), train=True, download=True,
                                              transform=tt.Compose([
                                                  tt.ToTensor(),
                                                  tt.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ]))
        test_loader = DataLoader(
            torchvision.datasets.MNIST(root=os.getcwd(), train=False, download=True,
                                              transform=tt.Compose([
                                                  tt.ToTensor(),
                                                  tt.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ])),
            batch_size= self.batch_size_test, shuffle=True)

        if self.flipped:
            train_loader, test_loader = Dataset.flip_labels(train_data, test_loader, self.seed, num_labels=10)

        train_loader, val_loader = self.split_train(train_data)


        return train_loader, val_loader, test_loader


    def Cifar100(self):
        """loads Cifar100"""
        train_data = torchvision.datasets.CIFAR100(root=os.getcwd(), train=True, download=True,
                                              transform=tt.Compose([
                                                  tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                                                  tt.RandomHorizontalFlip(),
                                                  tt.ToTensor(),
                                                  tt.Normalize(
                                                      (0.5,), (0.5,))
                                              ]))
        train_data.targets = torch.tensor(train_data.targets)

        test_loader = DataLoader(
            torchvision.datasets.CIFAR100(root=os.getcwd(), train=False, download=True,
                                              transform=tt.Compose([
                                                  tt.ToTensor(),
                                                  tt.Normalize(
                                                      (0.5,), (0.5,))
                                              ])),
            batch_size= self.batch_size_test, shuffle=True)
        test_loader.dataset.targets = torch.tensor(test_loader.dataset.targets)
        if self.flipped:
            train_loader, test_loader = Dataset.flip_labels(train_data, test_loader, self.seed, num_labels=100)

        train_loader, val_loader = self.split_train(train_data)
        # print('1 size:',len(train_loader.dataset))
        # val_loader = DataLoader(train_data, batch_size=self.batch_size_test, shuffle=True)
        # train_loader = DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True)
        print('2 size:', len(train_loader.dataset))

        return train_loader, val_loader, test_loader

