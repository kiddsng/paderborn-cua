# import modules and libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import numpy as np
import os


def to_np(x):
    return x.detach().cpu().numpy()

### datasets ###

class LoadDataset(Dataset):
    def __init__(self, dataset):
        super(LoadDataset, self).__init__()

        X = dataset['samples']
        y = dataset['labels']

        if len(X.shape) < 3: # change shape of tensor from ([8184, 5120]) to ([8184, 1, 5120])
            X = X.unsqueeze(1)
        
        self.data, self.labels = X, y
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return(len(self.data))

def generate_dataloaders(path, domain, batch_size):
    train_dataset = torch.load(os.path.join(path, 'train_' + domain + '.pt'))
    val_dataset = torch.load(os.path.join(path, 'val_' + domain + '.pt'))
    test_dataset = torch.load(os.path.join(path, 'test_' + domain + '.pt'))

    train_dataset = LoadDataset(train_dataset)
    val_dataset = LoadDataset(val_dataset)
    test_dataset = LoadDataset(test_dataset)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader

class ReplayDataset(Dataset):
    def __init__(self, train_dataset):
        # generate replays from source data
        tmp = {'x': [], 'y': []}
        for data in train_dataset:
            tmp['x'] += [data[0].numpy()]
            tmp['y'] += [data[1]]
        tmp['x'] = np.array(tmp['x'])
        tmp['y'] = np.array(tmp['y'])

        self.replay_dataset = SimpleDataset(np.array(tmp['x']), np.array(tmp['y']))

    @staticmethod
    def rand_sample(dataset):
        n = len(dataset)
        i = np.random.randint(n)
        return dataset[i]

    def __getitem__(self, index):
        x_rpy, y_rpy = self.rand_sample(self.replay_dataset)
        return x_rpy, y_rpy

    def __len__(self):
        return len(self.replay_dataset)

def generate_replay_dataloader(train_dataset, batch_size):
    replay_dataset = ReplayDataset(train_dataset=train_dataset)   
    replay_dataloader = DataLoader(dataset=replay_dataset, batch_size=batch_size)
    return replay_dataloader

# load train_data


class train_dataset(Dataset):
    def __init__(self, domain):
        data = torch.load('data/train/Paderborn_FD/train_' + domain + '.pt')
        self.data, self.labels = data['samples'], data['labels']

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

# load val_data


class val_dataset(Dataset):
    def __init__(self, domain):
        data = torch.load('data/val/Paderborn_FD/val_' + domain + '.pt')
        self.data, self.labels = data['samples'], data['labels']

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

# load test_data


class test_dataset(Dataset):
    def __init__(self, domain):
        data = torch.load('data/test/Paderborn_FD/test_' + domain + '.pt')
        self.data, self.labels = data['samples'], data['labels']

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# create replay data


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def update(self, data):
        x, y, = data
        self.x = np.concatenate([self.x, x], 0)
        self.y = np.concatenate([self.y, y], 0)
        return self

# CDA/CUA for Paderborn dataset


class DatasetCUA(Dataset):
    def __init__(self, src_domain, tgt_domains):
        self.src_train_data = train_dataset(domain=src_domain)
        # self.src_val_data = val_dataset(domain=src_domain)
        # self.src_test_data = test_dataset(domain=src_domain)

        self.tgt_train_data = []
        # self.tgt_val_data = []
        # self.tgt_test_data = []

        for domain in tgt_domains:
            self.tgt_train_data.append(train_dataset(domain=domain))
            # self.tgt_val_data.append(val_dataset(domain=domain))
            # self.tgt_test_data.append(test_dataset(domain=domain))

        # generate replays from source data
        tmp = {'x': [], 'y': []}
        for data in self.src_train_data:
            tmp['x'] += [data[0].numpy()]
            tmp['y'] += [data[1]]
        tmp['x'] = np.array(tmp['x'])
        tmp['y'] = np.array(tmp['y'])

        self.replay_data = SimpleDataset(
            np.array(tmp['x']), np.array(tmp['y']))
        self.phase = 0

    def set_phase(self, p):
        self.phase = p

    @staticmethod
    def rand_sample(data):
        n = len(data)
        i = np.random.randint(n)
        return data[i]

    def __getitem__(self, index):
        x_src, y_src = self.src_train_data[index]
        x_tgt, y_tgt = self.tgt_train_data[self.phase][index]
        x_rpy, y_rpy = self.rand_sample(self.replay_data)
        return x_src, y_src, x_tgt, y_tgt, x_rpy, y_rpy

    def __len__(self):
        return len(self.src_train_data)
