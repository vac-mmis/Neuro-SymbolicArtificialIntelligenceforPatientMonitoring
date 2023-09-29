import copy
from itertools import zip_longest
from typing import Callable

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as torchDataset

from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, Constant

class MNIST:
    
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train = MNISTTrain(transform)
        self.test = MNISTest(transform)

class MNISTTrain(torchDataset):
    
    def __init__(self, transform: Callable):
        self.data = torchvision.datasets.MNIST(root='data', train=True, download=True,
                                               transform=transform)
        self.data_indices = list(range(len(self.data)))

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx) -> tuple:
        mnist_index = self.data_indices[idx]
        data = self.data[mnist_index]
        return data

    def drop_samples(self, percentage: float):
        train_data = pd.DataFrame(self.data.targets, columns=['target'])
        train_data, _ = train_test_split(train_data, train_size=1 - percentage, random_state=42)
        self.data_indices = train_data.index.tolist()

class MNISTest(torchDataset):
    
    def __init__(self, transform: Callable):
        self.data = torchvision.datasets.MNIST(root='data', train=False, download=True,
                                               transform=transform)
        self.data_indices = list(range(len(self.data)))

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx) -> tuple:
        mnist_index = self.data_indices[idx]
        data = self.data[mnist_index]
        return data

    def drop_samples(self, percentage: float):
        train_data = pd.DataFrame(self.data.targets, columns=['target'])
        train_data, _ = train_test_split(train_data, train_size=1 - percentage, random_state=42)
        self.data_indices = train_data.index.tolist()





class TorchDataset:
    
    def __init__(self, train_path: str, test_path: str):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_set = TorchTrain(transform, train_path)
        self.test_set = TorchTest(transform, test_path)



class TorchTrain(torchDataset):
    def __init__(self, transform: Callable, path: str):
        self.data = torchvision.datasets.MNIST(root='data', train=True, download=True,
                                               transform=transform)
        self.metadata = pd.read_csv(path)
        self.target_transform = lambda x: x + 9

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx) -> tuple:
        idx1 = self.metadata.iloc[idx]['x1']
        idx2 = self.metadata.iloc[idx]['x2']
        image1, image2 = copy.deepcopy(self.data[idx1][0]), copy.deepcopy(self.data[idx2][0])
        label = self.metadata.iloc[idx]['class_label']
        label = self.target_transform(label)
        return (image1, image2), label

class TorchTest(torchDataset):
    def __init__(self, transform: Callable, path:str):
        self.data = torchvision.datasets.MNIST(root='data', train=False, download=True,
                                               transform=transform)
        self.metadata = pd.read_csv(path)
        # function to transform the labels
        self.target_transform = lambda x: x + 9

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx) -> tuple:
        idx1 = self.metadata.iloc[idx]['x1']
        idx2 = self.metadata.iloc[idx]['x2']
        image1, image2 = copy.deepcopy(self.data[idx1][0]), copy.deepcopy(self.data[idx2][0])
        label = self.metadata.iloc[idx]['class_label']
        label = self.target_transform(label)
        return (image1, image2), label
    

class DPLDataset:
    
    def __init__(self, train_path: str, test_path:str, function_name: str) -> None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_set = DPLTrain(transform, train_path, function_name)
        self.test_set = DPLTest(transform, test_path, function_name)

    
class DPLTrain(Dataset):
    def __init__(self, transform: Callable, path: str, function_name: str):
        self.data = torchvision.datasets.MNIST(root='data', train=True, download=True,
                                               transform=transform)
        self.metadata = pd.read_csv(path)
        self.function_name = function_name

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """ Get a specific item (pixel values of image and label) of the dataset.

        :param idx: The index of the item
        :return: Tuple with pixel data and the label
        """
        # get pixel matrix and transform it into 2d (8x8) tensor
        if type(idx) is tuple:
            idx = int(idx[0])
        image = self.data[idx][0]

        return image  
    
    def to_query(self, idx: int) -> Query:
        """Generate queries"""
        idx1: int = self.metadata.iloc[idx]['x1']
        idx2: int = self.metadata.iloc[idx]['x2']
        expected_result = self.data[idx2][1] - self.data[idx1][1]
        tensor_idx = [idx1, idx2]

        subs = dict()
        var_names = []
        for idx in tensor_idx:
            t = Term(f"image{idx}")
            subs[t] = Term(
                "tensor",
                Term(
                    'train',
                    Constant(idx),
                ),
            )
            var_names.append(t)

        # Build query
        return Query(
            Term(
                self.function_name,
                *(e for e in var_names),
                Constant(expected_result),
            ),
            subs,
        )
    


class DPLTest(Dataset):
    def __init__(self, transform: Callable, path:str, function_name: str):
        self.data = torchvision.datasets.MNIST(root='data', train=False, download=True,
                                               transform=transform)
        self.metadata = pd.read_csv(path)
        self.function_name = function_name

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """ Get a specific item (pixel values of image and label) of the dataset.

        :param idx: The index of the item
        :return: Tuple with pixel data and the label
        """
        # get pixel matrix and transform it into 2d (8x8) tensor
        if type(idx) is tuple:
            idx = int(idx[0])
        image = self.data[idx][0]

        return image
    
    def to_query(self, idx: int) -> Query:
        """Generate queries"""
        idx1: int = self.metadata.iloc[idx]['x1']
        idx2: int = self.metadata.iloc[idx]['x2']
        expected_result = self.data[idx2][1] - self.data[idx1][1]
        tensor_idx = [idx1, idx2]

        subs = dict()
        var_names = []
        for idx in tensor_idx:
            t = Term(f"image{idx}")
            subs[t] = Term(
                "tensor",
                Term(
                    "test",
                    Constant(idx),
                ),
            )
            var_names.append(t)

        # Build query
        return Query(
            Term(
                self.function_name,
                *(e for e in var_names),
                Constant(expected_result),
            ),
            subs,
        )