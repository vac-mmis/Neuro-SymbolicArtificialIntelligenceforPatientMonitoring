import copy
from itertools import zip_longest
from typing import Callable

import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class MNISTTrain(Dataset):
    def __init__(self, transform: Callable):
        self.data = torchvision.datasets.MNIST(root='data', train=True, download=True,
                                               transform=transform)
        self.data_indices = list(range(len(self.data)))

    def __len__(self):
        return len(self.data_indices)

    """def __getitem__(self, idx) -> tuple:
        mnist_index = self.data_indices[idx]
        data = self.data[mnist_index]
        return data"""

    def drop_samples(self, percentage: float):
        train_data = pd.DataFrame(self.data.targets, columns=['target'])
        train_data, _ = train_test_split(train_data, train_size=1 - percentage)
        self.data_indices = train_data.index.tolist()

    def print_data_distribution(self):
        elements, counts = self.data.targets.unique(return_counts=True)

        for element, count in zip(elements, counts):
            print(str(element)+": "+str(count))

class MNISTest(Dataset):
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
        train_data, _ = train_test_split(train_data, train_size=1 - percentage)
        self.data_indices = train_data.index.tolist()

    def print_data_distribution(self):
        elements, counts = self.data.targets.unique(return_counts=True)

        for element, count in zip(elements, counts):
            print(str(element)+": "+str(count))


class MNISTDiffTrain(Dataset):

    def __init__(self, mnist_data: MNISTTrain):
        self.mnist_data = mnist_data
        self.indicies = [(x, x+1) for x in mnist_data.data_indices[:-1]]
        self.metadata = pd.DataFrame(self.indicies, columns=['x1', 'x2'])

        targets = np.array(self.mnist_data.data.targets)
        x1_idx = self.metadata['x1'].values
        x2_idx = self.metadata['x2'].values
        self.metadata['x1_label'] = targets[x1_idx]
        self.metadata['x2_label'] = targets[x2_idx]
        self.metadata['class_label'] = self.metadata['x2_label']-self.metadata['x1_label']

        self.__ometadata = self.metadata.copy(deep=True)

    def print_data_statistics(self) -> None:
        print(self.metadata['class_label'].value_counts())

    def print_class_statistics(self, class_label: int) -> None:
        filtered_data = self.metadata[self.metadata['class_label'] == class_label]
        print(filtered_data.groupby(['x1_label', 'x2_label']).size().reset_index(name='count'))
    
    def print_MNIST_statistics(self) -> None:
        x1 = self.metadata['x1_label'].value_counts().sort_index()
        x2 = self.metadata['x2_label'].value_counts().sort_index()
        combined_counts = x1.add(x2, fill_value=0).astype(int)

        print(combined_counts)

    def set_num_class_samples(self, num_combinations: int) -> None:
        filtered_df = pd.DataFrame(columns=self.metadata.columns)
        for _, group in self.metadata.groupby('class_label'):
            combinations = list(zip(group['x1_label'], group['x2_label']))
            combinations = list(set(combinations))
            selected_combinations = combinations[:num_combinations]
            x1_values, x2_values = zip(*selected_combinations)
            filtered_group = group[(group['x1_label'].isin(x1_values)) & (group['x2_label'].isin(x2_values))]
            filtered_df = pd.concat([filtered_df, filtered_group]).reset_index(drop=True)
        self.metadata = filtered_df
    
    def reset_class_samples(self) -> None:
        self.metadata = self.__ometadata

    def save_metadata(self, path: str) -> None:
        self.metadata.to_csv(path)

    def __len__(self) -> int:
        return len(self.metadata)

class MNISTDiffTest(Dataset):

    def __init__(self, mnist_data: MNISTest):
        self.mnist_data = mnist_data
        self.indicies = [(x, x+1) for x in mnist_data.data_indices[:-1]]
        self.metadata = pd.DataFrame(self.indicies, columns=['x1', 'x2'])

        targets = np.array(self.mnist_data.data.targets)
        x1_idx = self.metadata['x1'].values
        x2_idx = self.metadata['x2'].values
        self.metadata['x1_label'] = targets[x1_idx]
        self.metadata['x2_label'] = targets[x2_idx]
        self.metadata['class_label'] = self.metadata['x2_label']-self.metadata['x1_label']

        self.__ometadata = self.metadata.copy(deep=True)

    def print_data_statistics(self) -> None:
        print(self.metadata['class_label'].value_counts())

    def print_class_statistics(self, class_label: int) -> None:
        filtered_data = self.metadata[self.metadata['class_label'] == class_label]
        print(filtered_data.groupby(['x1_label', 'x2_label']).size().reset_index(name='count'))
    
    def print_MNIST_statistics(self) -> None:
        x1 = self.metadata['x1_label'].value_counts().sort_index()
        x2 = self.metadata['x2_label'].value_counts().sort_index()
        combined_counts = x1.add(x2, fill_value=0).astype(int)
        print(combined_counts)

    def set_num_class_samples(self, num_combinations: int) -> None:
        filtered_df = pd.DataFrame(columns=self.metadata.columns)
        for _, group in self.metadata.groupby('class_label'):
            combinations = list(zip(group['x1_label'], group['x2_label']))
            combinations = list(set(combinations))
            selected_combinations = combinations[:num_combinations]
            x1_values, x2_values = zip(*selected_combinations)
            filtered_group = group[(group['x1_label'].isin(x1_values)) & (group['x2_label'].isin(x2_values))]
            filtered_df = pd.concat([filtered_df, filtered_group]).reset_index(drop=True)
        self.metadata = filtered_df
    
    def reset_class_samples(self) -> None:
        self.metadata = self.__ometadata

    def save_metadata(self, path: str) -> None:
        self.metadata.to_csv(path)

    def __len__(self) -> int:
        return len(self.metadata)
        