import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR100

class CIFAR_100_Loader:
    NORMALIZATION_STATS = ((0.507, 0.487, 0.441),(0.267, 0.256, 0.276))
    TRAIN_TRANSFORM = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(*NORMALIZATION_STATS),
                    ])
    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORMALIZATION_STATS)])

    def __init__(self) -> None:
        self.train_set = CIFAR100(root='./data', train=True, download=True, transform=CIFAR_100_Loader.TRAIN_TRANSFORM)
        self.valid_set = CIFAR100(root='./data', train=True, download=True, transform=CIFAR_100_Loader.TEST_TRANSFORM)
        self.test_set = CIFAR100(root='./data', train=False, download=True, transform=CIFAR_100_Loader.TEST_TRANSFORM)
        
        self.__train_size = None
        self.__valid_size = None

    @property
    def num_classes(self):
        return len(self.train_set.classes)

    @property
    def train_size(self):
        return self.__train_size

    @property
    def valid_size(self):
        return self.__valid_size

    def get_torch_loaders(self, valid_size: float=0.2, batch_size: int=32,
                          pin_memory: bool=False, random_seed: int=42):
        np.random.seed(random_seed)
        num_train = len(self.train_set)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.__train_size = len(train_idx)
        self.__valid_size = len(valid_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size, sampler=train_sampler,
            num_workers=2, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            self.valid_set, batch_size=batch_size, sampler=valid_sampler,
            num_workers=2, pin_memory=pin_memory,
        )

        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory,
        )

        return train_loader, valid_loader, test_loader
