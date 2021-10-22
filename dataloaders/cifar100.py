import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR100

class CIFAR_100_Loader:
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                    std=[0.267, 0.256, 0.276])
    TRAIN_TRANSFORM = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
                    ])
    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), normalize])

    def __init__(self):
        self.train_set = CIFAR100(root='./data', train=True, download=True, transform=CIFAR_100_Loader.TRAIN_TRANSFORM)
        self.test_set = CIFAR100(root='./data', train=False, download=True, transform=CIFAR_100_Loader.TEST_TRANSFORM)

    @property
    def num_classes(self):
        return len(self.train_set.classes)

    @property
    def train_size(self):
        return len(self.train_set)

    @property
    def test_set(self):
        return len(self.test_set)

    def get_torch_loaders(self, batch_size: int=32, pin_memory: bool=False):
        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory,
        )

        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory,
        )

        return train_loader, test_loader
