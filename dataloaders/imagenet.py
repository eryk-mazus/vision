import torch
from torchvision import transforms
from torchvision.datasets import ImageNet

class ImageNet_Loader:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    def __init__(self):
        self.train_set = ImageNet(root='./data', split='train', download=True, transform=ImageNet_Loader.TRAIN_TRANSFORM)
        self.test_set = ImageNet(root='./data', split='val', download=True, transform=ImageNet_Loader.TEST_TRANSFORM)
    
    @property
    def num_classes(self):
        return len(self.train_set.classes)

    @property
    def train_size(self):
        return len(self.train_set)

    @property
    def valid_size(self):
        return len(self.test_set)
    
    def get_torch_loaders(self, batch_size: int=32, pin_memory: bool=False):
        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory
        )

        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory
        )

        return train_loader, test_loader
