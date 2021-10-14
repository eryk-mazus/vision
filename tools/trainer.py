import os
import sys
import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as tt
from torchvision.datasets import CIFAR100

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from models.resnet import resnet34, resnet50

# defining transformations for training and test sets
stats = ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
tr_transform = tt.Compose([
                    tt.RandomChoice([tt.Resize(256), tt.Resize(480)]),
                    tt.RandomHorizontalFlip(),
                    tt.RandomCrop(224),
                    tt.ToTensor(),
                    tt.Normalize(*stats),
                   ])

te_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

class Trainer():
    def __init__(self, model, batch_size, epochs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.output_dir = './model-output/'
        # TODO: create output dir if not exists

        self.trainset = CIFAR100(root='./data', train=True, download=True, transform=tr_transform)
        self.testset = CIFAR100(root='./data', train=False, download=True, transform=te_transform)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.num_classes = self.trainset.classes
        self.input_channels = self.trainloader.dataset.data.shape[-1]

        self.model = resnet34(self.input_channels, self.num_classes) if model == 'resnet34' \
                              else resnet50(self.input_channels, self.num_classes)
        self.model.to(self.device)

        self.train_loss_func = None
    
    def train(self):
        self.train_loss_func = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

        for epoch in range(self.epochs):
            
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.train_loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 500 == 499:    # print every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500))
                    running_loss = 0.0
        
        # saving model
        torch.save(self.model.state_dict(), self.output_dir)










# We use SGD with a mini-batch size of 256. The learning rate
# starts from 0.1 and is divided by 10 when the error plateaus,
# and the models are trained for up to 60 × 104
# iterations. We
# use a weight decay of 0.0001 and a momentum of 0.9. We
# do not use dropout [14], following the practice in [16].



# 
# args to trainer
# config ?
# https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    # scheduling learning rate
# prints/logs