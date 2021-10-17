import os
import sys
import argparse
import torchvision.transforms as tt
from torchvision.datasets import CIFAR100

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tools.trainer import Trainer
from tools.model_zoo import get_arch

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
                    help="name of the model to train")
parser.add_argument('--epochs', default=5, type=int,
                    help='number of epochs')
parser.add_argument('--batch_size', default=32, type=int,
                    help='mini batch size for sgd')
parser.add_argument('--output_dir', default='./output/', type=str,
                    help='where to save the model and store logs')

args = parser.parse_args()

# defining transformations for training and test sets
# TODO: move CIFAR100 and its transformation to ./datasets/get_cifar100.py
stats = ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
tr_transform = tt.Compose([
                    tt.RandomChoice([tt.Resize(256), tt.Resize(480)]),
                    tt.RandomHorizontalFlip(),
                    tt.RandomCrop(224),
                    tt.ToTensor(),
                    tt.Normalize(*stats),
                   ])

te_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

if __name__ == "__main__":
    trainset = CIFAR100(root='./data', train=True, download=True, transform=tr_transform)
    testset = CIFAR100(root='./data', train=False, download=True, transform=te_transform)
    model = get_arch(args.model)

    trainer = Trainer(trainset, testset, model, args.epochs, args.batch_size,
                      args.output_dir)
    trainer.train()

