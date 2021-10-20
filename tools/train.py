import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tools.trainer import Trainer
from tools.model_zoo import get_arch
from dataloaders.cifar100 import CIFAR_100_Loader

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
                    help="name of the model to train")
parser.add_argument('--epochs', default=5, type=int,
                    help='number of epochs')
parser.add_argument('--batch_size', default=32, type=int,
                    help='mini batch size for sgd')
parser.add_argument('--valid_size', default=0.2, type=float,
                    help='validation dataset size')
parser.add_argument('--output_dir', default='./output/', type=str,
                    help='where to save the model and store logs')
args = parser.parse_args()

if __name__ == "__main__":
    cifar_loader = CIFAR_100_Loader()
    model = get_arch(args.model)
    trainer = Trainer(cifar_loader, model, args.epochs, args.batch_size,
                      args.valid_size, args.output_dir)
    trainer.train()

