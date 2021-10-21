import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim
from torch import nn

class Trainer():
    def __init__(
        self,
        dataloader,
        model,
        epochs,
        batch_size,
        valid_size,
        pin_memory,
        output_dir,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.trainloader, self.validloader, self.testloader = dataloader.get_torch_loaders(valid_size,
                                                                        batch_size, pin_memory)
        self.num_classes = dataloader.num_classes
        self.input_channels = self.trainloader.dataset.data.shape[-1]
        self.dataset_sizes = {
            'train': dataloader.train_size,
            'val': dataloader.valid_size
        }

        self.model = model(self.input_channels, self.num_classes)
        self.model.to(self.device)
        self.train_loss_func = nn.CrossEntropyLoss()

        self.output_dir = output_dir
    
    def train(self, plot: bool=True):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True)

        # storing the accuracy and weights of the best model (on valid dataset)
        best_model_wts, best_acc = None, 0.0
        training_history = {
            'train': {'loss': [], 'acc': []},
            'val': {'loss': [], 'acc': []}
        }

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode            

                running_loss = 0.0
                running_corrects = 0
                current_data_loader = self.trainloader if phase == 'train' else self.testloader

                for _, (inputs, labels) in tqdm(enumerate(current_data_loader), total=len(current_data_loader),
                                                desc=f'{phase}', ncols=80, leave=False):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.train_loss_func(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                training_history[phase]['loss'].append(epoch_loss)
                training_history[phase]['acc'].append(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val':
                    scheduler.step(epoch_loss)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            print()
        
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), self.output_dir)

        if plot:
            Trainer.plot_training_history(training_history)

    @staticmethod
    def plot_training_history(history: dict):
        TRAINING_COLOR, VALID_COLOR = '#58508d', '#ff6361'
        epoch = range(len(history['train']['loss']))
        y_max = max(history['train']['loss'] + history['valid']['loss'])

        fig, axs = plt.subplots(2, 1, figsize=(12,8))
        axs[0].plot(epoch, history['train']['loss'], '.-', label='Training', color=TRAINING_COLOR)
        axs[0].plot(epoch, history['valid']['loss'], '.-', label='Validation', color=VALID_COLOR)
        axs[0].set_ylim(0, y_max+.5)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(epoch, history['train']['acc'], '.-', label='Training', color=TRAINING_COLOR)
        axs[1].plot(epoch, history['valid']['acc'], '.-', label='Validation', color=VALID_COLOR)
        axs[1].set_ylim(0, 1.1)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        fig.tight_layout()
        plt.show()
