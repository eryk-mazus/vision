import copy
import torch
from torch import nn
import torch.optim as optim

class Trainer():
    def __init__(
        self,
        dataloader,
        model,
        epochs,
        batch_size,
        valid_size,
        output_dir,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.trainloader, self.validloader, self.testloader = dataloader.get_torch_loaders(valid_size, batch_size,
                                                                pin_memory=(True if self.device == 'cuda' else False))
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
    
    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True)

        best_model_wts = None
        best_acc = 0.0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode            

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in (self.trainloader if phase == 'train' else self.testloader):
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
