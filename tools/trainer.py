import torch
from torch import nn
import torch.optim as optim

class Trainer():
    def __init__(
        self,
        trainset,
        testset,
        model,
        epochs,
        batch_size,
        output_dir,
        logging_steps,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs

        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.num_classes = trainset.classes

        self.input_channels = self.trainloader.dataset.data.shape[-1]
        self.model = model(self.input_channels, self.num_classes)
        self.model.to(self.device)

        self.output_dir = output_dir
        self.logging_steps = logging_steps

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
                if i % self.logging_steps == self.logging_steps-1:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / self.logging_steps))
                    running_loss = 0.0
        
        # saving model
        torch.save(self.model.state_dict(), self.output_dir)
