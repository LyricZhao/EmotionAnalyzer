import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class trainer(object):
    def __init__(self, config, dataset, model):
        super(trainer, self).__init__()
        self.epoch = config['epoch']
        self.dataset = dataset
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        if config['loss'] == 'mse':
            self.loss = nn.MSELoss()
        elif config['loss'] == '':
            self.loss = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        print('[!] Training epoch {} ... '.format(epoch), flush=True)
        for iteration, batch in enumerate(self.dataset):
            self.optimizer.zero_grad()
            input, target = batch[0].cpu(), batch[1].cpu()
            output = self.model(input)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            print(' -  Epoch[{}] ({}/{}): loss: {:.4f}'.format(epoch, iteration, len(self.dataset), loss.item()), end='')
        print('[!] Done !', flush=True)

    def train(self):
        for epoch in range(1, self.epoch + 1):
            self.train_epoch(epoch)