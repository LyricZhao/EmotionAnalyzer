import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class trainer(object):
    def __init__(self, config, train_iter, test_iter, model):
        super(trainer, self).__init__()
        self.epoch = config['epoch']
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        if config['loss'] == 'mse':
            self.loss = nn.MSELoss()
        elif config['loss'] == 'cel':
            self.loss = F.cross_entropy

    def train_epoch(self, epoch):
        print('[!] Training epoch {} ... '.format(epoch), flush=True)
        for iteration, batch in enumerate(self.train_iter):
            self.optimizer.zero_grad()
            input, target = batch.text, batch.label.squeeze(1)
            output = self.model(input)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            print(' -  Epoch[{}] ({}/{}): loss: {:.4f}'.format(epoch, iteration, len(self.train_iter), loss.item()), flush=True)
        print('[!] Done !', flush=True)

    def evaluate(self):
        print('[!] Evaluating results ... ', flush=True)
        size, tot = 0, 0
        for iteration, batch in enumerate(self.test_iter):
            input, target = batch.text, batch.label.squeeze(1)
            output = self.model(input)
            size += target.shape[0]
            tot += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        print('[!] Acc: {:.5f}'.format(100.0 * tot / size), flush=True)

    def train(self):
        for epoch in range(1, self.epoch + 1):
            self.train_epoch(epoch)
            self.evaluate()