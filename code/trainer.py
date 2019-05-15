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
        self.cuda = config['cuda']
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        if config['loss'] == 'mse':
            self.loss = nn.MSELoss()
        elif config['loss'] == 'cel':
            self.loss = nn.CrossEntropyLoss()
        if self.cuda:
            self.model = self.model.cuda()

    def train_epoch(self, epoch):
        self.model.train()
        print('[!] Training epoch {} ... '.format(epoch), flush=True)
        for iteration, batch in enumerate(self.train_iter):
            self.optimizer.zero_grad()
            (input, lengths), target = batch.text, batch.label
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            target = target.squeeze(1)
            output = self.model(input, lengths)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            print(' -  Epoch[{}] ({}/{}): loss: {:.4f}'.format(epoch, iteration + 1, len(self.train_iter), loss.item()), flush=True)
        print('[!] Done !', flush=True)

    def evaluate(self):
        print('[!] Evaluating results ... ', flush=True)
        self.model.eval()
        size, tot = 0, 0
        for iteration, batch in enumerate(self.test_iter):
            (input, lengths), target = batch.text, batch.label
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            target = target.squeeze(1)
            output = self.model(input, lengths)
            size += target.shape[0]
            tot += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        print('[!] Acc: {:.5f}'.format(float(tot) / size), flush=True)

    def train(self):
        for epoch in range(1, self.epoch + 1):
            self.train_epoch(epoch)
            self.evaluate()