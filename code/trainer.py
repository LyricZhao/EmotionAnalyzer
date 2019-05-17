import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn import metrics
from scipy.stats import pearsonr

class trainer(object):
    def __init__(self, config, train_iter, test_iter, model):
        super(trainer, self).__init__()
        self.epoch = config['epoch']
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.model = model
        self.cuda = config['cuda']
        self.global_iter = 0
        self.tensorboard = config['tensorboard']
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.loss = nn.CrossEntropyLoss()
        if self.cuda:
            self.model = self.model.cuda()
        if self.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(comment=config['comment'])

    def train_epoch(self, epoch):
        self.model.train()
        print('[!] Training epoch {} ... '.format(epoch), flush=True)
        for iteration, batch in enumerate(self.train_iter):
            self.optimizer.zero_grad()
            (input, lengths), target = batch.text, batch.label
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            target = target[:, 0]
            output = self.model(input, lengths)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            print(' -  Epoch[{}] ({}/{}): loss: {:.4f}'.format(epoch, iteration + 1, len(self.train_iter), loss.item()), flush=True)
            if self.tensorboard:
                self.global_iter += 1
                self.writer.add_scalar('train/loss', loss.item(), self.global_iter)
        print('[!] Done !', flush=True)

    def evaluate(self, iter, comment, epoch):
        print('[!] Evaluating results ... ', flush=True)
        self.model.eval()
        pred, real = [], []
        pearson_sum = 0.0
        for iteration, batch in enumerate(iter):
            (input, lengths), target = batch.text, batch.label
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            output = self.model(input, lengths)
            output = F.softmax(output, dim=1)
            pred = pred + torch.max(output, 1)[1].view(target[:, 0].size()).data.tolist()
            real = real + target[:, 0].data.tolist()
            output, target = output.detach().cpu(), target.detach().cpu()
            for i in range(output.size(0)):
                dist = target[i, 1:].float()
                dist = dist / torch.sum(dist, dim=0)
                pearson_sum += pearsonr(output[i, :], dist)[0]
        acc = metrics.accuracy_score(real, pred)
        f_score = metrics.f1_score(real, pred, average='macro')
        pearson = pearson_sum / len(pred)
        print('[!] Result on {}: acc={:.5f} f_score={:.5f} pearson={:.5f}'.format(comment, acc, f_score, pearson), flush=True)
        if self.tensorboard:
            self.writer.add_scalar(comment + '/acc', acc, epoch)
            self.writer.add_scalar(comment + '/f_score', acc, epoch)
            self.writer.add_scalar(comment + '/pearson', acc, epoch)

    def train(self):
        self.global_iter = 0
        for epoch in range(1, self.epoch + 1):
            self.train_epoch(epoch)
            self.evaluate(self.train_iter, 'train_dataset', epoch)
            self.evaluate(self.test_iter, 'test_dataset', epoch)
        if self.tensorboard:
            self.writer.close()