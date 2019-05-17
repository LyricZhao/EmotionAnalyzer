import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as text_data

from torchtext.vocab import Vectors

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.vector_dim = config['vector_dim']
        self.class_num = config['class_num']
        self.vocabulary_size = config['vocabulary_size']
        self.fix_length = config['fix_length']
        self.hidden_dim = config['hidden_dim']

        self.embedding = nn.Embedding(self.vocabulary_size, self.vector_dim)
        if config['preload_w2v']:
            self.embedding = self.embedding.from_pretrained(config['vectors'], freeze=config['freeze'])
        self.fc1 = nn.Linear(self.fix_length * self.vector_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.class_num)

    def forward(self, x, lengths):
        x = self.embedding(x) # x: (batch_size, len, wv_dim)
        x = x.view(x.size(0), -1) # x: (batch_size, len * wv_dim)
        x = self.fc2(self.fc1(x)) # x: (batch_size, class_num)
        return x

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.vector_dim = config['vector_dim']
        self.class_num = config['class_num']
        self.kernel_size = config['kernel_size']
        self.filter_num = config['filter_num']
        self.vocabulary_size = config['vocabulary_size']

        self.embedding = nn.Embedding(self.vocabulary_size, self.vector_dim)
        if config['preload_w2v']:
            self.embedding = self.embedding.from_pretrained(config['vectors'], freeze=config['freeze'])
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.filter_num, kernel_size=(self.kernel_size, self.vector_dim))
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(self.filter_num, self.class_num)

    def forward(self, x, lengths):
        x = x.unsqueeze(1) # x: (batch_size, 1, len)
        x = self.embedding(x) # x: (batch_size, 1, len, wv_dim)
        x = self.conv(x) # x: (batch_size, filter_num, len - kernel_size + 1, 1)
        x = F.relu(x).squeeze(3) # x: (batch_size, filter_num, len - kernel_size + 1)
        x = F.max_pool1d(x, x.size(2)) # x: (batch_size, filter_num, 1)
        x = x.squeeze(2) # x: (batch_size, filter_num, 1)
        x = self.dropout(x)
        x = self.fc(x) # x: (batch_size, class_num)
        return x

pack_padded_sequence = nn.utils.rnn.pack_padded_sequence

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.vector_dim = config['vector_dim']
        self.class_num = config['class_num']
        self.vocabulary_size = config['vocabulary_size']
        self.hidden_dim = config['hidden_dim']
        self.layers = config['rnn_layers']
        self.use_cuda = config['cuda']
        self.batch_size = config['train_batch_size']
        self.rnn_type = config['rnn_type']

        self.embedding = nn.Embedding(self.vocabulary_size, self.vector_dim)
        if config['preload_w2v']:
            self.embedding = self.embedding.from_pretrained(config['vectors'], freeze=config['freeze'])
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.vector_dim, self.hidden_dim, batch_first=True, dropout=config['dropout'], num_layers=self.layers, bidirectional=config['bidirectional'])
        else:
            self.rnn = nn.GRU(self.vector_dim, self.hidden_dim, batch_first=True, dropout=config['dropout'], num_layers=self.layers, bidirectional=config['bidirectional'])
        self.fc = nn.Linear(self.hidden_dim, self.class_num)
    
    def forward(self, x, lengths):
        # x: (batch_size, len)
        x = self.embedding(x) # x: (batch_size, len, wv_dim)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        if self.rnn_type == 'lstm':
            x, (h, c) = self.rnn(x) # x: (batch_size, len, hidden_dim)
            x = h[-1, :, :] # x: (batch_size, hidden_dim)
        else:
            x, h = self.rnn(x) # x: (batch_size, len, hidden_dim)
            x = h[-1, :, :] # x: (batch_size, hidden_dim)
        x = self.fc(x) # x: (batch_size, class_num)
        return x
        