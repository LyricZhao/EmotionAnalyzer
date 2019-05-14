import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as text_data

from torchtext.vocab import Vectors

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
            self.embedding = self.embedding.from_pretrained(config.vectors, freeze=True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.filter_num, kernel_size=(self.kernel_size, self.vector_dim))
        self.fc = nn.Linear(self.filter_num, self.class_num)

    def forward(self, x):
        x = x.unsqueeze(1) # x: (batch_size, 1, len)
        x = self.embedding(x) # x: (batch_size, 1, len, wv_dim)
        x = self.conv(x) # x: (batch_size, filter_num, len - kernel_size + 1, 1)
        x = F.relu(x).squeeze(3) # x: (batch_size, filter_num, len - kernel_size + 1)
        x = F.max_pool1d(x, x.size(2)) # x: (batch_size, filter_num, 1)
        x = x.squeeze(2) # x: (batch_size, filter_num, 1)
        x = self.fc(x) # x: (batch_size, class_num)
        return x
        