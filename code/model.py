import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.vector_dim = config['vector_dim']
        self.class_num = config['class_num']
        self.kernel_size = config['kernel_size']
        self.filter_num = config['filter_num']
        
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.filter_num, kernel_size=(self.kernel_size, self.kernel_size))
        self.max_pool = nn.MaxPool1d(kernel_size=1)

    def forward(self, x):
        
        