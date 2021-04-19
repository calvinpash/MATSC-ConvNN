import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame, concat, read_csv
import numpy as np
from PIL import Image
import os

class Net(nn.Module):
    def __init__(self, output = 1):
        super(Net, self).__init__()

        padding1=1
        filter1=3
        output1 = 8

        self.conv1 = nn.Conv2d(3, output1, filter1, padding = padding1)

        size1=48+2*padding1-(filter1-1)

        pool_size=2 #pooling kernel
        pool_stride=2 #pooling size

        self.pool = nn.MaxPool2d(pool_size, pool_stride)#summarizes the most activated presence of a feature

        size2=np.floor((size1 - pool_size)/pool_stride + 1).astype(int)

        padding2=1
        filter2=3
        output2=8

        self.conv2 = nn.Conv2d(output1, output2, filter2, padding = padding2)

        size3 = size2+2*padding2-(filter2-1)
        size4=np.floor((size3 - pool_size)/pool_stride + 1).astype(int)

        #padding3 = 1
        #filter3 = 3
        #output3 = 32

        #self.conv3 = nn.Conv2d(output2, output3, filter3, padding = padding3)

        #size5 = size4+2*padding3-(filter3-1)
        #size6 = np.floor((size5 - pool_size)/pool_stride + 1).astype(int)

        #filter4 = 3
        #padding4 = 1
        #output4 = 8

        #self.conv4 = nn.Conv2d(output3, output4, filter4, padding = padding4)

        #size7 = size6+2*padding4-(filter4-1)
        #size8 = np.floor((size7 - pool_size)/pool_stride + 1).astype(int)

        self.fc1 = nn.Linear(output2 * size4 * size4, 20)
        #self.fc2 = nn.Linear(120,20)
        self.fc_out = nn.Linear(20,1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        #x = self.conv3(x)
        #x = self.pool(F.relu(x))
        #x = self.conv4(x)
        #x = self.pool(F.relu(x))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc_out(x)

        x = x.reshape(x.shape[0])
        
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  #all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_output_size(self):
        return self.fcs.out_features

class StressDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        data = np.load(data_dir)
        self.layers = np.array(list(data.values())[0])
        self.labels = np.array(list(data.values())[1])
        
        #self.labels = (np.array(list(data.values())[1])).astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'inputs': self.layers[idx], 'labels': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, labels  = sample['inputs'], sample['labels']

        image = image.transpose((2,0,1))

        return {'inputs': torch.from_numpy(image).float(),
                'labels': labels}
