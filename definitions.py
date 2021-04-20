import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame, concat, read_csv
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, output = 1):
        super(Net, self).__init__()

        pool_size=2 #pooling kernel
        pool_stride=2 #pooling size

        self.pool = nn.MaxPool2d(pool_size, pool_stride)#summarizes the most activated presence of a feature

        up_pool_size = 2
        self.up_pool = nn.Upsample(scale_factor = up_pool_size)
        
        #layer = [kernel, padding, # of filters]
        layer1 = [7,3,8]
        layer2 = [7,3,8]
        layer3 = [5,2,16]
        layer4 = [5,2,16]
        layer5 = [3,1,32]
        layer6 = [3,1,32]
        layer7 = [1,0,1]
        layer8 = [3,1,8]
        layer9 = [3,1,1]

        self.conv1 = nn.Conv2d(3, layer1[2], layer1[0], padding = layer1[1])
        size1=48+2*layer1[2]-(layer1[0]-1)
        size2=np.floor((size1 - pool_size)/pool_stride + 1).astype(int)

        self.conv2 = nn.Conv2d(layer1[2], layer2[2], layer2[0], padding = layer2[1])
        size3 = size2+2*layer2[1]-(layer2[0]-1)
        size4=np.floor((size3 - pool_size)/pool_stride + 1).astype(int)

        self.conv3 = nn.Conv2d(layer2[2], layer3[2], layer3[0], padding = layer3[1])
        size5 = size4+2*layer3[1]-(layer3[0]-1)
        size6 = np.floor((size5 - pool_size)/pool_stride + 1).astype(int)

        self.conv4 = nn.Conv2d(layer3[2], layer4[2], layer4[0], padding = layer4[1])
        size7 = size6+2*layer4[1]-(layer4[0]-1)
        size8 = np.floor((size7 - pool_size)/pool_stride + 1).astype(int)

        self.conv5 = nn.Conv2d(layer4[2], layer5[2], layer5[0], padding = layer5[1])
        self.conv6 = nn.Conv2d(layer5[2], layer6[2], layer6[0], padding = layer6[1])
        self.conv7 = nn.Conv2d(layer6[2], layer7[2], layer7[0], padding = layer7[1])
        #self.conv8 = nn.Conv2d(layer7[2], layer8[2], layer8[0], padding = layer8[1])
        #self.conv9 = nn.Conv2d(layer8[2], layer9[2], layer9[0], padding = layer9[1])
        
        #self.fc1 = nn.Linear(output4 * size8 * size8, )
        #self.fc2 = nn.Linear(120,20)
        #self.fc_out = nn.Linear(20,1)

    def forward(self, x):
        x = self.up_pool(x)
        x = self.up_pool(x)
        
        x = self.conv1(x)
        #x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.conv3(x)
        #x = self.pool(F.relu(x))
        x = self.conv4(x)
        x = self.pool(F.relu(x))
        x = self.conv5(x)
        #x = self.up_pool(x)
        x = self.conv6(x)
        #x = self.pool(F.relu(x))
        
        x = self.conv7(x)
        #x = self.up_pool(x)
        
        #x = self.conv8(x)
        #x = self.conv9(x)
        
        #x = x.view(-1, self.num_flat_features(x))

        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc_out(x)

        #x = x.reshape(x.shape[0])
        x = np.squeeze(x)
        
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
        labels = np.squeeze(labels)

        #print()
        #print(f"Image dtype: {image.dtype}")
        #print(f"Labels dtype: {labels.dtype}")
        #print()
        
        return {'inputs': torch.from_numpy(image).float(),
                'labels': labels}
