import torch
import os
import numpy as np
import csv
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from PIL import Image
import argparse

class resnet_plus(nn.Module):
    def __init__(self, arch):
        super(resnet_plus, self).__init__()
        resnet = getattr(models, arch)(pretrained = True)
        
        for param in resnet.parameters():  
            param.requires_grad = False
        self.conv1 = resnet.conv1  # H/2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # H/4

        self.encoder1 = resnet.layer1  # H/4
        self.encoder2 = resnet.layer2  # H/8
        self.encoder3 = resnet.layer3  # H/16
        self.encoder4 = resnet.layer4  # H/32

        self.avgpool = resnet.avgpool

        self.classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.25)),
            ('inputs', nn.Linear(512*4, 5))
             ]))
        self.trainable_vars = [param for param in self.classifier.parameters()]
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

def mode(arch, lr, arguments):
    model = resnet_plus(arch)
    trainble_vars = model.trainable_vars
    criterion = nn.CrossEntropyLoss()
    
    if(os.path.exists(arguments['pretrained_model'])):
        dicts = torch.load(arguments['pretrained_model'])
        lr = dicts['lr']
        opt = dicts['optimizer_state']

    if arguments['optimizer']=='Adam':
        optimizer = optim.Adam(model.parameters(), lr,weight_decay=0)
    else:
        optimizer = optim.SGD(model.parameters(), lr,momentum=0.8, weight_decay=0)

    lr_decay = lr_scheduler.StepLR(optimizer, step_size=arguments['lr_decay_step'], gamma=arguments['lr_decay'])

    return model, optimizer, criterion, lr_decay
