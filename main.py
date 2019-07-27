# -*- coding: utf-8 -*-
import torch
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
from torch.utils.data import Dataset, DataLoader
import os

import mymodel


class_to_idx = {}

# path informations
data_dir = "/home/ace313/Desktop/xyyxiaokelian/flower_classification/flower-classification-2019"
pretrained_model = "/home/ace313/Desktop/xyyxiaokelian/flower_classification/flower-classification-2019/models/resnet10129checkpoint_flower_0.5+2+2lock.pth"

save_dir = './'
basemodel = 'resnet101'

def pil_loader(path):
    with Image.open(path) as img:
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class mydataset(Dataset):
    def __init__(self, img_path,class_to_idx, data_transforms = None, loader = default_loader):
        self.prefixlen = len(img_path)
        self.img_name = [img_path + a for a in os.listdir(img_path)]
        self.data_transforms = data_transforms
        self.loader = loader
    
    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, item):
        img_name = self.img_name[item]
        id = int(img_name[self.prefixlen:-4])
        img = self.loader(img_name)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, id

def data_loaders(data_dir):
    data_dir = data_dir
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    class_to_idx = {v:k for k, v in train_data.class_to_idx.items()}
    test_data = mydataset(test_dir, class_to_idx, data_transforms=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=16)
    return trainloader, validloader, testloader, train_data, valid_data, test_data


# training settings
epochs = 30
gpu = True
# optimizer settings
optimizer = 'Adam'
lr = 1e-3
lr_decay = 0.5
lr_decay_step = 2
criterion = 'NLLLoss'

arguments = dict(
                # training settings
                epochs = 30,
                gpu = True,
                pretrained_model = pretrained_model,
                # optimizer settings
                optimizer = 'Adam', 
                lr = 1e-3, 
                lr_decay = 0.5, 
                lr_decay_step = 2, 
                criterion = 'NLLLoss')


n_classes = 5
steps = 0
times = 100

trainloader, validloader, testloader, train_data, valid_data, test_data = data_loaders(data_dir)

model, optimizer, criterion, lr_scheduler = mymodel.mode(basemodel, lr, arguments)

if(os.path.exists(pretrained_model)):
    dicts = torch.load(pretrained_models)
    st = dicts['model_state']
    model.load_state_dict(st, strict=False)
    class_to_idx = {v:k for k, v in dicts['img_mapping'].items()}
    print(class_to_idx)
force_processor = 'cuda'
processor = ('cuda' if torch.cuda.is_available() and force_processor == 'cuda' else 'cpu')
model.to(processor)

for e in range(epochs):
    if(e == -1):
        for param in model.parameters():  # 先不改变
            param.requires_grad = True
        for param in model.conv1.parameters():  
            param.requires_grad = False
        for param in model.encoder1.parameters():  
            param.requires_grad = False
    model.train()
    running_loss = 0
    lr_scheduler.step()
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        inputs, labels = inputs.to(processor), labels.to(processor)
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % times == 0:
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                accuracy = 0
                for images, labels in validloader:
                    
                    images = images.to(processor)
                    labels = labels.to(processor)

                    output = model.forward(images)
                    valid_loss += criterion(output, labels).item()

                    ps = torch.exp(output)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "training loss: {:.3f} ".format(running_loss / times),
                  "valid loss: {:.3f} ".format(valid_loss / len(validloader)),
                  "valid acc: {:.3f}".format(accuracy / len(validloader)))

            running_loss = 0
            model.train()
    save_params_dict = {
                'lr': lr,
                'epochs_trained': epochs,
                'basemodel': basemodel,
                'optimizer_state': optimizer.state_dict(),
                'model_state': model.state_dict()}

    if (e + 1) % 10 == 0:
        torch.save(save_params_dict, basemodel+str(e)+'_checkpoint_flower.pth')
print('finished training')

model.eval()
csvFile2 = open(basemodel+'_ans.csv', 'w', newline='')
writer = csv.writer(csvFile2)
writer.writerow(('Id', 'Expected'))
tmpdict = {}
with torch.no_grad():
    for data in testloader:
        images, id = data
        id = int(id)
        if processor == 'cuda':
            images = images.cuda()
        outputs = model(images)
        prob, predicted = torch.max(outputs.data, 1)
        np_classes =np.asarray(predicted.cpu())[0]
        tmpdict[id] = np_classes
print(tmpdict)

class_to_idx = {v:k for k, v in train_data.class_to_idx.items()}
for i in range(0, len(tmpdict)):
    writer.writerow((i, class_to_idx[tmpdict[i]]))
print('finished testing')

# saving checkpoint
if (basemodel.startswith('res')):
    save_params_dict = {
                    'lr': lr,
                    'epochs_trained': epochs,
                    'basemodel': basemodel,
                    'optimizer_state': optimizer.state_dict(),
                    'model_state': model.state_dict()}

torch.save(save_params_dict, basemodel+'checkpoint_flower.pth')

print('Done!')
