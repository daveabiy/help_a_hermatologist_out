update_model = False
from codecs import ignore_errors
import os
from time import sleep
from glob import glob
import random
from tqdm import tqdm
import copy
import ntpath

import numpy as np
from imageio import imread
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, matthews_corrcoef, classification_report,confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,  precision_score, recall_score
from skimage import io as io
from skimage.util import *

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models


import data_processing as dp


data_path = {
        "Ace_20": "/beegfs/desy/user/hailudaw/challenge/Datasets/Acevedo_20", # Acevedo_20 Dataset
        "Mat_19": "/beegfs/desy/user/hailudaw/challenge/Datasets/Matek_19", # Matek_19 Dataset
        "Ace_20_noisy": "/beegfs/desy/user/hailudaw/challenge/Datasets/Acevedo_20_noisy", # Acevedo_20 Dataset
        "Mat_19_noisy": "/beegfs/desy/user/hailudaw/challenge/Datasets/Matek_19_noisy", # Matek_19 Dataset
        "WBC1": "/beegfs/desy/user/hailudaw/challenge/Datasets/WBC1", # WBC1 dataset
        "WBC2": "/beegfs/desy/user/hailudaw/challenge/Datasets/WBC2"
    }

label_map_all = {
        'basophil': 0,
        'eosinophil': 1,
        'erythroblast': 2,
        'myeloblast' : 3,
        'promyelocyte': 4,
        'myelocyte': 5,
        'metamyelocyte': 6,
        'neutrophil_banded': 7,
        'neutrophil_segmented': 8,
        'monocyte': 9,
        'lymphocyte_typical': 10
    }

label_map_reverse = {
        0: 'basophil',
        1: 'eosinophil',
        2: 'erythroblast',
        3: 'myeloblast',
        4: 'promyelocyte',
        5: 'myelocyte',
        6: 'metamyelocyte',
        7: 'neutrophil_banded',
        8: 'neutrophil_segmented',
        9: 'monocyte',
        10: 'lymphocyte_typical'
    }

# The unlabeled WBC dataset gets the classname 'Data-Val' for every image

label_map_pred = {
        'DATA-VAL': 0
    }
label_map_after = {
        'DATA-TEST': 0
    }

# ## Data loading
# We use pandas dataframes to systematically order and later load the data:

savepaths=['metadata.csv', 'metadata_all.csv', 'metadata_with_noisy.csv', 'metadata2.csv', 'metadata3.csv'] # path where the created dataframe will be stored
savepath = savepaths[1]

metadata = pd.read_csv(savepath)
print(metadata.head())

ace_metadata=metadata.loc[metadata['dataset']=='Ace_20'].reset_index(drop = True)
ace_metadata_noisy = metadata.loc[metadata['dataset']=='Ace_20_noisy'].reset_index(drop = True)
mat_metadata=metadata.loc[metadata['dataset']=='Mat_19'].reset_index(drop = True)
mat_metadata_noisy = metadata.loc[metadata['dataset']=='Mat_19_noisy'].reset_index(drop = True)
wbc_metadata=metadata.loc[metadata['dataset']=='WBC1'].reset_index(drop = True)
wbc2_metadata=metadata.loc[metadata['dataset']=='WBC2'].reset_index(drop = True)
ace_metadata_noisy_drop_basophil = ace_metadata_noisy[ace_metadata_noisy['label'] != 'basophil'].reset_index(drop = True)
mat_metadata_noisy_drop_neutrophil_segmented = mat_metadata_noisy[mat_metadata_noisy['label'] != 'neutrophil_segmented'].reset_index(drop = True)

def output_data(metadata=wbc2_metadata):
    outputdata =wbc_metadata.drop(columns=['file','label', 'dataset', 'set', 'mean1', 'mean2', 'mean3', 'std1', 'std2', 'std3', 'max1', 'max2', 'max3', 'min1', 'min2', 'min3'])
    outputdata['Label']=None
    outputdata['LabelID']=None
    for i in range(len(outputdata)):
        outputdata['LabelID'].loc[i]=random.randint(0, 10) #for the 10 possible classes
        outputdata['Label'].loc[i]=label_map_reverse[outputdata['LabelID'].loc[i]]
    outputdata.to_csv('submission.csv', index=False)
    return outputdata

outputdata1 = output_data(wbc_metadata)
outputdata2 = output_data(wbc2_metadata)

example_metadata=metadata
source_domains=['Ace_20', 'Mat_19', 'Ace_20_noisy', 'Mat_19_noisy']
source_index = example_metadata.dataset.isin(source_domains)
example_metadata = example_metadata.loc[source_index,:].copy().reset_index(drop = True)

test_fraction=0.2 #of the whole dataset
val_fraction=0.125 #of 0.8 of the dataset (corresponds to 0.1 of the whole set)
train_index, test_index, train_label, test_label = train_test_split(
    example_metadata.index,
    example_metadata.label + "_" + example_metadata.dataset,
    test_size=test_fraction,
    random_state=0, 
    shuffle=True,
    stratify=example_metadata.label
    )
example_metadata.loc[test_index, 'set']='test'
train_val_metadata=example_metadata.loc[train_index]

train_index, val_index, train_label, val_label = train_test_split(
    train_val_metadata.index,
    train_val_metadata.label + "_" + train_val_metadata.dataset,
    test_size=val_fraction,
    random_state=0, 
    shuffle=True, 
    stratify=train_val_metadata.label
    )
train_size=len(example_metadata.loc[example_metadata['set'] == 'train'])
val_size=len(example_metadata.loc[example_metadata['set'] == 'val'])
test_size=len(example_metadata.loc[example_metadata['set'] == 'test'])

print(example_metadata.loc[:, 'set'].value_counts())

import dataset_generator as dg
resize=224 #image pixel size
number_workers=3

random_crop_scale=(0.8, 1.0)
random_crop_ratio=(0.8, 1.2)

mean=[0.485, 0.456, 0.406] #values from imagenet
std=[0.229, 0.224, 0.225] #values from imagenet

bs=25 #batchsize
normalization = torchvision.transforms.Normalize(mean,std)

train_transform = transforms.Compose([ 
        normalization,
        transforms.RandomResizedCrop(resize, scale=random_crop_scale, ratio=random_crop_ratio),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
])
val_transform = transforms.Compose([ 
        normalization,
        transforms.Resize(resize)])

test_transform = transforms.Compose([ 
        normalization,
        transforms.Resize(resize)])


train_dataset = dg.DatasetGenerator(example_metadata.loc[train_index,:], 
                                 reshape_size=resize, 
                                 dataset = source_domains,
                                 label_map=label_map_all, 
                                 transform = train_transform,
                                 )
val_dataset = dg.DatasetGenerator(example_metadata.loc[val_index,:], 
                                 reshape_size=resize, 
                                 dataset = source_domains,
                                 label_map=label_map_all, 
                                 transform = val_transform,
                                 )

test_dataset = dg.DatasetGenerator(example_metadata.loc[test_index,:], 
                                 reshape_size=resize, 
                                 dataset = source_domains,
                                 label_map=label_map_all, 
                                 transform = test_transform,
                                 )
train_loader = DataLoader(
    train_dataset, batch_size=bs, shuffle=True, num_workers=number_workers)
valid_loader = DataLoader(
    val_dataset, batch_size=bs, shuffle=True, num_workers=number_workers)
test_loader = DataLoader(
    test_dataset, batch_size=bs, shuffle=False, num_workers=number_workers)

epochs=20 # max number of epochs
lr=0.003 # learning rate
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(label_map_all)
architectures = ['resnet18', 'resnet50', 'resnet152']
arch = architectures[1]   #set the architecture type here
import torchvision.models as models

if (arch == 'resnet18'):
    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    model_save_path='model' #path where model with best f1_macro should be stored
elif (arch == 'resnet50'):
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    model_save_path='model50' #path where model with best f1_macro should be stored
    
else:# (arch == 'resnet152'):
    from torchvision.models import resnet152
    model = resnet152(pretrained=True)
    model_save_path='model152' #path where model with best f1_macro should be stored
if update_model == True:
    model = torch.load(model_save_path)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = nn.DataParallel(model) 
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# #running variables
epoch=0
update_frequency=5 # number of batches before viewed acc and loss get updated
counter=0 #counts batches
f1_macro_best=0 #minimum f1_macro_score of the validation set for the first model to be saved
loss_running=0
acc_running=0
val_batches=0

y_pred=torch.tensor([], dtype=int)
y_true=torch.tensor([], dtype=int)
y_pred=y_pred.to(device)
y_true=y_true.to(device)


for epoch in range(0, epochs):
    #training
    model.train()
    
    with tqdm(train_loader) as tepoch:  
        for i, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}")
            counter+=1

            x, y = data
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            logits = torch.softmax(out.detach(), dim=1)
            predictions = logits.argmax(dim=1)
            acc = accuracy_score(y.cpu(), predictions.cpu())
            
            if counter >= update_frequency:
                tepoch.set_postfix(loss=loss.item(), accuracy=acc.item())
                counter=0
                
    #validation       
    model.eval()
    with tqdm(valid_loader) as vepoch: 
        for i, data in enumerate(vepoch):
            vepoch.set_description(f"Validation {epoch+1}")
    
            x, y = data
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)
            
            logits = torch.softmax(out.detach(), dim=1)
            predictions = logits.argmax(dim=1)
            y_pred=torch.cat((y_pred, predictions), 0)
            y_true=torch.cat((y_true, y), 0)
            
            acc = accuracy_score(y_true.cpu(), y_pred.cpu())
            
            loss_running+=(loss.item()*len(y))
            acc_running+=(acc.item()*len(y))
            val_batches+=len(y)
            loss_mean=loss_running/val_batches
            acc_mean=acc_running/val_batches
            
            vepoch.set_postfix(loss=loss_mean, accuracy=acc_mean)
            
        f1_micro=f1_score(y_true.cpu(), y_pred.cpu(), average='micro')
        f1_macro=f1_score(y_true.cpu(), y_pred.cpu(), average='macro')
        print(f'f1_micro: {f1_micro}, f1_macro: {f1_macro}')  
        if f1_macro > f1_macro_best:
            f1_macro_best=f1_macro
            torch.save(model.state_dict(), model_save_path)
            print('model saved')
        
        #reseting running variables
        loss_running=0
        acc_running=0
        val_batches=0
            
        y_pred=torch.tensor([], dtype=int)
        y_true=torch.tensor([], dtype=int)
        y_pred=y_pred.to(device)
        y_true=y_true.to(device)
            
        
    
print('Finished Training')

#loading the model with the highest validation accuracy
model.load_state_dict(torch.load(model_save_path))
print(model)