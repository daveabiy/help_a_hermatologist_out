import numpy as np
from imageio import imread
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy 

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
dataset_image_size = {}

resize=224 #image pixel size
number_workers=3

random_crop_scale=(0.8, 1.0)
random_crop_ratio=(0.8, 1.2)

mean=[0.485, 0.456, 0.406] #values from imagenet
std=[0.229, 0.224, 0.225] #values from imagenet

bs=32 #batchsize
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

class DatasetGenerator:

    def __init__(self, 
                metadata, 
                reshape_size=64, 
                label_map=[],
                dataset = [],
                transform=None,
                selected_channels = [0,1,2],
                dataset_image_size=None):

        self.metadata = metadata.copy().reset_index(drop = True)
        self.label_map = label_map
        self.transform = transform
        self.selected_channels = selected_channels
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset =  self.metadata.loc[idx,"dataset"]
        # crop_size = dataset_image_size[dataset]
        crop_size = 224
        h5_file_path = self.metadata.loc[idx,"file"]
        image= imread(h5_file_path)[:,:,self.selected_channels]
        image = image / 255.
        h1 = (image.shape[0] - crop_size) /2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) /2
        h2 = int(h2)
        
        w1 = (image.shape[1] - crop_size) /2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) /2
        w2 = int(w2)
        image = image[h1:h2,w1:w2, :]
        image = np.transpose(image, (2, 0, 1))
        label = self.metadata.loc[idx,"label"]
 

        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image)) 
        image = image.float()
        
        if self.transform:
            image = self.transform(image) 
        
        label = self.label_map[label]
        label = torch.tensor(label).long()
        return image.float(),  label