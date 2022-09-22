#this function shows information about the datasets classes and rgb means

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import ntpath
import os
import skimage.io as io
import torch
import copy

savepaths=['metadata.csv', 'metadata_noisy.csv', 'metadata_rescaled.csv'] # path where the created dataframe will be stored
savepath = savepaths[0]  # path where the created dataframe will be stored

metadata = []
ace_metadata, mat_metadata, wbc_metadata = [], [], []
datapath = ''


def finding_classes(data_dir):
    """
    this function finds the folders in the root path and considers them
    as classes
    """
    classes = [folder for folder in sorted(os.listdir(data_dir)) if not folder.startswith('.') and not folder.startswith('_')]
    return classes


def metadata_generator(data_path):
    #this function generates a pandas dataframe containing image information (paths, labels, dataset)
    metadata = pd.DataFrame(columns=["Image", "file", "label", "dataset", "set"])
    for ds in data_path:
        list_of_classes = finding_classes(data_path[ds])
        for cl in list_of_classes:
            list_of_pds = []
            metadata_dummy = pd.DataFrame(columns=["Image", "file", "label", "dataset", "set", 'mean1', 'mean2', 'mean3', 'std1', 'std2', 'std3', 'max1', 'max2', 'max3', 'min1', 'min2', 'min3'])
            metadata_dummy["Image"] = None
            metadata_dummy["file"] =  io.imread_collection(os.path.join(data_path[ds], cl, "*")).files
            metadata_dummy["label"] = cl
            metadata_dummy["dataset"] = ds
            metadata_dummy["set"] = "train"
            metadata_dummy["mean1"] = None
            metadata_dummy["mean2"] = None
            metadata_dummy["mean3"] = None
            metadata_dummy["std1"] = None
            metadata_dummy["std2"] = None
            metadata_dummy["std3"] = None
            metadata_dummy["max1"] = None
            metadata_dummy["max2"] = None
            metadata_dummy["max3"] = None
            metadata_dummy["min1"] = None
            metadata_dummy["min2"] = None
            metadata_dummy["min3"] = None

            for i in range(len(metadata_dummy)):
                metadata_dummy['Image'].loc[i]=ntpath.basename(metadata_dummy['file'][i])
            metadata = pd.concat([metadata, metadata_dummy], axis=0, ignore_index=True)
            metadata_dummy = None
            
    return metadata

    
def compute_mean(dataframe=metadata, savepath=savepath, selected_channels=[0,1,2]):
    """
    this function computes the mean of the images in the dataframe
    the mean is computed for each channel and stored in the dataframe
    the dataframe is saved in the savepath
    """
    for idx in tqdm.tqdm(range(len(dataframe))):
        img = io.imread(dataframe['file'][idx])
        for ch in selected_channels:
            dataframe['mean'+str(ch+1)].loc[idx] = np.mean(img[:,:,ch])
        # dataframe.to_csv(savepath)  
    return dataframe

import rasterio as rio
def make_stat(metadata = metadata, savepath = 'metadata3.csv'):
    """
    This function computes the mean and std of the images in the dataframe
    the mean and std are computed for each channel and stored in the dataframe
    the dataframe is saved in the savepath (advisable)
    """

    for idx in tqdm.tqdm(range(len(metadata))):
        added = metadata.values[idx]
        try:
            image = io.imread(added[1]) 
        except:  
            src = rio.open(added[1])
            image = src.read()

        for ch in range(3):
            added[5+ch] = np.mean(image[:,:,ch])
            added[8+ch] = np.std(image[:,:,ch])
            added[11+ch] = np.max(image[:,:,ch])
            added[14+ch] = np.min(image[:,:,ch])
        metadata.loc[idx] = added
    metadata.to_csv(savepath, index=False)
    return metadata

# def compute_mean(dataframe=metadata, savepath=savepath, selected_channels=[0,1,2]):
#     """
#     this function computes the mean of the selected channels
#     """
#     for idx in tqdm(range(len(dataframe)), position=0, leave=True):
#         if dataframe.loc(idx, "dataset") != "WBC1":
#             h5_file_path = dataframe.loc[idx,"file"]
#             try:
#                 image= io.imread(h5_file_path)[:,:,selected_channels]
#             except ValueError: 
#                 print(h5_file_path)
#                 break
#             #image = rgb2hsv(image)
#             dataframe.loc[idx, 'mean1']= np.mean(image[:,:,0])
#             dataframe.loc[idx, 'mean2']= np.mean(image[:,:,1])
#             dataframe.loc[idx, 'mean3']= np.mean(image[:,:,2])
#     dataframe.to_csv(savepath, index=False)
#     print(f'The dataframe was saved to {savepath}')
#     print(dataframe)
#     return dataframe

def data_report(dataframe=metadata, label=None, color1='lightblue', color2='darkblue'):
    """
    this function shows information about the datasets classes and rgb means
    """

    print('\033[1m' + 'label \t \t \timages'+ '\033[0m')
    print('')
    print(f'total \t \t \t{len(dataframe)}')
    print(dataframe.label.value_counts())
 
    x1=np.array(dataframe['mean1'])
    x2=np.array(dataframe['mean2'])
    x3=np.array(dataframe['mean3'])
    mean1=np.mean(np.array(dataframe['mean1']))
    mean2=np.mean(np.array(dataframe['mean2']))
    mean3=np.mean(np.array(dataframe['mean3']))
    std1=np.std(np.array(dataframe['mean1']))
    std2=np.std(np.array(dataframe['mean2']))
    std3=np.std(np.array(dataframe['mean3']))
    print('\033[1m' + 'mean \t \tstd'+ '\033[0m')
    print(f'red: {np.round_(mean1, decimals=2)} \tred: {np.round_(std1, decimals=2)}')
    print(f'green: {np.round_(mean2, decimals=2)} \tgreen: {np.round_(std2, decimals=2)}')
    print(f'blue: {np.round_(mean3, decimals=2)} \tblue: {np.round_(std3, decimals=2)}')
    print('')



def data_plot(dataframes=[ace_metadata, mat_metadata, wbc_metadata],
                labels=['Ace_20', 'Mat_19', 'WBC1'],
                colors1=['lightblue', 'orange', 'greenyellow'],
                colors2=['darkblue', 'red', 'limegreen'],
                save_name='plot_all'):
    """
    this function plots the data
    """
    f, axarr = plt.subplots(1,3, figsize=(15,5))
    df=0
    while df<len(dataframes):
        
        dataframe = dataframes[df]
        label=labels[df]
        color1=colors1[df]
        color2=colors2[df]

        x1=np.array(dataframe['mean1'])
        x2=np.array(dataframe['mean2'])
        x3=np.array(dataframe['mean3'])
        mean1=np.mean(np.array(dataframe['mean1']))
        mean2=np.mean(np.array(dataframe['mean2']))
        mean3=np.mean(np.array(dataframe['mean3']))
        std1=np.std(np.array(dataframe['mean1']))
        std2=np.std(np.array(dataframe['mean2']))
        std3=np.std(np.array(dataframe['mean3']))
        
        # red vs green
        
        axarr[0].set_xlabel("red")
        axarr[0].set_ylabel("green")

        a=np.array((x1,x2)).T

        axarr[0].scatter(a[:, 0], a[:, 1], s=3, color=color1, alpha=1)
        axarr[0].scatter(x=mean1, y=mean2, s=1, color=color2)
        axarr[0].plot([mean1-std1, mean1+std1],[mean2, mean2], color=color2, label=label)
        axarr[0].plot([mean1, mean1],[mean2-std2, mean2+std2], color=color2)


        # red vs blue

        axarr[1].set_xlabel("red")
        axarr[1].set_ylabel("blue")

        b=np.array((x1,x3)).T

        axarr[1].scatter(b[:, 0], b[:, 1], s=3, color=color1, alpha=1)
        axarr[1].scatter(x=mean1, y=mean3, s=1, color=color2)
        axarr[1].plot([mean1-std1, mean1+std1],[mean3, mean3], color=color2, label=label)
        axarr[1].plot([mean1, mean1],[mean3-std3, mean3+std3], color=color2)


        # green vs blue

        axarr[2].set_xlabel("green")
        axarr[2].set_ylabel("blue")

        b=np.array((x2,x3)).T

        axarr[2].scatter(b[:, 0], b[:, 1], s=3, color=color1, alpha=1)
        axarr[2].scatter(x=mean1, y=mean3, s=1, color=color2)
        axarr[2].plot([mean2-std2, mean2+std2],[mean3, mean3], color=color2, label=label)
        axarr[2].plot([mean2, mean2],[mean3-std3, mean3+std3], color=color2)

        plt.legend()

        f.tight_layout()

        df+=1
        
    plt.savefig(save_name)

def crop(image, crop_size):
    h1 = (image.shape[0] - crop_size) /2
    h1 = int(h1)
    h2 = (image.shape[0] + crop_size) /2
    h2 = int(h2)

    w1 = (image.shape[1] - crop_size) /2
    w1 = int(w1)
    w2 = (image.shape[1] + crop_size) /2
    w2 = int(w2)
    cropped_image = image[h1:h2,w1:w2, :]
    return cropped_image

"""
only for parallelization with ray
"""
"""
import ray 
if not ray.is_initialized():
    ray.init()

@ray.remote
def make_stat(metadata = metadata, idx = 0):
    added = metadata.values[idx]
    image = io.imread(metadata.values[:,1].tolist()[idx])
    for ch in range(3):
        added[5+ch] = np.mean(image[:,:,ch])
        added[8+ch] = np.std(image[:,:,ch])
        added[11+ch] = np.max(image[:,:,ch])
        added[14+ch] = np.min(image[:,:,ch])
    return added
meta_ray = ray.put(metadata)
x =ray.get([make_stat.remote(ray.get(meta_ray), id) for id in tqdm.tqdm(range(len(metadata)))])
metadata.values[0:len(metadata)] = x[0:len(metadata)]
metadata.to_csv('metadata3.csv', index=False)
"""