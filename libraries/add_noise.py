import os
from time import sleep
from glob import glob
import random
from tqdm import tqdm

import numpy as np
from imageio import imread
import pandas as pd
import matplotlib.pyplot as plt


import sklearn
from skimage import io as io
from skimage.util import *
from random import choice
from skimage import img_as_ubyte
from skimage import img_as_float

data_path = {
        "Ace_20": "/beegfs/desy/user/hailudaw/challenge/Datasets/Acevedo_20", # Acevedo_20 Dataset
        "Mat_19": "/beegfs/desy/user/hailudaw/challenge/Datasets/Matek_19", # Matek_19 Dataset
        "WBC1": "/beegfs/desy/user/hailudaw/challenge/Datasets/WBC1" # WBC1 dataset
    }

import ray
if not ray.is_initialized():
    ray.init()

def finding_classes(data_dir):
    """
    this function finds the folders in the root path and considers them
    as classes
    """
    classes = [folder for folder in sorted(os.listdir(data_dir)) if not folder.startswith('.') and not folder.startswith('_')]
    return classes


output_path = ''
noises = ['gaussian', 'localvar','poisson','salt', 'pepper', 's&p', 'speckle']

@ray.remote
def add_noise_and_save(image, noise = noises , save = False, output_path = output_path):
    noise = choice(noises)
    noisy_image = random_noise(io.imread(image), mode=noise, seed=None, clip=True)
    
    

    if save==False:
        return noisy_image.astype('float32')
    else:
        output_path = os.path.join(output_path, os.path.basename(image))
        io.imsave(output_path, img_as_ubyte(noisy_image))
        return output_path



for key in data_path.keys():
    print(key)
    folders = finding_classes(data_path[key])
    for folder in folders:
        print("********* ",folder,"************")
        tif_collection = io.imread_collection(((os.path.join(data_path[key], folder, '*.tiff'))))
        jpg_collection = io.imread_collection(((os.path.join(data_path[key], folder, '*.jpg'))))
        output_path = os.path.join(data_path[key]+'_noisy', folder)
        try:
            os.mkdir(output_path) 
        except OSError:  
            if len(tif_collection.files)>0:
                ray.get([add_noise_and_save.remote(image, noises, save=True, output_path= output_path) for image in tif_collection.files])
            if len(jpg_collection.files)>0:
                ray.get([add_noise_and_save.remote(image, noises, save=True, output_path= output_path) for image in jpg_collection.files])

