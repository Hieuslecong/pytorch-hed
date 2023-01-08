# dataproc.py: Dataset loader classes for BSDS
# Author: Nishanth Koganti
# Date: 2017/10/11

# Source: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# Issues:
# Merge TrainDataset and TestDataset classes

# import libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
from torch._C import import_ir_module
import glob
# import torch modules
import torch
from torch.utils.data import Dataset

# BSDS dataset class for training data
class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir, 
                 transform=None, target_transform=None):        
        self.rootDir = rootDir
        self.transform = transform
        self.targetTransform = target_transform
        self.frame = glob.glob(r'/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/train/image/*.jpg')
        self.label = glob.glob(r'/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/train/label/*.png')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # input and target images
        name_img =  (os.path.split(self.frame[idx])[-1]).split(".")[0]
        name_mask = name_img.split("img")[0] + 'msk'+ name_img.split("img")[1]
        inputName = '/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/train/image/'+(os.path.split(self.frame[idx])[-1]).split(".")[0]+'.jpg'
        #inputName = os.path.join(self.frame[idx])
        targetName = '/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/train/label/'+name_mask +'.png'#os.path.join(self.label[idx])

        # process the images
        inputImage = Image.open(inputName).convert('RGB')
        inputImage = inputImage.resize((512, 512), Image.ANTIALIAS)
        if self.transform is not None:
            inputImage = self.transform(inputImage)

        targetImage = Image.open(targetName).convert('L')
        targetImage = targetImage.resize((512, 512), Image.ANTIALIAS)
        if self.targetTransform is not None:
            targetImage = self.targetTransform(targetImage)

        return inputImage, targetImage
    
# dataset class for test dataset
class TestDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = glob.glob(r'/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/train/image/*.jpg')
        self.label = glob.glob(r'/content/drive/Shareddrives/pix2pixHD/model_crack_detection/data_set/train/label/*.png')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        fname = self.frame[idx, 0]
        inputName = os.path.join(fname)

        # process the images
        inputImage = np.asarray(Image.open(inputName).convert('RGB'))
        inputImage = inputImage.resize((512, 512), Image.ANTIALIAS)
        if self.transform is not None:
            inputImage = self.transform(inputImage)

        return inputImage, fname