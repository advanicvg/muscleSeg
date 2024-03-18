import os,glob
import json
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import pydicom
import cv2
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
import albumentations as albu

train_transform = Compose([
    albu.HorizontalFlip(p=0.5),
    albu.OneOf(
        [
            albu.RandomBrightness(limit=0.2, p=0.5),   
        ],
        p=0.7,
    ),
    albu.PadIfNeeded(512,512,border_mode=0,p=1),
    albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=5, shift_limit=0.1, p=0.9, border_mode=0),
    albu.CenterCrop(512,512,p=1),
    albu.Resize(512,512,1,p=1),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

valid_transform = Compose([
    albu.PadIfNeeded(512,512,border_mode=0,p=1),
    albu.CenterCrop(512,512,p=1),
    albu.Resize(512,512,1,p=1),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


class ImageFolder(data.Dataset):
    def __init__(self, root,image_size=512,mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.GT_paths = root[:-1]+'_GT/'
       
        if mode=='train':
            self.image_paths=glob.glob(root+'/*.dcm')
            np.save('train.list',self.image_paths)
        elif mode=='valid':
            self.image_paths=glob.glob(root+'/*.dcm')
            np.save('valid.list',self.image_paths)
        else:
            self.image_paths=glob.glob(root+'/*.dcm')
        
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
        
        self.label_name_to_value = {}
        with open(os.path.join(os.path.dirname(os.path.dirname(self.root)), "classes.txt")) as f:
            value = 0
            for line in f:
                self.label_name_to_value[line[:-1]] = value
                value += 1
        if mode=='train':
            self.aug=train_transform
        else:
            self.aug=valid_transform

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        ds = pydicom.read_file(image_path,force=True)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = ds.pixel_array

        intercept = ds.RescaleIntercept
        slope = ds.RescaleSlope
        img = slope*img+intercept
        img[img<-190]=-190 #np.nan
        img[img>150]=150 #np.nan
        img_ori = img
        if np.nanmin(img_ori) < 0:
            img_ori += abs(np.nanmin(img_ori))+1
        else:
            img_ori -= abs(np.nanmin(img_ori))+1
        img_ori *= 255/np.nanmax(img_ori)
        img_ori=img_ori.astype('uint8')
        img_ori=cv2.cvtColor(img_ori, cv2.COLOR_GRAY2BGR)

        dir_type=os.path.dirname(image_path)
        if 'train' in dir_type.split('/')[-1]:
            GT_path = image_path.replace('train','train_GT').replace('.dcm','.npy')
        elif dir_type.split('/')[-1] == 'valid':
            GT_path = image_path.replace('valid','valid_GT').replace('.dcm','.npy')
        else:
            print('Error - folder name')
            return
        gt_label = np.load(GT_path)
    
    
        aug=self.aug(image=img_ori, mask=gt_label)
        image=aug['image']
        
        gt_label=aug['mask']
    
        gt_label = torch.tensor(np.array(gt_label), dtype=torch.int64)
        gt_label = torch.nn.functional.one_hot(gt_label, 7).to(torch.float).permute(2,0,1)#[:6,:,:] # #if loss is crossentropy 

        image=torch.tensor(image).permute(2,0,1)

        return image, gt_label, image_path.split('.dcm')[0]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    print(image_path)
    print(augmentation_prob)
    dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    print(batch_size)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                 # sampler=sampler,
                                  num_workers=num_workers)
    return data_loader


