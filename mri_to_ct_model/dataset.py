import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import nibabel as nib

def random_rot(img1,img2):
    k = np.random.randint(0, 3)
    img1 = np.rot90(img1, k+1)
    img2 = np.rot90(img2, k+1)
    return img1,img2

def random_flip(img1,img2):
    axis = np.random.randint(0, 2)
    img1 = np.flip(img1, axis=axis).copy()
    img2 = np.flip(img2, axis=axis).copy()
    return img1,img2

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']

        if random.random() > 0.5:
            lr, hr = random_rot(lr, hr)
        if random.random() > 0.5:
            lr, hr = random_flip(lr, hr)
        sample = {'lr': lr,'hr': hr}
        return sample


class Train_Data(Dataset):
    def __init__(self, train_mri_path, train_ct_path):       
        self.mri_files = sorted(glob.glob(train_mri_path + '/*.nii.gz'))
        self.len = len(self.mri_files)
        array = nib.load(self.mri_files[0]).get_fdata()
        self.h, self.w = array.shape[0], array.shape[1]
        self.ct_files = sorted(glob.glob(train_ct_path + '/*.nii.gz'))
        self.transform = transforms.Compose([RandomGenerator(output_size=[self.h, self.w])])
        
    def __getitem__(self, index):
        mri_file = self.mri_files[index]
        ct_file = self.ct_files[index]
        
        mri_image = nib.load(mri_file)
        ct_image = nib.load(ct_file)
        
        mri_data = mri_image.get_fdata()
        ct_data = ct_image.get_fdata()
        
        x = self.norm(mri_data)
        y = self.norm(ct_data)
        
        sample = {'mri': x, 'ct': y}
        if self.transform:
            sample = self.transform(sample)
            
        x, y = sample['mri'], sample['ct']

        xx = np.zeros((1, self.h, self.w))
        yy = np.zeros((1, self.h, self.w))
        
        xx[0,:,:] = x.copy()
        yy[0,:,:] = y.copy()
                  
        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)
        
        xx = xx.type(torch.FloatTensor)
        yy = yy.type(torch.FloatTensor)
        
        return xx, yy
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x


class Valid_Data(Dataset):
    def __init__(self, val_mri_path, val_ct_path):       
        self.mri_files = sorted(glob.glob(val_mri_path + '/*.nii.gz'))
        self.ct_files = sorted(glob.glob(val_ct_path + '/*.nii.gz'))
        array = nib.load(self.mri_files[0]).get_fdata()
        self.h, self.w = array.shape[0], array.shape[1]
        ####
        #####
        self.len = 27 #Will surely need to be changed in the future (but kept same as original value)
        ######
        
    def __getitem__(self, index):
        mri_file = self.mri_files[index * 5]
        ct_file = self.ct_files[index * 5]
        
        mri_image = nib.load(mri_file)
        ct_image = nib.load(ct_file)
        
        mri_data = mri_image.get_fdata()
        ct_data = ct_image.get_fdata()
        
        x = self.norm(mri_data)
        y = self.norm(ct_data)
        
        xx = np.zeros((1, self.h, self.w))
        yy = np.zeros((1, self.h, self.w))
        
        xx[0,:,:] = x.copy()
        yy[0,:,:] = y.copy()
                  
        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)
        
        xx = xx.type(torch.FloatTensor)
        yy = yy.type(torch.FloatTensor)
        
        return xx, yy

    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x


class Test_Data(Dataset):
    def __init__(self, test_mri_path, test_ct_path):       
        self.mri_files = sorted(glob.glob(test_mri_path + '/*.nii.gz'))
        self.ct_files = sorted(glob.glob(test_ct_path + '/*.nii.gz'))
        array = nib.load(self.mri_files[0]).get_fdata()
        self.h, self.w = array.shape[0], array.shape[1]
        self.len = len(self.lr_files)

        self.len = 135 ### SAME AS BEFORE : NEED TO BE CHANGED
        
    def __getitem__(self, index):
        mri_file = self.mri_files[index]
        ct_file = self.ct_files[index]
        
        mri_image = nib.load(mri_file)
        ct_image = nib.load(ct_file)
        
        mri_data = mri_image.get_fdata()
        ct_data = ct_image.get_fdata()
        
        x = self.norm(mri_data)
        y = self.norm(ct_data)
        
        xx = np.zeros((1, self.h, self.w))
        yy = np.zeros((1, self.h, self.w))
        
        xx[0,:,:] = x.copy()
        yy[0,:,:] = y.copy()
                  
        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)
        
        xx = xx.type(torch.FloatTensor)
        yy = yy.type(torch.FloatTensor)
        
        return xx, yy
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x
