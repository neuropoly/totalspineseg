#This script's objective is to convert given dataset into training and testing datasets

#to run it :python3 totalsegmentator-mri/mri_to_ct_model/preprocess_datasets.py --path-train-data ./data/GoldAtlas,./data/synthRAD2023  --path-out ./data/preprocessed_data --path-test-data ./data/ICSFUS --path-val-data ./data/ICSFUS

import argparse
import pathlib
from pathlib import Path
import json
import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import nibabel as nib
import numpy as np

# parse command line arguments
parser = argparse.ArgumentParser(description='Pre-processes datasets in training, validation and test folder.')
parser.add_argument('--path-train-data', required=True,
                    help='Path to the training dataset (Can contain multiple dataset seperated by a comma with no space)')
parser.add_argument('--path-val-data', help='Path to the validation dataset (Can contain multiple dataset seperated by a comma with no space)', required=True)
parser.add_argument('--path-test-data', help='Path to the testing dataset (Can contain multiple dataset seperated by a comma with no space)', required=True)
parser.add_argument('--path-out', help='Path to output folder', required=True)
args = parser.parse_args()

paths_train = args.path_train_data.split(',')
paths_val = args.path_val_data.split(',')
paths_test = args.path_test_data.split(',')
path_out = Path(args.path_out)


path_out_mri_Tr = Path(os.path.join(path_out, 'mri_Tr'))
path_out_ct_Tr = Path(os.path.join(path_out, 'ct_Tr'))
path_out_mri_val = Path(os.path.join(path_out, 'mri_val'))
path_out_ct_val = Path(os.path.join(path_out, 'ct_val'))
path_out_mri_Ts = Path(os.path.join(path_out, 'mri_Ts'))
path_out_ct_Ts = Path(os.path.join(path_out, 'ct_Ts'))

if __name__ == '__main__':

    nb_training_files=0
    nb_validation_files=0
    nb_testing_files=0

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_mri_Tr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_ct_Tr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_mri_val).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_ct_val).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_mri_Ts).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_ct_Ts).mkdir(parents=True, exist_ok=True)

    #Iterate over all training dataset
    for path_train in paths_train: 
        dataset_name= path_train.split('/')[-1]
        path_train = Path(path_train)
        
        # Extraction of all MRI images (taking only T1w and T2w)  
        MRI_files = sorted(list(path_train.rglob('*_T1w.nii.gz')) + list(path_train.rglob('*_T2w.nii.gz') ))
        MRI_imgs = [str(k) for k in MRI_files]

        #The image folders
        CT_files = sorted(list(path_train.rglob('*_CT.nii.gz')))
    
        #Iterate over each MRI image
        for mri_img in MRI_imgs:
            if('T1w' in mri_img):
                subject = mri_img.split('_T1w')[0]
                subject_modality = mri_img.split('.nii.gz')[0].split('/')[-1]
                modality = 'T1w'
            elif('T2w' in mri_img):
                subject = mri_img.split('_T2w')[0]
                subject_modality = mri_img.split('.nii.gz')[0].split('/')[-1]
                modality = 'T2w'

            #search for corresponding CT image and copy it to destination folder
            ct_img = subject+'_CT.nii.gz'
            if(ct_img in str(CT_files)):
                #copy CT image
                CT_image_name = subject.split(dataset_name)[-1].replace('/anat/','/').replace('/','_') + '_' + modality
                preprocessed_ct_img = os.path.join(path_out_ct_Tr,dataset_name + '_' + CT_image_name + '_CT.nii.gz')
                shutil.copyfile(ct_img, preprocessed_ct_img)
                #copy MRI image
                MRI_image_name = subject.split(dataset_name)[-1].replace('/anat/','/').replace('/','_') + '_' + modality
                preprocessed_mri_img = os.path.join(path_out_mri_Tr,dataset_name + '_' + MRI_image_name + '.nii.gz')
                shutil.copyfile(mri_img, preprocessed_mri_img)
                nb_training_files+=1
            
    #Iterate over all validation dataset
    for path_val in paths_val: 
        dataset_name= path_val.split('/')[-1]
        path_val = Path(path_val)
        
        # Extraction of all MRI images (taking only T1w and T2w)  
        MRI_files = sorted(list(path_val.rglob('*_T1w.nii.gz')) + list(path_val.rglob('*_T2w.nii.gz') ))
        MRI_imgs = [str(k) for k in MRI_files]

        #The image folders
        CT_files = sorted(list(path_val.rglob('*_CT.nii.gz')))
    
        #Iterate over each MRI image
        for mri_img in MRI_imgs:
            if('T1w' in mri_img):
                subject = mri_img.split('_T1w')[0]
                subject_modality = mri_img.split('.nii.gz')[0].split('/')[-1]
                modality = 'T1w'
            elif('T2w' in mri_img):
                subject = mri_img.split('_T2w')[0]
                subject_modality = mri_img.split('.nii.gz')[0].split('/')[-1]
                modality = 'T2w'

            #search for corresponding CT image and copy it to destination folder
            ct_img = subject+'_CT.nii.gz'
            if(ct_img in str(CT_files)):
                #copy CT image
                CT_image_name = subject.split(dataset_name)[-1].replace('/anat/','/').replace('/','_') + '_' + modality
                preprocessed_ct_img = os.path.join(path_out_ct_val,dataset_name + '_' + CT_image_name + '_CT.nii.gz')
                shutil.copyfile(ct_img, preprocessed_ct_img)
                #copy MRI image
                MRI_image_name = subject.split(dataset_name)[-1].replace('/anat/','/').replace('/','_') + '_' + modality
                preprocessed_mri_img = os.path.join(path_out_mri_val,dataset_name + '_' + MRI_image_name + '.nii.gz')
                shutil.copyfile(mri_img, preprocessed_mri_img)

                nb_validation_files+=1


    #Iterate over all testing dataset
    for path_test in paths_test: 
        dataset_name= path_test.split('/')[-1]
        path_test = Path(path_test)
        
        # Extraction of all MRI images (taking only T1w and T2w)  
        MRI_files = sorted(list(path_test.rglob('*_T1w.nii.gz')) + list(path_test.rglob('*_T2w.nii.gz') ))
        MRI_imgs = [str(k) for k in MRI_files]

        #The image folders
        CT_files = sorted(list(path_test.rglob('*_CT.nii.gz')))
    
        #Iterate over each MRI image
        for mri_img in MRI_imgs:
            if('T1w' in mri_img):
                subject = mri_img.split('_T1w')[0]
                subject_modality = mri_img.split('.nii.gz')[0].split('/')[-1]
                modality = 'T1w'
            elif('T2w' in mri_img):
                subject = mri_img.split('_T2w')[0]
                subject_modality = mri_img.split('.nii.gz')[0].split('/')[-1]
                modality = 'T2w'

            #search for corresponding CT image and copy it to destination folder
            ct_img = subject+'_CT.nii.gz'
            if(ct_img in str(CT_files)):
                #copy CT image
                CT_image_name = subject.split(dataset_name)[-1].replace('/anat/','/').replace('/','_') + '_' + modality
                preprocessed_ct_img = os.path.join(path_out_ct_Ts,dataset_name + '_' + CT_image_name + '_CT.nii.gz')
                shutil.copyfile(ct_img, preprocessed_ct_img)
                #copy MRI image
                MRI_image_name = subject.split(dataset_name)[-1].replace('/anat/','/').replace('/','_') + '_' + modality
                preprocessed_mri_img = os.path.join(path_out_mri_Ts,dataset_name + '_' + MRI_image_name + '.nii.gz')
                shutil.copyfile(mri_img, preprocessed_mri_img)

                nb_testing_files+=1
    print("Training files: ",nb_training_files)
    print("Validation files: ", nb_validation_files)
    print("Testing files: ", nb_testing_files)


    