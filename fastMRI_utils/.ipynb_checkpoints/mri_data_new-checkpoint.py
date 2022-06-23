"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import h5py
from torch.utils.data import Dataset

import xml.etree.ElementTree as ET

import os
import numpy as np

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)

def MiddleSliceData(root, transform, challenge, sample_rate=1, use_mid_slices=True, num_mid_slices=1, fat_supress=False, strength_3T=True, restrict_size=False):
    return SelectiveSliceData(root, transform, challenge, sample_rate=sample_rate, use_mid_slices=use_mid_slices, num_mid_slices=num_mid_slices, fat_supress=fat_supress, strength_3T=strength_3T, restrict_size=restrict_size)

def MiddleSliceData_Singlecoil(root, transform, challenge='singlecoil', sample_rate=1, use_mid_slices=True, num_mid_slices=1, fat_supress=False, strength_3T=True, restrict_size=False):
    return SelectiveSliceData_Singlecoil(root, transform, challenge = challenge , sample_rate=sample_rate, use_mid_slices=use_mid_slices, num_mid_slices=num_mid_slices, fat_supress=fat_supress, strength_3T=strength_3T, restrict_size=restrict_size)

class SelectiveSliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, use_mid_slices=False, num_mid_slices=1, fat_supress=None, strength_3T=None, restrict_size=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []

        files = list(pathlib.Path(root).iterdir())
        # remove files with wrong modality or scanner
        keep_files = []
        for fname in sorted(files):
            with h5py.File(fname, 'r') as data:
                
                if (fat_supress is None) or ((data.attrs['acquisition'] == 'CORPDFS_FBK')==fat_supress):
                    scanner_str = findScannerStrength(data['ismrmrd_header'][()])
                    three_tesla_scanner = scanner_str > 2.2
                    if (strength_3T is None) or (three_tesla_scanner == strength_3T):
                        keep_files.append(fname)
#                         print(fname)

        files = keep_files
        
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            
            if restrict_size and ((kspace.shape[1]!=640) or (kspace.shape[2]!=368)):
                continue # skip non uniform sized images
            num_slices = kspace.shape[0]
            if use_mid_slices:
                start_idx = round((num_slices - num_mid_slices)/2)
                end_idx = start_idx + num_mid_slices
                self.examples += [(fname, slice) for slice in range(start_idx, end_idx)]
            else:
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)

# Helper function to get scanner strength from XML header info
def findScannerStrength(xml_header_str):
    root = ET.fromstring(xml_header_str)
    for child in root:
        if 'acquisitionSystemInformation' in child.tag:
            for deep_child in child:
                if 'systemFieldStrength_T' in deep_child.tag:
                    return float(deep_child.text)

                
class SelectiveSliceData_Singlecoil(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge = 'singlecoil', sample_rate=1, use_mid_slices=False, num_mid_slices=1, fat_supress=None, strength_3T=None, restrict_size=False):

        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []

        files = list(pathlib.Path(root).iterdir())
        # remove files with wrong modality or scanner
        keep_files = []
        for fname in sorted(files):
            with h5py.File(fname, 'r') as data:
                
                if (fat_supress is None) or ((data.attrs['acquisition'] == 'CORPDFS_FBK')==fat_supress):
                    scanner_str = findScannerStrength(data['ismrmrd_header'][()])
                    three_tesla_scanner = scanner_str > 2.2
                    if (strength_3T is None) or (three_tesla_scanner == strength_3T):
                        keep_files.append(fname)
#                         print(fname)

        files = keep_files
        
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            
            if restrict_size and ((kspace.shape[1]!=640) or (kspace.shape[2]!=368)):
                continue # skip non uniform sized images
            num_slices = kspace.shape[0]
            if use_mid_slices:
                start_idx = round((num_slices - num_mid_slices)/2)
                end_idx = start_idx + num_mid_slices
                self.examples += [(fname, slice) for slice in range(start_idx, end_idx)]
            else:
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace)

        
class SelectiveSliceData_Train(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, use_top_slices=True, number_of_top_slices=8, fat_supress=None, strength_3T=None, restrict_size=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        files = list(pathlib.Path(root).iterdir())

        # remove files with wrong modality or scanner
        keep_files = []
        f = sorted(files)

#         print('total volume files in training data set: ', len(f) )
        
        for fname in f[0:]: 
            kspace = h5py.File(fname, 'r')['kspace']
            with h5py.File(fname, 'r') as data:
                if (data.attrs['acquisition'] == 'CORPD_FBK'):
                    scanner_str = findScannerStrength(data['ismrmrd_header'][()])
                    if (scanner_str > 2.2):
                        if kspace.shape[1]>=8:
                            keep_files.append(fname)
        
        
        files = keep_files
        
#         print('total volume files we are using for training: ',len(files))

        random.seed(1000)
        np.random.seed(1000)

        random.shuffle(files)

        num_files = (len(files))
 
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):

            kspace = h5py.File(fname, 'r')['kspace']
            
            if kspace.shape[-1] < 368:
                continue
            else:
#                 if restrict_size and ((kspace.shape[1]!=640) or (kspace.shape[2]!=368)):
#                     continue # skip non uniform sized images
                num_slices = kspace.shape[0]
                if use_top_slices:
                    start_idx = round((num_slices - 8)/2)
                    end_idx = start_idx + number_of_top_slices
                    self.examples += [(fname, slice) for slice in range(start_idx, end_idx)]
                else:
                    self.examples += [(fname, slice) for slice in range(num_slices)]
                    
#             print("number of total images used for training: ",len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
#         print('HI')
#         print(len(self.examples))
        fname, slice = self.examples[i]
#         print('fname')
#         print(fname)
#         print('slice')
#         print(slice)
#         print('hello')
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace)

class SelectiveSliceData_Val(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, use_top_slices=True, number_of_top_slices=8, fat_supress=None, strength_3T=None, restrict_size=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        files = list(pathlib.Path(root).iterdir())

        # remove files with wrong modality or scanner
        keep_files = []
        f = sorted(files)

#         print('total volume files in validation data set: ', len(f) )
        
        for fname in f[0:]:
            kspace = h5py.File(fname, 'r')['kspace']
            with h5py.File(fname, 'r') as data:
                if (data.attrs['acquisition'] == 'CORPD_FBK'):
                    scanner_str = findScannerStrength(data['ismrmrd_header'][()])
                    if (scanner_str > 2.2):
                        if kspace.shape[1]>=8:
                            keep_files.append(fname)
        
        
        files = keep_files
        
#         print('total volume files we are using for validation: ',len(files))

        random.seed(1000)
        np.random.seed(1000)

        random.shuffle(files)

        num_files = (len(files))
 
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):

            kspace = h5py.File(fname, 'r')['kspace']
            
            if kspace.shape[-1] < 368:
                continue
            else:
#                 if restrict_size and ((kspace.shape[1]!=640) or (kspace.shape[2]!=368)):
#                     continue # skip non uniform sized images
                num_slices = kspace.shape[0]
                if use_top_slices:
                    start_idx = round((num_slices - 8)/2)
                    end_idx = start_idx + number_of_top_slices
                    self.examples += [(fname, slice) for slice in range(start_idx, end_idx)]
                else:
                    self.examples += [(fname, slice) for slice in range(num_slices)]
                    
#         print("number of total images used for validation: ",len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
#         print('HI')
#         print(len(self.examples))
        fname, slice = self.examples[i]
#         print('fname')
#         print(fname)
#         print('slice')
#         print(slice)
#         print('hello')
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace)

        
        

        
class SelectiveSliceData_Train_Brain(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, use_top_slices=True, number_of_top_slices=8, fat_supress=None, strength_3T=None, restrict_size=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        files = list(pathlib.Path(root).iterdir())

        # remove files with wrong modality or scanner
        keep_files = []
        f = sorted(files)

#         print('total volume files in training data set: ', len(f) )
        
        for fname in f[1:]: # in the validation data, the 0th file has some wierd name and it gives a file not found error. Hence we start from 1
            kspace = h5py.File(fname, 'r')['kspace']
            with h5py.File(fname, 'r') as data:
                if (data.attrs['acquisition'] == 'AXT2'):
                    scanner_str = findScannerStrength(data['ismrmrd_header'][()])
                    if (scanner_str > 2.2):
                        if kspace.shape[1]>=8:
                            keep_files.append(fname)
        
        
        files = keep_files
        
#         print('total volume files we are using for training: ',len(files))

        random.seed(1000)
        np.random.seed(1000)

        random.shuffle(files)

        num_files = (len(files))
 
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):

            kspace = h5py.File(fname, 'r')['kspace']
            
            if kspace.shape[-1] < 384:
                continue
            else:
#                 if restrict_size and ((kspace.shape[1]!=640) or (kspace.shape[2]!=368)):
#                     continue # skip non uniform sized images
                num_slices = kspace.shape[0]
                if use_top_slices:
                    start_idx = 0
                    end_idx = start_idx + number_of_top_slices
                    self.examples += [(fname, slice) for slice in range(start_idx, end_idx)]
                else:
                    self.examples += [(fname, slice) for slice in range(num_slices)]
                    
#             print("number of total images used for training: ",len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
#         print('HI')
#         print(len(self.examples))
        fname, slice = self.examples[i]
#         print('fname')
#         print(fname)
#         print('slice')
#         print(slice)
#         print('hello')
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace)
        
        


class SelectiveSliceData_Val_Brain(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, use_top_slices=True, number_of_top_slices=8, fat_supress=None, strength_3T=None, restrict_size=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        files = list(pathlib.Path(root).iterdir())

        # remove files with wrong modality or scanner
        keep_files = []
        f = sorted(files)

#         print('total volume files in validation data set: ', len(f) )
        
        for fname in f[1:]: # in the validation data, the 0th file has some wierd name and it gives a file not found error. Hence we start from 1
            kspace = h5py.File(fname, 'r')['kspace']
            with h5py.File(fname, 'r') as data:
                if (data.attrs['acquisition'] == 'AXT2'):
                    scanner_str = findScannerStrength(data['ismrmrd_header'][()])
                    if (scanner_str > 2.2):
                        if kspace.shape[1]>=8:
                            keep_files.append(fname)
        
        
        files = keep_files
        
#         print('total volume files we are using for validation: ',len(files))

        random.seed(1000)
        np.random.seed(1000)

        random.shuffle(files)

        num_files = (len(files))
 
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            
            kspace = h5py.File(fname, 'r')['kspace']
            
            if kspace.shape[-1] < 384:
                continue
            else:
                print(fname)
#                 if restrict_size and ((kspace.shape[1]!=640) or (kspace.shape[2]!=368)):
#                     continue # skip non uniform sized images
                num_slices = kspace.shape[0]
                if use_top_slices:
                    start_idx = 0
                    end_idx = start_idx + number_of_top_slices
                    self.examples += [(fname, slice) for slice in range(start_idx, end_idx)]
                else:
                    self.examples += [(fname, slice) for slice in range(num_slices)]
                    
#         print("number of total images used for validation: ",len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
#         print('HI')
#         print(len(self.examples))
        fname, slice = self.examples[i]
#         print('fname')
#         print(fname)
#         print('slice')
#         print(slice)
#         print('hello')
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace)