import glob
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import h5py
import random

def dataset(cfgs, flag, trans):
    if cfgs['dataset'] == 'BSDS':
        dataset = BSDS_500(root=cfgs['dataset_path'], flag=flag, VOC=False, transform=trans)
    elif cfgs['dataset'] == 'BSDS-VOC':
        dataset = BSDS_500(root=cfgs['dataset_path'], flag=flag, VOC=True, transform=trans)
    elif cfgs['dataset'] == 'NYUD-image':
        dataset = NYUD(root=cfgs['dataset_path'], flag=flag, rgb=True, transform=trans)
    elif cfgs['dataset'] == 'NYUD-hha':
        dataset = NYUD(root=cfgs['dataset_path'], flag=flag, rgb=False, transform=trans)
    else:
        raise NameError
    return dataset


'''  
#######################################################################################
BSDS_500
#######################################################################################
''' 
def read_from_pair_txt(path, filename):
    pfile = open(os.path.join(path, filename))
    filenames = pfile.readlines()
    pfile.close()

    filenames = [f.strip() for f in filenames]
    filenames = [c.split(' ') for c in filenames]
    filenames = [(os.path.join(path, c[0]),
                  os.path.join(path, c[1])) for c in filenames]
    return filenames

class BSDS_500(Dataset):
    def __init__(self, root, flag='train', VOC=False, transform=None):
        if flag == 'train':
            if VOC:
                filenames = read_from_pair_txt(root['BSDS-VOC'], 'bsds_pascal_train_pair_s5.lst')
            else:
                filenames = read_from_pair_txt(root['BSDS'], 'train_pair.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            self.im_list = glob.glob(os.path.join(root['BSDS'], r'test/*.jpg'))
            self.gt_list = [path.split('/')[-1][:-4] for path in self.im_list]

        self.length = self.im_list.__len__()

        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label  = Image.fromarray(label.astype(np.float32) / 255.0)
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


'''  
#######################################################################################
NYUD-v2
#######################################################################################
''' 
class NYUD(Dataset):
    def __init__(self, root, flag='train', rgb=True, transform=None):
        if flag == 'train':
            if rgb:
                filenames = read_from_pair_txt(root['NYUD-V2'], 'image-train.lst')
            else:
                filenames = read_from_pair_txt(root['NYUD-V2'], 'hha-train.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            if rgb:
                self.im_list = glob.glob(os.path.join(root['NYUD-V2'], 'test/Images/*.png'))
            else:
                self.im_list = glob.glob(os.path.join(root['NYUD-V2'], 'test/HHA/*.png'))
            self.gt_list = [path.split('/')[-1][:-4] for path in self.im_list]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label  = Image.fromarray(label.astype(np.float32) / 255.0)
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])


        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
