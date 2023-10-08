import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import h5py
import torchvision


class Dataset(data.Dataset):
    def __init__(self, file_path, name):
        super(Dataset, self).__init__()

        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms

        # LMS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.lms_crops = MS_crop(ms)
        self.lms_crops = lms

        # PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        # self.pan_crops = PAN_crop(pan)
        self.pan_crops = pan

    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops

    def __len__(self):
        return 1
