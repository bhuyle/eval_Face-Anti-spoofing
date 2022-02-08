import cv2
import json
import numpy as np
import pandas as pd
import random
random.seed(10)

import torch
from torch.utils.data import Dataset, DataLoader

class celebA_spoof(Dataset):

    def __init__(self,type, transform=None):

        # json_path_train = '/home/Databases/face_dataset/CelebA-Spoof/CelebA_Spoof/metas/intra_test/train_label.json'
        # json_path_test = '/home/Databases/face_dataset/CelebA-Spoof/CelebA_Spoof/metas/intra_test/test_label.json'
        # LOCAL_IMAGE_LIST_PATH = json_path_train

        # with open(json_path_train) as f:
        #     self.image_list = json.load(f)
        #     self.image_list = list(self.image_list.keys())
        # random.shuffle(self.image_list)
        # train_len = len(self.image_list)
        # self.image_train, self.image_val = self.image_list[:int(train_len*0.8)],self.image_list[int(train_len*0.8):]

        # with open(json_path_test) as f:
        #     self.image_test = json.load(f)
        #     self.image_test = list(self.image_test.keys())
        # print("Total images train:",len(self.image_train))
        # print("Total images val:",len(self.image_val))
        # print("Total images test:",len(self.image_test))

        self.image_test = ['face.jpg']
        self.type = type
        self.transform = transform
    def __len__(self):
        if self.type=='train':
            return len(self.image_train)
        elif self.type=='val':
            return len(self.image_val)
        else:
            return len(self.image_test)

    def __getitem__(self, idx):
        # if self.type=='train':
        #     (index,name) = self.image_train[idx].split('/')[-2:]
        #     root = '../data/train/'
        # elif self.type=='val':
        #     index,name = self.image_val[idx].split('/')[-2:]
        #     root = '../data/train/'
        # else:
        #     index,name = self.image_test[idx].split('/')[-2:]
        #     root = '../data/test/'

        # if index == 'spoof':
        #     label = 0
        # else: label = 1
        # img = cv2.imread(f'{root}{index}/{name}')
        img = cv2.imread(self.image_test[idx])
        if self.transform:
            img = self.transform(img)
        return 1,img


