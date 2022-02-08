import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

import cv2
import numpy as np
import tqdm
from load import celebA_spoof
import time

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),        
        transforms.Normalize(mean, std)
    ])

test_data = celebA_spoof(type='test', transform=tr)

testloader = DataLoader(
test_data, batch_size=1, shuffle=False, num_workers=0)
for labels,images in tqdm.tqdm(testloader):
    print(images.shape)
    break
# model = torch.load('./output/resnet50.pt').cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.fc.parameters(),lr=0.001)

# model.eval()

# times = []
# d = 0
# correct_val = 0
# n_samples = 0
# epoch_loss_val = 0
# for labels,images in tqdm.tqdm(testloader): 
#     images = images.cuda()
#     start = time.time()
#     outputs = model(images)
#     end = time.time()
#     times.append(end-start)
#     d+=1
#     if d > 300:
#         break
# print(sum(times)/len(times))

    
