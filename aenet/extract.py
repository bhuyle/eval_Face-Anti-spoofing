from PIL.Image import RASTERIZE
import cv2
import time
import logging
import json
import numpy as np
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)

import torch
from torchvision import transforms

from tsn_predict import pretrain
from models_aenet import AENet


def load_model_aenet():
    num_class = 2
    net = AENet(num_classes = num_class)
    checkpoint = torch.load('../aenet/ckpt_iter.pth.tar',map_location=torch.device('cpu'))
    pretrain(net,checkpoint['state_dict'])
    net.cpu()
    net.eval()
    return net
def extract_aenet(net,images):
    tr = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),        
    ])
    
    pred = []
    real_data = []
    # for i,image in enumerate(images):
    data = tr(images)
    real_data.append(data)
    ## eval
    data = torch.stack(real_data,dim=0)
    channel = 3
    input_var = data.view(-1, channel, data.size(2), data.size(3)).cpu()
    with torch.no_grad():
        rst = net(input_var).detach().reshape(-1,2)
    rst = torch.nn.functional.softmax(rst, dim=1).cpu().numpy().copy()
    rst = np.array(rst)
    for index in rst:
        pred.append(1 if np.argmax(index) else 0)
    return pred[0]
