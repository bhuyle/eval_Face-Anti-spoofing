from PIL import Image
import sys
import numpy as np
import torchvision
import torch

from models_aenet import AENet


# from detector import CelebASpoofDetector

def pretrain(model, state_dict):
    own_state = model.state_dict()

    for name, param in state_dict.items():
        realname = name.replace('module.','')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[realname].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(realname, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")
