import torch
import cv2
import numpy as np

import torchvision.transforms as transforms

def load_model_resnet():
    model = torch.load('../resnet50/resnet50_best.pt',map_location=torch.device('cpu')).cpu()
    model.eval()
    return model

def extract_resnet(model,images):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    tr = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),        
            transforms.Normalize(mean, std)
        ])
    images = tr(images).cpu()
    # print(images.shape)
    images = torch.tensor(np.expand_dims(images, axis=0)).cpu()

    outputs = model(images)
    _, pred = torch.max(outputs, 1)
    if pred[0] == 0:
        return 1
    else: 
        return 0