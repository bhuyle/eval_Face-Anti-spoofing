import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T
from models.DGFAS import DG_model

thresh = 0.9
def load_model_ssdg():
    # model = DG_model('resnet18')
    # model.load_state_dict('./model.pt').cpu()

    model = torch.load('../ssdg/model.pt',map_location=torch.device('cpu')).cpu()
    model = model.module.cpu()
    return model

transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
def extract_ssdg(model,image):
    with torch.no_grad():
        image = cv2.resize(image,(224,224))
        input = transforms(image)
        input = Variable(input)
        input = torch.tensor(np.expand_dims(input, axis=0)).cpu()

        cls_out, _ = model(input, True)
        prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
        if prob > thresh:
            return 0
        else:
            return 1
        # return prob
        