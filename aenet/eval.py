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

from calculator import calculate_bpcer,calculate_apcer,calculate_acc
from tsn_predict import pretrain
from models import AENet

LOCAL_ROOT = '/home/Databases/face_dataset/CelebA-Spoof/CelebA_Spoof/'
LOCAL_IMAGE_LIST_PATH = 'metas/intra_test/test_label.json'
with open(LOCAL_ROOT+LOCAL_IMAGE_LIST_PATH) as f:
    image_list = json.load(f)


def read_image(path):
    index,name = path.split('/')[-2:]
    img = cv2.imread(f'../../data/test/{index}/{name}')
    return img

def get_image(batch):
    
    Batch_size = batch
    n = 0
    final_image = []
    final_image_id = []
    for idx,image_id in enumerate(image_list):
        image = read_image(image_id)
        final_image.append(image)
        final_image_id.append(image_id)
        n += 1
        if n == Batch_size or idx == len(image_list) - 1:
            np_final_image_id = np.array(final_image_id)
            np_final_image = np.array(final_image)
            n = 0
            final_image = []
            final_image_id = []
            yield np_final_image_id, np_final_image
            # return np_final_image_id,np_final_image

def eval(dataloader):

    tr = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),        
    ])
    
    # detector = CelebASpoofDetector()
    num_class = 2
    net = AENet(num_classes = num_class)
    checkpoint = torch.load('./ckpt_iter.pth.tar')
    pretrain(net,checkpoint['state_dict'])

    net.cuda()
    net.eval()

    image_iter = get_image(batch=700)
    label,pred = [],[]
    times = []
    d = 0
    for path,images in image_iter:
        # images = image_iter
        ### OK
        # prob = detector.predict(images)
        # print('prob',prob.shape)

        real_data = []
        for i,image in enumerate(images):
            data = tr(image)
            real_data.append(data)
            tmp = image_list[path[i]][43]
            label.append(1 if tmp==1 else 0)
        
        ## eval
        start = time.time()
        data = torch.stack(real_data,dim=0)
        channel = 3
        input_var = data.view(-1, channel, data.size(2), data.size(3)).cuda()
        with torch.no_grad():
            rst = net(input_var).detach().reshape(-1,2)
            end = time.time()
        times.append(end-start)
        rst = torch.nn.functional.softmax(rst, dim=1).cpu().numpy().copy()
        rst = np.array(rst)
        for index in rst:
            pred.append(1 if np.argmax(index) else 0)
        d+=1
        print(f'{d*700}/{len(image_list)}')
    print(sum(times)/len(times))
    apcer = calculate_apcer(label,pred)
    bpcer = calculate_bpcer(label,pred)
    acc = calculate_acc(label,pred)
    print(f'APCER: {apcer}, BPCER: {bpcer}, Acc {acc}')
    out = 'Label Pred \n'
    for i,index in enumerate(pred):
        out += f'{label[i]} {index}\n' 
    with open('result.txt','w+') as f:
        f.write(out)
if __name__ == '__main__':
    tr = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),        
    ])
    # test_data = celebA_spoof(type='test', transform=tr)
    # testloader = DataLoader(test_data, batch_size=90, shuffle=False, num_workers=0)
    
    # eval(testloader)
    eval(1)
    # (a,b) = get_image(5)
    # print(a,image_list[a[0]][43])



