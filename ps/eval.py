from dataload import DataLoader
from model import RevisitResNet50
import numpy as np
import time
from statistics import mean

loader = DataLoader('../data/test/', batch_size=1)
revisit_model = RevisitResNet50()
revisit_model.load_weights('./checkpoints/resnet50/cp.ckpt')
ths = 0.12
print('Loading done')
result1 = []
result2 = []
result = []
times = []

d = 0
for step, batch in enumerate(loader.dataset):
    start = time.time()
    pred = revisit_model(batch['image'], batch['label_8'], batch['label_4'], batch['label_2'], batch['label_1'], training=False)
    end = time.time()
    times.append(end-start)
    d+=1
    if d>300:
        break
print(mean(times))
print(times)
#     pred_ = [1 if i>ths else 0 for i in pred.numpy().reshape(batch['label'].shape[0])]
#     labels = [int(i) for i in batch['label'].numpy().reshape(batch['label'].shape[0])]
#     #   print(pred.numpy().reshape(batch['label'].shape[0]))
#     #   print(pred_)
#     #   print(labels)
#     result1.append(calculate_apcer(labels, pred_))
#     result2.append(calculate_bpcer(labels, pred_))
#     result.append(calculate_acc(labels, pred_))
# print("APCER:",np.array(result1).mean())
# print("BPCER:",np.array(result2).mean())
# print("Acc:",np.array(result).mean())

