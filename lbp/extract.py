import numpy as np
from skimage import feature as skif
import cv2
import os
import tqdm
import random
from time import time
# from sklearn.svm import SVC
from sklearn.externals import joblib
# import joblib
# import joblib

# from metric import metric,calculate_apcer,calculate_bpcer,calculate_acc


def lbp_histogram(image,P=8,R=1,method = 'nri_uniform'):
    ''' 
    image: shape is N*M 
    '''
    lbp = skif.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
    # cv2.imwrite("lbp.png",lbp)
    max_bins = int(lbp.max() + 1) # max_bins is related P
    hist,_= np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return hist

def load_model_lbp():
    model = joblib.load("../lbp/model.m")
    return model
def extract_lbp(model,image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_h = lbp_histogram(image[:,:,0]) # y channel
    cb_h = lbp_histogram(image[:,:,1]) # cb channel
    cr_h = lbp_histogram(image[:,:,2]) # cr channel
    feature = np.expand_dims(np.concatenate((y_h,cb_h,cr_h)),axis=0)
    predict_proba = model.predict_proba(feature)
    if predict_proba[0][0] > 0.5:
        return 1
    else:
        return 0

if __name__ == "__main__":
    model = load_model_lbp()
    import glob
    paths = glob.glob('../../show/1534/live/*')
    print(paths)
    for path in paths:
        img = cv2.imread(path)
        print(extract_lbp(model,img))


