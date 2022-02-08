from model import RevisitResNet50
import numpy as np
import time
from statistics import mean
import cv2
import tensorflow as tf


def load_model_ps():
    revisit_model = RevisitResNet50()
    revisit_model.load_weights('../ps/model/cp.ckpt')
    return revisit_model
ths = 0.12

def extract_ps(model,image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256,256))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = np.expand_dims(image,axis=0)

    label_8 = tf.ones_like(tf.random.uniform(shape=(8,8,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
    label_4 = tf.ones_like(tf.random.uniform(shape=(4,4,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
    label_2 = tf.ones_like(tf.random.uniform(shape=(2,2,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
    label_1 = tf.ones_like(tf.random.uniform(shape=(1,1,1), minval=0, maxval=1, dtype=tf.dtypes.float32))

    pred = model(image, label_8, label_4, label_2, label_1, training=False)
    if pred[0][0] > ths: return 0
    else:
        return 1
