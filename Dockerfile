FROM ubuntu:18.04

WORKDIR /home

RUN apt update && \
    apt upgrade -y && \
    apt-get install ffmpeg libsm6 libxext6 -y &&\
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y python3-pip && \
    python3 -m pip install --upgrade pip && \
    pip install torch==1.6.0 torchvision==0.7.0 && \
    pip install tensorflow===2.4.1 && \
    pip install scikit-learn==0.20.4 scikit-image==0.17.2 && \
    pip install opencv-python==4.5.5 && \
    pip install flask==2.0.2 flask-socketio==5.1.1 tqdm onnxruntime==1.10.0 gdown





