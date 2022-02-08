import cv2
import sys
import time
import copy
import socket
sys.path.append('src')

import argparse
from tracker_v2 import Tracker, get_model_base_path

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()
args.threshold = None
args.max_threads = 1
args.faces = 5
args.discard_after = 10
args.scan_every = 3
args.silent = 0
args.model = 2
args.model_dir = './models'
args.gaze_tracking = 1
args.detection_threshold = 0.6
args.scan_retinaface = 0
args.max_feature_updates = 900
args.no_3d_adapt = 1
args.try_hard = 0


log = None
out = None
first = True
height = 0
width = 0
tracker = None
sock = None
total_tracking_time = 0.0
tracking_time = 0.0
tracking_frames = 0
frame_count = 0


cap = cv2.VideoCapture(0)


while True:
    _,frame = cap.read()
    attempt = 0
    need_reinit = 0
    frame_count += 1
    now = time.time()

    if first:
        first = False
        height, width, channels = frame.shape
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, 
            max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, 
            silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir,
            no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, 
            detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, 
            max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, 
            try_hard=args.try_hard == 1)

    ## Detect
    inference_start = time.perf_counter()
    faces = tracker.predict_bboxonly(frame)
    start = time.time()
    if len(faces) > 0:
        inference_time = (time.perf_counter() - inference_start)
        total_tracking_time += inference_time
        tracking_time += inference_time / len(faces)
        tracking_frames += 1
    packet = bytearray()
    detected = False
    for face_num, f in enumerate(faces):
        f = copy.copy(f)
        f.id += 0
        detected = True
        x,y,w,h = int(f.bbox[0]), int(f.bbox[1]),int(f.bbox[0]), int(f.bbox[1])
        frame = cv2.rectangle(frame,(x,y),(w,y), (0,0,255))
        frame = cv2.putText(frame, str(f.id), (int(f.bbox[0]), int(
                f.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255))
    cv2.imshow('OpenSeeFace Visualization', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break