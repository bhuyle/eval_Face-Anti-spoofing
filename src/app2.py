from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import time
import socket
import copy
import sys
sys.path.append('src')
sys.path.append('..')

sys.path.append('../lbp')
sys.path.append('../aenet')
sys.path.append('../resnet50')
sys.path.append('../ssdg')
sys.path.append('../ps')



from lbp.extract import extract_lbp, load_model_lbp
from resnet50.extract import load_model_resnet,extract_resnet
from aenet.extract import load_model_aenet,extract_aenet
from ps.extract import load_model_ps,extract_ps
from ssdg.extract import load_model_ssdg,extract_ssdg


from tracker_v2 import Tracker
import argparse

async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('demo.html')

choice = 3
model_lbp, model_aenet, model_resnet, model_ps, model_ssdg = load_model_lbp(),\
    load_model_aenet(),load_model_resnet(),load_model_ps(),load_model_ssdg()


@app.route('/ajax', methods = ['POST'])
def ajax_request():
    id = request.args.get("method")
    global choice
    choice = int(id)
    print('choice',choice)
    
    return 'OK'



def gen_frame():

    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.threshold = None
    args.max_threads = 1
    args.faces = 5
    args.discard_after = 10
    args.scan_every = 3
    args.silent = 0
    args.model = 1
    args.model_dir = './model_face'
    args.gaze_tracking = 1
    args.detection_threshold = 0.6
    args.scan_retinaface = 0
    args.max_feature_updates = 900
    args.no_3d_adapt = 1
    args.try_hard = 0


    first = True
    height = 0
    width = 0
    tracker = None
    total_tracking_time = 0.0
    tracking_time = 0.0
    tracking_frames = 0
    frame_count = 0

    ### Load model face anti-spoofing
    cap = cv2.VideoCapture(0)

    while True:
        global choice

        _,frame = cap.read()
        frame = cv2.resize(frame,(640,480))
        frame_count += 1

        if first:
            first = False
            height, width, _ = frame.shape
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
        if len(faces) > 0:
            inference_time = (time.perf_counter() - inference_start)
            total_tracking_time += inference_time
            tracking_time += inference_time / len(faces)
            tracking_frames += 1
        start = time.time()

        for face_num, f in enumerate(faces):
            f = copy.copy(f)
            f.id += 0
            box = f.bbox

            x1 = int(box[0]) - int(box[2]*0.2) 
            y1 = int(box[1]) - int(box[3]*0.7)
            x2 = int(box[2]*1.3) + x1
            y2 = int(box[3]*1.8) + y1
            face = frame[y1:y2,x1:x2]
            
            try:
                start = time.time()
                if choice == 1:
                    result = extract_lbp(model_lbp,face)
                elif choice == 2:
                    result = extract_resnet(model_resnet,face)
                elif choice == 3:
                    result = extract_aenet(model_aenet,face)
                elif choice == 4:    
                    result = extract_ps(model_ps,face)
                elif choice == 5:    
                    result = extract_ssdg(model_ssdg,face)
                elif choice == 5:    
                    result = extract_resnet(model_resnet,face)
                    if result == 1:
                        result = 0
                if result:
                    text = 'Spoof'
                    color = (0,0,255)
                else:
                    text = 'Live'
                    color = (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                frame = cv2.putText(frame, str(text), (x1,y1-10), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color)
            except:
                continue
        end = time.time()
        try:
            fps = int(1/(end-start))
        except:
            fps = 5
        frame = cv2.putText(frame, 'FPS:'+str(fps), (20,35), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, (246,0,255))
        yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # socketio.run(app, debug=True)
    app.run(debug=True)
