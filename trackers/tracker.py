from ultralytics import YOLO
import supervision as sv
import cv2
import sys
import pickle
import os
sys.path.append('../')
from utils import get_center_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1) 
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks        
        detections = self.detect_frames(frames)
        tracks = {"players":[], "referees":[], "ball":[]}
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
        detection_supervision = sv.Detections.from_ultralytics(detection)    
        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
        tracks["players"].append({})
        tracks["referees"].append({})
        tracks["ball"].append({})
        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]
        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks     

    def draw_ellipse(self, frame, read_from_stub=False, stub_path=None):
        x_center = get_center_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35*width)), angle=0.0, startAngle=45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)

    def draw_annotations(self, vid_frames, tracks):
        output_vid_frames = []
        for frame_num, frame in enumerate(vid_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            # draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
            output_vid_frames.append(frame)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))
                output_vid_frames.append(frame)
        return output_vid_frames        
           