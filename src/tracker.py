import os
import pickle

import numpy as np
import supervision as sv
from ultralytics import YOLO


def get_detections(frames, model_path="model/best.pt"):
    model = YOLO(model_path)
    batch_size = 20
    detections = []
    for i in range(0, len(frames), batch_size):
        detections_batch = model.predict(frames[i : i + batch_size], conf=0.1)
        detections += detections_batch

    return detections


def predict_tracks(frames):
    detections = get_detections(frames)

    tracks = {
        "players": [],
        "referees": [],
        "ball": [],
    }

    tracker = sv.ByteTrack()

    cls_names = detections[0].names
    cls_names_inv = {v: k for k, v in cls_names.items()}

    player_id, referee_id, ball_id, goalkeeper_id = (
        cls_names_inv.get("player"),
        cls_names_inv.get("referee"),
        cls_names_inv.get("ball"),
        cls_names_inv.get("goalkeeper"),
    )

    for frame_num, detection in enumerate(detections):

        # convert to supervision detection format
        detection_supervision = sv.Detections.from_ultralytics(detection)

        # convert goalkeeper to player id
        detection_supervision.class_id = np.array(
            [
                player_id if cls_id == goalkeeper_id else cls_id
                for cls_id in detection_supervision.class_id
            ],
            dtype=np.int32,
        )

        # Track objects
        detection_with_tracks = tracker.update_with_detections(detection_supervision)

        frame_players = {}
        frame_referees = {}
        frame_ball = {}

        for frame_detection in detection_with_tracks:
            bbox, _, _, cls_id, track_id, *_ = frame_detection
            bbox = bbox.tolist()

            if cls_id == player_id:
                frame_players[track_id] = {"bbox": bbox}

            if cls_id == referee_id:
                frame_referees[track_id] = {"bbox": bbox}

        for frame_detection in detection_supervision:
            bbox, _, _, cls_id, *_ = frame_detection
            bbox = bbox.tolist()

            if cls_id == ball_id:
                frame_ball[1] = {"bbox": bbox}

        tracks["players"].append(frame_players)
        tracks["referees"].append(frame_referees)
        tracks["ball"].append(frame_ball)

    return tracks


def get_object_tracks(frames, read_from_stub=False, stub_path=None):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            tracks = pickle.load(f)
        return tracks

    tracks = predict_tracks(frames)

    if stub_path is not None:
        with open(stub_path, "wb") as f:
            pickle.dump(tracks, f)

    return tracks
