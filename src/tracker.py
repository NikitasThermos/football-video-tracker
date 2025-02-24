import os

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
    tracker = sv.ByteTracker()
    cls_names = detections[0].names
    cls_names_inv = {v: k for k, v in cls_names.items()}

    player_id, _, ball_id, goalkeeper_id = (
        cls_names_inv.get("player"),
        cls_names_inv.get("referee"),
        cls_names_inv.get("ball"),
        cls_names_inv.get("goalkeeper"),
    )

    sv_detections = [
        sv.Detections.from_ultralytics(detection) for detection in detections
    ]
    detection_with_tracks = (
        tracker.update_with_detections(sv_detection) for sv_detection in sv_detections
    )

    dtype = np.dtype(
        [
            ("frame_num", np.int32),
            ("track_id", np.int32),
            ("cls_id", np.int32),
            ("bbox", np.float32, (4,)),
        ]
    )
    detection_list = []

    for frame_num, (tracked_detections, original_detections) in enumerate(
        zip(detection_with_tracks, sv_detections)
    ):

        tracked_detections.class_id = np.array(
            [
                player_id if cls_id == goalkeeper_id else cls_id
                for cls_id in tracked_detections.class_id
            ],
            dtype=np.int32,
        )

        for detection in tracked_detections:
            bbox, _, _, cls_id, track_id, *_ = detection
            if cls_id != ball_id:
                bbox = np.array(bbox)
                detection_list.append((frame_num, track_id, cls_id, bbox))

        for detection in original_detections:
            bbox, _, _, cls_id, *_ = detection
            if cls_id == ball_id:
                bbox = np.array(bbox)
                detection_list.append((frame_num, 1, cls_id, bbox))

    return np.array(detection_list, dtype=dtype)


def get_object_tracks(frames, read_file=False, file_path=None):
    if read_file and file_path is not None and os.path.exists(file_path):
        return np.load(file_path)

    tracks = predict_tracks(frames)

    if file_path is not None:
        np.save(file_path, tracks)

    return tracks
