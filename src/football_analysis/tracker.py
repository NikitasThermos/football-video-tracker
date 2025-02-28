import os

import numpy as np
import pandas as pd
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

def interpolate_ball_positions(ball): 
    ball_positions = [b[3] for b in ball]
    df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()

    ball_interpolated = [(frame_num, 1, 0, bbox) for frame_num, bbox in enumerate(df_ball_positions.to_numpy().tolist())]
    return ball_interpolated

def predict_tracks(frames):
    detections = get_detections(frames)

    tracker = sv.ByteTrack()

    cls_names = detections[0].names
    cls_names_inv = {v: k for k, v in cls_names.items()}

    player_id, referee_id, ball_id, goalkeeper_id = (
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

    player = np.dtype(
        [
            ("frame_num", np.int32),
            ("track_id", np.int32),
            ("cls_id", np.int32),
            ("bbox", np.float32, (4,)),
            ("team", np.int32),
            ("color", np.float32, (3,)),
            ("has_ball", np.bool_),
        ]
    )

    other = np.dtype(
        [
            ("frame_num", np.int32),
            ("track_id", np.int32),
            ("cls_id", np.int32),
            ("bbox", np.float32, (4,)),
        ]
    )

    player_detecctions = []
    referee_detections = []
    ball_detections = []

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
            if cls_id == player_id:
                bbox = np.array(bbox)
                player_detecctions.append(
                    (frame_num, track_id, cls_id, bbox, -1, np.array([0, 0, 255]), False)
                )
            elif cls_id == referee_id:
                bbox = np.array(bbox)
                referee_detections.append((frame_num, track_id, cls_id, bbox))

        for detection in original_detections:
            bbox, _, _, cls_id, *_ = detection
            if cls_id == ball_id:
                bbox = np.array(bbox)
                ball_detections.append((frame_num, 1, cls_id, bbox))
                break
        else: 
            ball_detections.append((frame_num, 1, ball_id, []))
        
        ball = interpolate_ball_positions(ball_detections)

    return (
        np.array(player_detecctions, dtype=player),
        np.array(referee_detections, dtype=other),
        np.array(ball, dtype=other),
    )


def get_object_tracks(frames, read_file=False, path=None):
    if read_file and path is not None and os.path.exists(path):
        players, referees, ball = (
            np.load(path + "players.npy"),
            np.load(path + "referee.npy"),
            np.load(path + "ball.npy"),
        )
        return players, referees, ball

    players, referees, ball = predict_tracks(frames)

    if path is not None:
        np.save(path + 'players.npy', players)
        np.save(path + 'referee.npy', referees)
        np.save(path + 'ball.npy', ball)

    return players, referees, ball
