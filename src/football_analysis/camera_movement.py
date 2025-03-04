import os
import pickle

import cv2
import numpy as np
from bbox_utils import measure_distance, measure_xy_distance


def get_camera_movement(frames, read_from_stub=False, stub_path=None, min_distance=5):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            return pickle.load(f)

    first_frame_grayscale = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    mask_features = np.zeros_like(first_frame_grayscale)
    mask_features[:, 0:20] = 1
    mask_features[:, 900:1050] = 1

    features = {
        "maxCorners": 100,
        "qualityLevel": 0.3,
        "minDistance": 3,
        "blockSize": 7,
        "mask": mask_features,
    }

    camera_movement = [[0, 0]] * len(frames)
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    old_features = cv2.goodFeaturesToTrack(old_gray, **features)

    for frame_num, frame in enumerate(frames[1:]):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_features, _, _ = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            old_features,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        max_distance = 0
        camera_movement_x, camera_movement_y = 0, 0

        for i, (new, old) in enumerate(zip(new_features, old_features)):
            new_features_point = new.ravel()
            old_features_point = old.ravel()

            distance = measure_distance(new_features_point, old_features_point)
            if distance > max_distance:
                max_distance = distance
                camera_movement_x, camera_movement_y = measure_xy_distance(
                    old_features_point, new_features_point
                )

        if max_distance > min_distance:
            camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
            old_features = cv2.goodFeaturesToTrack(frame_gray, **features)

        old_gray = frame_gray.copy()

    if stub_path is not None:
        with open(stub_path, "wb") as f:
            pickle.dump(camera_movement, f)

    return camera_movement


def adjust_position(frames, players, path=None):
    camera_movement = get_camera_movement(frames, stub_path=path, read_from_stub=True)
    for num_frame in range(len(frames)):
        players["position"][players["frame_num"] == num_frame] -= camera_movement[
            num_frame
        ]

    return players
