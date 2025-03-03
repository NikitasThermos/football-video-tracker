import cv2
import numpy as np


def get_perspective_transformer():
    court_width = 68
    court_length = 23.32

    pixel_vertices = np.array(
        [
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915],
        ]
    ).astype(np.float32)

    target_vertices = np.array(
        [
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width],
        ]
    ).astype(np.float32)

    perspective_transformer = cv2.getPerspectiveTransform(
        pixel_vertices, target_vertices
    )
    return perspective_transformer, pixel_vertices


def transform_point(point, perspective_transformer, pixel_vertices):
    p = (int(point[0]), int(point[1]))
    is_inside = cv2.pointPolygonTest(pixel_vertices, p, False) >= 0

    if not is_inside:
        return None

    reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)

    transform_point = cv2.perspectiveTransform(reshaped_point, perspective_transformer)
    return transform_point.reshape(-1, 2)


def transform_positions(players):
    perspective_transformer, pixel_vertices = get_perspective_transformer()
    positions = players["position"]
    for i, position in enumerate(positions):
        transformed_position = transform_point(
            position, perspective_transformer, pixel_vertices
        )
        if transformed_position is not None:
            players["position"][i] = transformed_position

    return players
