import cv2
import numpy as np
from bbox_utils import get_bbox_width, get_center_of_bbox


def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox()
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )

    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    if track_id is not None:
        cv2.rectangle(
            frame,
            (int(x1_rect), int(y1_rect)),
            (int(x2_rect), int(y2_rect)),
            color,
            cv2.FILLED,
        )

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
    return frame


def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array(
        [
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ]
    )

    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame


def draw_annotations(video_frames, tracks):
    output_video_frames = []
    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()

        player_list = tracks[
            (tracks["cls_id"] == 2) & (tracks["frame_num"] == frame_num)
        ]
        ball_list = tracks[(tracks["cls_id"] == 0) & (tracks["frame_num"] == frame_num)]
        referee_list = tracks[
            (tracks["cls_id"] == 3) & (tracks["frame_num"] == frame_num)
        ]

        for player in player_list:
            frame = draw_ellipse(frame, player["bbox"], (0, 0, 255), player["track_id"])

        for referee in referee_list:
            frame = draw_ellipse(
                frame, referee["bbox"], (0, 0, 255), referee["track_id"]
            )

        for ball in ball_list:
            frame = draw_triangle(frame, ball["bbox"], (0, 255, 0))

        output_video_frames.append(frame)

    return output_video_frames
