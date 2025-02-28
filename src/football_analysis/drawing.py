import cv2
import numpy as np
from bbox_utils import get_bbox_width, get_center_of_bbox


def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
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


def draw_team_ball_control(frame, team_ball_control):
    overlay = frame.copy()
    cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    team_1_percentage = team_ball_control.count(0) / len(team_ball_control) * 100
    team_2_percentage = team_ball_control.count(1) / len(team_ball_control) * 100

    cv2.putText(
        frame,
        f"Team 1: {team_1_percentage:.2f}%",
        (1400, 900),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        3,
    )

    cv2.putText(
        frame,
        f"Team 2: {team_2_percentage:.2f}%",
        (1400, 950),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        3,
    )

    return frame


def draw_annotations(frames, players_list, referees_list, ball_list, team_ball_control):
    output_video_frames = []
    for frame_num, frame in enumerate(frames):
        frame = frame.copy()

        frame_players = players_list[players_list["frame_num"] == frame_num]
        frame_referees = referees_list[referees_list["frame_num"] == frame_num]
        frame_ball = ball_list[ball_list["frame_num"] == frame_num]

        for player in frame_players:
            frame = draw_ellipse(
                frame, player["bbox"], player["color"].tolist(), player["track_id"]
            )
            if player["has_ball"]:
                frame = draw_triangle(frame, player["bbox"], (0, 0, 255))

        for referee in frame_referees:
            frame = draw_ellipse(
                frame, referee["bbox"], (0, 255, 255), referee["track_id"]
            )

        for ball in frame_ball:
            frame = draw_ellipse(frame, ball["bbox"], (0, 255, 0))

        frame = draw_team_ball_control(frame, team_ball_control[: frame_num + 1])

        output_video_frames.append(frame)

    return output_video_frames
