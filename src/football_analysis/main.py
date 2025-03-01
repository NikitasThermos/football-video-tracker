from ball_assigner import assign_ball_to_players
from camera_movement import get_camera_movement
from drawing import draw_annotations
from team_assigning import assign_teams
from tracker import get_object_tracks
from video_utils import read_video, save_video


def main():
    video_frames = read_video("videos/input_videos/08fd33_4.mp4")
    players, referees, ball = get_object_tracks(
        video_frames, read_file=True, file_path="tracks/"
    )

    players = assign_teams(video_frames, players)
    players, team_ball_control = assign_ball_to_players(video_frames, players, ball)

    camera_movement = get_camera_movement(
        video_frames, stub_path="stubs/camera.pkl", read_from_stub=True
    )

    output_frames = draw_annotations(
        video_frames, players, referees, ball, team_ball_control, camera_movement
    )

    save_video(output_frames, "videos/output_videos/annotated_video.mp4")


if __name__ == "__main__":
    main()
