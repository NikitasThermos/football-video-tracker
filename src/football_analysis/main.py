from ball_assigner import assign_ball_to_players
from camera_movement import adjust_position
from drawing import draw_annotations
from speed_estimator import add_speed
from team_assigning import assign_teams
from tracker import get_object_tracks
from video_utils import read_video, save_video
from view_transformer import transform_positions


def main():
    video_frames = read_video("videos/input_videos/08fd33_4.mp4")
    players, referees, ball = get_object_tracks(
        video_frames, read_file=True, file_path="tracks/"
    )

    players = assign_teams(video_frames, players)
    players = adjust_position(video_frames, players)
    players = transform_positions(players)
    players = add_speed(players, len(video_frames))

    players, team_ball_control = assign_ball_to_players(video_frames, players, ball)

    output_frames = draw_annotations(
        video_frames, players, referees, ball, team_ball_control)

    save_video(output_frames, "videos/output_videos/annotated_video.mp4")


if __name__ == "__main__":
    main()
