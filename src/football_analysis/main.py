from drawing import draw_annotations
from tracker import get_object_tracks
from video_utils import read_video, save_video


def main():
    video_frames = read_video("videos/input_videos/08fd33_4.mp4")
    players, referees, ball = get_object_tracks(
        video_frames, read_file=True, file_path="tracks/"
    )

    output_frames = draw_annotations(video_frames, players, referees, ball)

    save_video(output_frames, "videos/output_videos/annotated_video.mp4")


if __name__ == "__main__":
    main()
