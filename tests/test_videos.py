from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from football_analysis.video_utils import read_video, save_video


def test_read_valid_video():
    """test reading a valid video file"""
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [(True, "frame1"), (True, "frame2"), (False, None)]

    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames = read_video("valid_video.mp4")

    assert frames == ["frame1", "frame2"]


def test_save_video():
    """Test savind a video.
    Creates two dummy frames and
    test if VideoWriter was called with the right arguments
    and the right amount of times"""
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

    frames = [frame1, frame2]
    output_path = "test_output.avi"

    with patch("cv2.VideoWriter") as mock_videowriter:
        mock_writer_instance = MagicMock()
        mock_videowriter.return_value = mock_writer_instance

        save_video(frames, output_path)

        # Check if VideoWriter was instasiated with correct parameters
        mock_videowriter.assert_called_once_with(
            output_path, cv2.VideoWriter.fourcc(*"XVID"), 24, (640, 480)
        )

        # Check if write() was called twice (for two frames)
        assert mock_writer_instance.write.call_count == 2

        # Check if release() was called
        mock_writer_instance.release.assert_called_once()
