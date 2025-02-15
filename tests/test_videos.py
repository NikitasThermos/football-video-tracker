from unittest.mock import MagicMock, patch

import cv2
import pytest

from football_analysis.video_utils import read_video


def test_read_valid_video():
    """test reading a valid video file"""
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [(True, "frame1"), (True, "frame2"), (False, None)]

    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames = read_video("valid_video.mp4")

    assert frames == ["frame1", "frame2"]
    
