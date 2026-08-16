from unittest.mock import patch

from app.core.video import VideoProcessor


def test_remove_segments_interleaves_video_and_audio_inputs():
    """ffmpeg's multi-stream concat filter requires inputs interleaved per
    segment ([v0][a0][v1][a1]...) - grouping all video then all audio inputs
    raises a "media type mismatch" filtergraph error at runtime."""
    with patch("app.core.video.VideoProcessor.get_duration", return_value=100.0), \
         patch("app.core.video.subprocess.run") as mock_run:
        VideoProcessor.remove_segments(
            "in.mp4", "out.mp4",
            [{"start": 10.0, "end": 20.0}],
        )

    cmd = mock_run.call_args[0][0]
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    concat_clause = filter_complex.split(";")[-1]

    assert concat_clause.startswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]")
