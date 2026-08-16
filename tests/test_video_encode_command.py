"""The encode decisions in VideoProcessor.build_encode_command.

Split out of remove_segments precisely so they can be asserted without running ffmpeg.
Nothing checked them before, which is how a job whose purpose is to REMOVE material came
to produce a 3.4 GB output from a 2.3 GB source while burning eight cores an hour.
"""
import pytest

from app.core.video import VideoProcessor, AUDIO_BITRATE


FILTER = "[0:v]trim=start=0:end=1[v0];[0:a]atrim=start=0:end=1[a0];[v0][a0]concat=n=1:v=1:a=1[outv][outa]"


def _cmd(monkeypatch, bitrate):
    monkeypatch.setattr(VideoProcessor, "source_video_bitrate", staticmethod(lambda p: bitrate))
    return VideoProcessor.build_encode_command("in.mp4", "out.mp4", FILTER)


def _val(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_caps_the_output_at_the_source_video_rate(monkeypatch):
    cmd = _cmd(monkeypatch, 2_000_000)
    assert _val(cmd, "-maxrate") == "2000000"
    # bufsize at 2x maxrate is the conventional pairing; without a bufsize, maxrate is
    # advisory and the encoder can overshoot it for long stretches.
    assert _val(cmd, "-bufsize") == "4000000"


def test_encodes_uncapped_when_the_source_rate_is_unknown(monkeypatch):
    # Unknown must not become a guess: a wrong cap degrades every output, while no cap
    # only risks the size regression. The warning in the log is the signal.
    cmd = _cmd(monkeypatch, 0)
    assert "-maxrate" not in cmd
    assert "-bufsize" not in cmd


def test_uses_a_fast_preset_rather_than_medium(monkeypatch):
    # The encode is the entire cost of this job. medium was the default and ran eight
    # cores for over an hour per episode.
    assert _val(_cmd(monkeypatch, 1_000_000), "-preset") == "veryfast"


def test_preset_and_crf_are_overridable_by_env(monkeypatch):
    monkeypatch.setenv("PODCAST_X264_PRESET", "slow")
    monkeypatch.setenv("PODCAST_X264_CRF", "20")
    import importlib
    import app.core.video as video
    importlib.reload(video)
    try:
        monkeypatch.setattr(video.VideoProcessor, "source_video_bitrate", staticmethod(lambda p: 0))
        cmd = video.VideoProcessor.build_encode_command("in.mp4", "out.mp4", FILTER)
        assert cmd[cmd.index("-preset") + 1] == "slow"
        assert cmd[cmd.index("-crf") + 1] == "20"
    finally:
        monkeypatch.delenv("PODCAST_X264_PRESET", raising=False)
        monkeypatch.delenv("PODCAST_X264_CRF", raising=False)
        importlib.reload(video)


def test_keeps_the_filtergraph_and_stream_mapping_intact(monkeypatch):
    cmd = _cmd(monkeypatch, 1_000_000)
    assert _val(cmd, "-filter_complex") == FILTER
    assert cmd[-1] == "out.mp4"
    assert cmd.count("-map") == 2
    assert _val(cmd, "-b:a") == str(AUDIO_BITRATE)


class _Result:
    def __init__(self, stdout):
        self.stdout = stdout


def test_stream_bitrate_wins_over_the_container_bitrate(monkeypatch):
    # ffprobe prints bit_rate for BOTH the stream and the format section. The stream's
    # comes first and is the one we want; letting the format's overwrite it would cap
    # the video at the whole container's rate, which is too generous by the audio.
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _Result("codec_name=h264\nbit_rate=1500000\nbit_rate=1700000\n"),
    )
    assert VideoProcessor.source_video_bitrate("x.mp4") == 1500000


def test_falls_back_to_the_container_rate_less_audio_when_the_stream_says_na(monkeypatch):
    # YouTube MP4 routinely reports N/A on the stream. An approximate ceiling beats none.
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _Result("codec_name=h264\nbit_rate=N/A\nbit_rate=1700000\n"),
    )
    assert VideoProcessor.source_video_bitrate("x.mp4") == 1_700_000 - AUDIO_BITRATE


def test_returns_zero_rather_than_raising_when_ffprobe_fails(monkeypatch):
    def boom(*a, **k):
        raise OSError("no ffprobe")
    monkeypatch.setattr("subprocess.run", boom)
    assert VideoProcessor.source_video_bitrate("x.mp4") == 0
