import subprocess
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

# Output audio rate. Also the allowance subtracted from a container bit rate when the
# video stream does not report its own.
AUDIO_BITRATE = 192_000

# x264 preset. `medium` was the default and is the wrong trade here: this cuts three ad
# breaks out of an hour-long podcast, so the encode is the entire cost of the job and it
# was running eight cores for over an hour per episode. `veryfast` is a few percent
# larger at the same CRF and several times quicker.
X264_PRESET = os.environ.get("PODCAST_X264_PRESET", "veryfast")

# Quality floor. Unchanged; the bloat was never CRF's fault on its own.
X264_CRF = os.environ.get("PODCAST_X264_CRF", "23")

# The output must never be larger than the input. A CRF encode targets a QUALITY, not a
# size, so re-encoding an already-efficient source at CRF 23 can and does inflate it: a
# 2.3 GB source produced a 3.4 GB output while still climbing. Capping the rate at the
# source's own turns that into "no worse than the original", which is the only sane
# ceiling for a job whose entire purpose is to remove material.
BITRATE_CAP_HEADROOM = 1.0

class VideoProcessor:
    @staticmethod
    def get_duration(file_path: str) -> float:
        """Get duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get duration: {e}")
            return 0.0

    @staticmethod
    def extract_audio(video_path: str, audio_path: str):
        """Extract audio from video for transcription."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Extracted audio from video to {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e.stderr}")
            raise

    @staticmethod
    def probe_stream(file_path: str) -> Dict[str, str]:
        """Codec name and bit rate of the first video stream, plus the container's
        overall bit rate. Empty dict on any failure, which every caller must treat as
        "unknown" rather than as a value."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,bit_rate:format=bit_rate",
            "-of", "default=noprint_wrappers=1",
            file_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except Exception as e:
            logger.warning(f"Could not probe {file_path}: {e}")
            return {}
        out: Dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            # ffprobe prints bit_rate for BOTH the stream and the format section, in that
            # order. They mean different things and both are wanted: the stream's is the
            # cap we want, the format's is the fallback when the stream reports N/A. So
            # the second one is kept under its own name rather than dropped or allowed to
            # overwrite the first.
            if k in out:
                out[f"format_{k}"] = v
            else:
                out[k] = v
        return out

    @staticmethod
    def source_video_bitrate(file_path: str) -> int:
        """Bits per second of the source video, or 0 when it cannot be established.

        Prefers the stream's own bit_rate. MP4 from YouTube often reports N/A there, so
        it falls back to the container rate minus a nominal audio allowance rather than
        giving up: an approximate ceiling is worth far more here than no ceiling.
        """
        info = VideoProcessor.probe_stream(file_path)
        raw = info.get("bit_rate", "")
        try:
            if raw and raw != "N/A":
                return int(raw)
        except ValueError:
            pass
        try:
            total = int(info.get("format_bit_rate", "0") or "0")
        except ValueError:
            total = 0
        if total > AUDIO_BITRATE:
            return total - AUDIO_BITRATE
        return 0

    @staticmethod
    def build_encode_command(input_path: str, output_path: str, filter_complex: str) -> List[str]:
        """The ffmpeg command for the cut-and-concat encode.

        Split out so the encoding decisions are testable without running ffmpeg, which
        is how the size regression went unnoticed: nothing asserted anything about the
        output beyond it existing.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", X264_PRESET,
            "-crf", X264_CRF,
        ]

        # Cap at the source's own video rate when it can be established. CRF alone
        # targets quality and will happily spend more bits than the original did.
        # Unknown means uncapped rather than guessed: a wrong cap would degrade every
        # output, while no cap only risks the size regression this guards against, and
        # the log line says which case we are in.
        src = VideoProcessor.source_video_bitrate(input_path)
        if src > 0:
            cap = int(src * BITRATE_CAP_HEADROOM)
            cmd += ["-maxrate", str(cap), "-bufsize", str(cap * 2)]
            logger.info(f"Capping output video at the source rate: {cap} bps")
        else:
            logger.warning(
                f"Could not establish the source video bit rate of {input_path}; "
                f"encoding uncapped, so the output may exceed the input."
            )

        # Audio is always re-encoded, and cannot be copied: the filtergraph's atrim
        # decodes it to cut it, so there is no original stream left to pass through.
        # The no-segments path above is the one that copies, and it copies both streams.
        cmd += ["-c:a", "aac", "-b:a", str(AUDIO_BITRATE)]
        cmd.append(output_path)
        return cmd

    @staticmethod
    def remove_segments(input_path: str, output_path: str, remove_segments: List[Dict[str, float]]):
        """
        Remove specified segments from video (both video and audio streams).
        """
        if not remove_segments:
            logger.info("No segments to remove, copying video.")
            subprocess.run(["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path], check=True)
            return

        total_duration = VideoProcessor.get_duration(input_path)
        keep_segments = []
        current_time = 0.0

        # Calculate keep segments (inverse of remove segments)
        sorted_segments = sorted(remove_segments, key=lambda x: x['start'])

        for seg in sorted_segments:
            start = seg['start']
            end = seg['end']

            if start > current_time:
                keep_segments.append((current_time, start))
            current_time = max(current_time, end)

        if current_time < total_duration:
            keep_segments.append((current_time, total_duration))

        if not keep_segments:
            raise ValueError("No segments to keep after removal")

        # Build FFmpeg filter_complex for video + audio
        video_filters = []
        audio_filters = []

        for i, (start, end) in enumerate(keep_segments):
            # Video trim
            video_filters.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]"
            )
            # Audio trim
            audio_filters.append(
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
            )

        # Concatenate all segments. ffmpeg's multi-stream concat filter expects
        # inputs interleaved per segment ([v0][a0][v1][a1]...), not grouped by
        # stream type - grouping causes a "media type mismatch" filtergraph error.
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(keep_segments)))
        concat_filter = f"{concat_inputs}concat=n={len(keep_segments)}:v=1:a=1[outv][outa]"

        filter_complex = ";".join(video_filters + audio_filters + [concat_filter])

        cmd = VideoProcessor.build_encode_command(input_path, output_path, filter_complex)

        logger.info(f"Processing video with {len(keep_segments)} keep segments")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Video processed successfully: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr.decode()}")
            raise
