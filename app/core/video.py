import subprocess
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

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

        # Concatenate all segments
        v_concat = "".join(f"[v{i}]" for i in range(len(keep_segments)))
        a_concat = "".join(f"[a{i}]" for i in range(len(keep_segments)))
        concat_filter = f"{v_concat}{a_concat}concat=n={len(keep_segments)}:v=1:a=1[outv][outa]"

        filter_complex = ";".join(video_filters + audio_filters + [concat_filter])

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ]

        logger.info(f"Processing video with {len(keep_segments)} keep segments")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Video processed successfully: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr.decode()}")
            raise
