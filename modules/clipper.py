import subprocess
from pathlib import Path

def run_ffmpeg(command: list):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg command failed: {' '.join(command)}") from e


def extract_clip(input_video: str, output_video: str,
                 start: str = "00:00:15", end: str = "00:00:30"):
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-ss", start,
        "-to", end,
        "-c", "copy",
        output_video
    ]
    run_ffmpeg(command)


def extract_audio(input_video: str, output_audio: str):
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-q:a", "0",
        "-map", "a",
        output_audio
    ]
    run_ffmpeg(command)