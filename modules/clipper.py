import subprocess
from pathlib import Path

DEFAULT_START = "00:00:45"
DEFAULT_END = "00:01:00"

def run_ffmpeg(command: list):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg command failed: {' '.join(command)}") from e


def extract_clip(input_video: str, output_video: str,
                 start: str = DEFAULT_START, end: str = DEFAULT_END):
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-ss", start,
        "-to", end,
        "-c:v", "copy",
        "-an",
        output_video
    ]
    run_ffmpeg(command)


def extract_audio(input_video: str, output_audio: str, 
                  start: str = None, end: str = None):
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y"
    ]
    
    if start:
        command.extend(["-ss", start])
    if end:
        command.extend(["-to", end])
        
    command.extend([
        "-i", input_video,
        "-ac", "1",
        "-ar", "16000",
        "-af", "loudnorm,silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-35dB",
        "-vn",
        output_audio
    ])
    run_ffmpeg(command)