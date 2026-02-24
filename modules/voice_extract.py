import subprocess

def extract_reference_voice(input_audio, output_wav):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_audio,
        "-ss", "00:00:07",
        "-to", "00:00:27",
        "-ac", "1",
        "-ar", "22050",
        "-af", "silenceremove=stop_periods=-1:stop_duration=0.3:stop_threshold=-35dB",
        output_wav
    ]
    subprocess.run(cmd, check=True)
