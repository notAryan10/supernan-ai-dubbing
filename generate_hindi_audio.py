import json
import os
import subprocess
from pydub import AudioSegment
from modules.voice_extract import extract_reference_voice
from modules.tts_clone import speak_hindi

def stretch_audio_ffmpeg(input_wav, output_wav, rate):
    """
    Stretches audio using ffmpeg's atempo filter.
    atempo supports between 0.5 and 2.0. We chain them for higher rates.
    """
    filters = []
    temp_rate = rate
    while temp_rate > 2.0:
        filters.append("atempo=2.0")
        temp_rate /= 2.0
    while temp_rate < 0.5:
        filters.append("atempo=0.5")
        temp_rate /= 0.5
    filters.append(f"atempo={temp_rate}")
    
    filter_str = ",".join(filters)
    
    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-af", filter_str,
        output_wav
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def generate_all_audio(transcript_json):
    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = "temp/hindi_full.wav"
    combined_audio = AudioSegment.empty()

    for i, seg in enumerate(data["segments"]):
        text = seg["text"].strip()
        if not text:
            continue
            
        segment_wav = f"temp/hindi_segment_{i}.wav"
        print(f"Generating audio for segment {i}: {text}")
        speak_hindi(
            text=text,
            speaker_wav="temp/ref_voice.wav",
            output_wav=segment_wav
        )
        
        # Audio Duration Matching using ffmpeg atempo
        try:
            target_duration = seg["end"] - seg["start"]
            seg_audio = AudioSegment.from_wav(segment_wav)
            current_duration = seg_audio.duration_seconds
            
            if current_duration > 0:
                rate = current_duration / target_duration
                # Allow higher rates to fit the clip (up to 4x)
                rate = max(0.5, min(rate, 4.0))
                
                if abs(rate - 1.0) > 0.05: # Only stretch if significant difference
                    print(f"Stretching segment {i}: {current_duration:.2f}s -> {target_duration:.2f}s (rate: {rate:.2f})")
                    temp_stretched = f"temp/hindi_segment_{i}_stretched.wav"
                    stretch_audio_ffmpeg(segment_wav, temp_stretched, rate)
                    seg_audio = AudioSegment.from_wav(temp_stretched)
                    os.remove(temp_stretched)
        except Exception as e:
            print(f"Warning: Could not stretch segment {i}: {e}")

        combined_audio += seg_audio
        os.remove(segment_wav)

    print(f"Exporting combined audio to {out_file}")
    combined_audio.export(out_file, format="wav")

if __name__ == "__main__":
    audio_wav = "temp/audio.wav"
    transcript_file = "temp/transcript_hindi.json"

    os.makedirs("temp", exist_ok=True)

    if not os.path.exists(audio_wav):
        print(f"Error: Required file '{audio_wav}' not found.")
        print("Please run 'python dub_video.py' first to extract the audio and generate transcripts.")
        exit(1)

    if not os.path.exists(transcript_file):
        print(f"Error: Hindi transcript '{transcript_file}' not found.")
        print("Please run 'python dub_video.py' first to generate the translated transcript.")
        exit(1)

    if not os.path.exists("temp/ref_voice.wav"):
        print(f"Extracting reference voice from {audio_wav}...")
        try:
            extract_reference_voice(audio_wav, "temp/ref_voice.wav")
            print("Reference voice extracted successfully.")
        except Exception as e:
            print(f"Error extracting reference voice: {e}")
            exit(1)
        
    print(f"Generating Hindi audio using transcript {transcript_file}...")
    generate_all_audio(transcript_file)
    print("Done generating audio files!")
