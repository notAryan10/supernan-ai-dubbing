import json
import os
from pydub import AudioSegment
from modules.voice_extract import extract_reference_voice
from modules.tts_clone import speak_hindi

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
        
        seg_audio = AudioSegment.from_wav(segment_wav)
        combined_audio += seg_audio
        
        os.remove(segment_wav)

    print(f"Exporting combined audio to {out_file}")
    combined_audio.export(out_file, format="wav")

if __name__ == "__main__":
    if not os.path.exists("temp/ref_voice.wav"):
        print("Extracting reference voice from temp/audio.wav...")
        extract_reference_voice("temp/audio.wav", "temp/ref_voice.wav")
        print("Reference voice extracted successfully.")
        
    transcript_file = "temp/transcript_hindi.json"
    print(f"Generating Hindi audio using transcript {transcript_file}...")
    generate_all_audio(transcript_file)
    print("Done generating audio files!")
