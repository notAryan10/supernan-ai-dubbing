from modules.clipper import extract_clip, extract_audio, DEFAULT_START, DEFAULT_END
from modules.transcriber import transcribe_audio
from modules.hindi_translator import translate_to_hindi
import json

def main():
    input_video = "data/input.mp4"
    clip_video = "temp/clip.mp4"
    audio_file = "temp/audio.wav"
    translated_transcript_file = "temp/transcript_english.json"
    hindi_transcript_file = "temp/transcript_hindi.json"

    print(f"Extracting silent {DEFAULT_START}-{DEFAULT_END} clip...")
    extract_clip(input_video, clip_video, start=DEFAULT_START, end=DEFAULT_END)

    print("Extracting audio from source...")
    extract_audio(input_video, audio_file, start=DEFAULT_START, end=DEFAULT_END)

    print("Transcribing and translating audio to English directly...")
    data = transcribe_audio(audio_file, translated_transcript_file)

    for s in data["segments"]:
        print(f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}")

    print("\nTranslation to English complete!")

    print("\nTranslating English text to Hindi...")
    hindi_data = {
        "language": "hi",
        "segments": []
    }
    
    for s in data["segments"]:
        hindi_text = translate_to_hindi(s["text"])
        hindi_data["segments"].append({
            "start": s["start"],
            "end": s["end"],
            "text": hindi_text
        })
        print(f"[{s['start']:.2f} - {s['end']:.2f}] {hindi_text}")
        
    with open(hindi_transcript_file, "w", encoding="utf-8") as f:
        json.dump(hindi_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved Hindi translation to {hindi_transcript_file}")

if __name__ == "__main__":
    main()
