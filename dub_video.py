from modules.clipper import extract_clip, extract_audio
from modules.transcriber import transcribe_audio


def main():
    input_video = "data/input.mp4"
    clip_video = "temp/clip.mp4"
    audio_file = "temp/audio.wav"
    transcript_file = "temp/transcript.json"

    print("Extracting 15 second clip...")
    extract_clip(input_video, clip_video)

    print("Extracting audio...")
    extract_audio(clip_video, audio_file)

    print("Transcribing audio...")
    data = transcribe_audio(audio_file, transcript_file, translate=True)

    print(f"\nDetected Language: {data['language']}\n")

    for s in data["segments"]:
        print(f"[{s['start']:.2f} - {s['end']:.2f}]")
        print(f"  {data['language'].upper()}: {s['text']}")
        if "english" in s:
            print(f"  EN: {s['english']}")

    print("\nTranscription complete!")



if __name__ == "__main__":
    main()
