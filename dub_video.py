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
    data = transcribe_audio(audio_file, transcript_file)

    print(f"\nDetected Language: {data['language']}\n")

    for s in data["segments"]:
        print(f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}")

    print("\nTranscription complete!")



if __name__ == "__main__":
    main()
