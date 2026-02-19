from modules.clipper import extract_clip, extract_audio


def main():
    input_video = "data/input.mp4"
    clip_video = "temp/clip.mp4"
    audio_file = "temp/audio.wav"

    print("Extracting 15 second clip...")
    extract_clip(input_video, clip_video)

    print("Extracting audio...")
    extract_audio(clip_video, audio_file)

    print("Clip and audio ready!")


if __name__ == "__main__":
    main()