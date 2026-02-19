import whisper
from pathlib import Path
import json


def load_model(size="base"):
    return whisper.load_model(size)


def detect_language(model, audio_path: str) -> str:
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    print(f"Detected language: {detected}")
    return detected



def transcribe_audio(audio_path: str, output_json: str):
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    base_model = load_model("base")
    detected_language = detect_language(base_model, audio_path)

    if detected_language != "en":
        print(f"Non-English detected ({detected_language}) → switching to LARGE model")
        model = load_model("large")
    else:
        print("English detected → using BASE model")
        model = base_model

    result = model.transcribe(
        audio_path,
        fp16=False,
        language=detected_language,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        beam_size=5,
    )

    segments = []
    for seg in result["segments"]:
        if seg.get("no_speech_prob", 0) > 0.6:
            continue
        if seg.get("avg_logprob", 0) < -1.0:
            continue
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    output = {
        "language": detected_language,
        "segments": segments
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output
