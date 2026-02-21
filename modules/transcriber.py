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


def transcribe_audio(audio_path: str, output_json: str, task: str = "transcribe"):
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    base_model = load_model("base")
    detected_language = detect_language(base_model, audio_path)
    del base_model  

    if detected_language != "en":
        print(f"Non-English detected ({detected_language}) → loading large-v3 for max accuracy")
        model = load_model("large-v3")
    else:
        print("English detected → using base model")
        model = load_model("base")

    result = model.transcribe(
        audio_path,
        fp16=False,
        language=detected_language,
        task=task,
        condition_on_previous_text=True,  
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), 
        beam_size=5,
        best_of=5,                        
        patience=1.0,                      
        compression_ratio_threshold=2.4,   
        no_speech_threshold=0.5,           
        word_timestamps=True,              
    )

    segments = []
    for seg in result["segments"]:
        if seg.get("no_speech_prob", 0) > 0.5:
            continue
        if seg.get("avg_logprob", 0) < -0.8:
            continue
        if seg.get("compression_ratio", 0) > 2.4:
            continue
        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        })

    output = {
        "language": detected_language,
        "segments": segments,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Transcription saved → {output_json}  ({len(segments)} segments)")
    return output
