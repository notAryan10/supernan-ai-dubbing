import whisper
from pathlib import Path
from transformers import pipeline
import json

_kn_en_translator = None


def _get_translator():
    global _kn_en_translator
    if _kn_en_translator is None:
        print("Loading Kannada→English translation model (facebook/nllb-200-distilled-600M)...")
        _kn_en_translator = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang="kan_Knda",
            tgt_lang="eng_Latn",
            max_length=512,
        )
    return _kn_en_translator


def load_model(size="base"):
    return whisper.load_model(size)


def detect_language(model, audio_path: str) -> str:
    """Use the base model for a fast language detection pass."""
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    print(f"Detected language: {detected}")
    return detected


def translate_to_english(kannada_texts: list[str]) -> list[str]:
    if not kannada_texts:
        return []
    translator = _get_translator()
    print(f"Translating {len(kannada_texts)} segment(s) KN → EN...")
    results = translator(kannada_texts)
    return [r["translation_text"].strip() for r in results]


def transcribe_audio(audio_path: str, output_json: str, translate: bool = True):
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
        if not seg["text"].strip():
            continue
        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        })

    if translate and detected_language != "en" and segments:
        kannada_texts = [s["text"] for s in segments]
        english_texts = translate_to_english(kannada_texts)
        for seg, en_text in zip(segments, english_texts):
            seg["english"] = en_text

    output = {
        "language": detected_language,
        "segments": segments,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Transcription saved → {output_json}  ({len(segments)} segments)")
    return output
