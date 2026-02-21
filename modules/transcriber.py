import whisper
from pathlib import Path
import json



def transcribe_audio(audio_path: str, output_json: str):
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper large-v3...")
    model = whisper.load_model("large-v3")

    result = model.transcribe(
        audio_path,
        task="translate",
        language=None,
        fp16=False,

        condition_on_previous_text=False,

        temperature=0.0,

        compression_ratio_threshold=3.0,
        logprob_threshold=-1.2,
        no_speech_threshold=0.3,

        beam_size=3,
        best_of=3
    )

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if text:
            segments.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": text
            })

    output = {
        "language": "auto",
        "segments": segments
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved → {output_json} ({len(segments)} segments)")
    return output
