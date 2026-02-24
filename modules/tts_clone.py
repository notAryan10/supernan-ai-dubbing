from TTS.api import TTS
import torch

from unittest.mock import patch

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

print("Loading XTTS v2 voice cloning model...")


original_load = torch.load
with patch('torch.load', lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})):
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2"
    ).to(device)

def speak_hindi(text, speaker_wav, output_wav):
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="hi",
        file_path=output_wav,
        temperature=0.3,         
        length_penalty=0.9,       
        repetition_penalty=2.5,
        top_k=50,
        top_p=0.85
    )
