from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"

_tokenizer = None
_model = None
if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

def load_model():
    global _tokenizer, _model
    if _model is None:
        print("Loading English → Hindi IndicTrans2 model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if _device != "cpu" else torch.float32,
            trust_remote_code=True
        ).to(_device)
    return _tokenizer, _model


def translate_to_hindi(text: str) -> str:
    tokenizer, model = load_model()

    prompt = f"eng_Latn hin_Deva {text}"

    inputs = tokenizer(prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            repetition_penalty=1.2,
            use_cache=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)