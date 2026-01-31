# asr.py
import whisper

_model = None

def load_model(name: str = "base"):
    global _model
    if _model is None:
        _model = whisper.load_model(name)
    return _model

def transcribe_audio_file(path: str, model_name: str = "base") -> str:
    model = load_model(model_name)
    result = model.transcribe(path, fp16=False)
    return (result.get("text") or "").strip()