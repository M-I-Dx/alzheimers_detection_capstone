import os
from tqdm import tqdm
import pandas as pd
import whisper


def transcribe_audio(audio_path, model_size, language="en"):
    model = whisper.load_model(model_size)
    if language:
        result = model.transcribe(audio_path, language=language)
    else:
        result = model.transcribe(audio_path)
    return result["text"]


if __name__ == "__main__":
    audio_dir = "<PATH>"
    classes = ["ad", "cn"]
    model_sizes = ["tiny.en", "base.en", "small.en", "medium.en", "large"]


    for model_size in model_sizes:
        print(f"Transcribing using {model_size} model...")
        transcriptions = {}
        if model_size == "large":
            language = "en"
        else:
            language = None
        for class_name in classes:
            class_dir = os.path.join(audio_dir, class_name)
            for filename in tqdm(os.listdir(class_dir)):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(class_dir, filename)
                    transcription = transcribe_audio(audio_path, model_size, language)
                    transcriptions[filename] = [transcription, class_name]
        df = pd.DataFrame.from_dict(transcriptions, orient="index", columns=["transcription", "class"])
        df.to_csv(f"{model_size}_transcriptions.csv")