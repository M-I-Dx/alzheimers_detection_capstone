import gradio as gr
import whisper
import pandas as pd
from model import AlzheimerClassifierPredictor


MODEL = whisper.load_model("medium.en")


def transcribe_audios(audio_files):
    transcripts = []
    for audio_file in audio_files:
        if not (audio_file.endswith(".mp3") or audio_file.endswith(".wav")):
            return "Error: Only .mp3 and .wav files are supported."
        transcriptions = MODEL.transcribe(audio_file, language="en")
        transcripts.append(transcriptions["text"])
    return transcripts


def classifier(audio_files):
    transcripts = transcribe_audios(audio_files)
    filenames = [audio_file.split("/")[-1] for audio_file in audio_files]
    predictor = AlzheimerClassifierPredictor()
    predictions = predictor.predict(transcripts)
    predictions = pd.DataFrame({"Filename": filenames, "Transcript": transcripts, "Prediction": predictions})
    predictions.to_csv("transcriptions.csv", index=False)
    return predictions, "transcriptions.csv"


iface = gr.Interface(
    fn=classifier,
    inputs=gr.Files(type="filepath", label="Upload your audio files (.mp3 or .wav)"),
    outputs=[
        gr.DataFrame(headers=["Filename", "Transcript", "Prediction"]),
        gr.File(type="filepath", label="Download CSV")
    ],
    title="Audio Transcription with Whisper",
    description="Upload multiple .mp3 or .wav audio files to get their transcriptions using OpenAI's Whisper model."
)
# Launch the app
if __name__ == "__main__":
    iface.launch()

