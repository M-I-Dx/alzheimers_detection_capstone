import os

import librosa
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def get_file_list(path, ext=".wav"):
    """
    Get all filenames in a folder with a specific extension.

    Args:
    path (str): Directory path to search for files.
    ext (str): File extension to filter by. Default is ".wav".

    Returns:
    dict: Dictionary with filenames (without extension) as keys and full paths as values.
    """
    result = dict()
    print("Reading from folder =", path)
    files = [item for item in os.listdir(path) if item.endswith(ext)]
    for fname in files:
        result[fname.replace(ext, "")] = os.path.join(path, fname)
    print("Count =", len(result))
    return result


def audio2vec(fname, model, extractor, device, fl16=True):
    """
    Convert an audio file to an embedding vector using mean pooling.

    Args:
    fname (str): Full path of the audio file.
    model (Wav2Vec2Model): Pretrained Wav2Vec2 model for audio processing.
    extractor (Wav2Vec2FeatureExtractor): Feature extractor corresponding to the model.
    device (str): Device to run the model on ('cuda' or 'cpu').
    fl16 (bool): Whether to convert audio to float16 for processing. Default is True.

    Returns:
    torch.Tensor: Mean pooled embedding vector of the audio.
    """
    print("Loading file =", fname)
    input_audio, sample_rate = librosa.load(fname, sr=16000)
    if fl16:
        input_audio = input_audio.astype(np.float16)
    result = extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate).to(
        device
    )
    with torch.no_grad():
        output = model(result.input_values)
    result = torch.mean(output.extract_features[0], dim=0).cpu()
    print("Shape =", result.shape)
    return result


def main():
    """
    Main function to process audio files, generate embeddings, and perform classification.

    Steps:
    1. Check for available CUDA device.
    2. Load and list audio files from specified directories.
    3. Initialize and load the Wav2Vec2 model and feature extractor.
    4. Process and generate embeddings for audio files.
    5. Load existing embeddings and merge with new embeddings.
    6. Load training and development datasets and preprocess text data.
    7. Train a logistic regression model using combined text and audio embeddings.
    8. Evaluate the model on training and development sets.
    9. Process and classify test data.
    10. Save test results to a CSV file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    paths = [
        "../../data/Transcribed Data-2024-Feb-13/Diagnosis Training Data/ad/",
        "../../data/Transcribed Data-2024-Feb-13/Diagnosis Training Data/cn/",
        "../../data/Test Dataset/Diagnosis Testing dataset/diagnosis/test-dist/audio/",
    ]

    file_paths = [get_file_list(path) for path in paths]
    file_paths_dict = {k: v for d in file_paths for k, v in d.items()}
    print(len(file_paths_dict))

    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    model = Wav2Vec2Model.from_pretrained(model_name, torch_dtype=torch.float16).to(
        device
    )
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    fname = file_paths_dict["adrso002"]
    result = audio2vec(fname, model, extractor, device)
    assert len(result) == 512
    print("All passed")

    path_out = "../../data/audio_embed/"
    fname_existing = "train_audio_embed.json"

    try:
        existing_df = pd.read_json(
            os.path.join(path_out, fname_existing), orient="records"
        )
    except Exception as e:
        print("Error reading file =", fname_existing)
        existing_df = pd.DataFrame({"file_id": [], "audio_embed": []})
    print(len(existing_df))
    existing_df.head()

    embed_data = dict()
    fnames = list(file_paths_dict.keys())
    max_n = len(fnames)

    for i, fname in enumerate(fnames[:max_n]):
        print(
            f"Processing {i+1}/{max_n} = {(i+1)*100/max_n:.0f} %, file name = {fname}"
        )
        if fname in list(existing_df["file_id"]):
            print("File existed. Skipping...")
            continue
        try:
            v = audio2vec(file_paths_dict[fname], model, extractor, device)
            embed_data[fname] = v
        except Exception as e:
            print(f"Error = {e}. Skipping...")
        print("---------------")

    f_get_embed = lambda fid: np.asarray(embed_data[fid].cpu().numpy())

    df_out = pd.DataFrame({"file_id": list(embed_data.keys())})
    df_out["audio_embed"] = df_out["file_id"].apply(f_get_embed)
    df_out = pd.concat([df_out, existing_df], ignore_index=True)
    df_out.to_json(os.path.join(path_out, "train_audio_embed.json"), orient="records")

    train_fname = "../../data/splits/train_50pc.csv"
    dev_fname = "../../data/splits/dev_50pc.csv"

    train_df = pd.read_csv(train_fname)
    dev_df = pd.read_csv(dev_fname)

    train_df["text_lower"] = train_df["text"].apply(lambda text: text.lower().strip())
    dev_df["text_lower"] = dev_df["text"].apply(lambda text: text.lower().strip())

    vectorizer = TfidfVectorizer(max_features=10000)
    corpus = train_df["text_lower"]
    X_train = vectorizer.fit_transform(corpus)

    audio_vec_file = os.path.join(path_out, "train_audio_embed.json")
    audio_df = pd.read_json(audio_vec_file, orient="records")
    audio_df.set_index("file_id", inplace=True)

    train_df["audio_embed"] = train_df["file_id"].apply(
        lambda file_id: audio_df.loc[file_id]["audio_embed"]
    )
    dev_df["audio_embed"] = dev_df["file_id"].apply(
        lambda file_id: audio_df.loc[file_id]["audio_embed"]
    )

    vectorizer_embed = SentenceTransformer("LaBSe")
    X_train_embed2 = vectorizer_embed.encode(train_df["text_lower"])
    X_train_embed = np.array(train_df["audio_embed"].tolist())

    X_train = np.concatenate((X_train.toarray(), X_train_embed2), axis=1)
    y_train = train_df["target"]

    clf = LogisticRegression(random_state=42, max_iter=200).fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    print(f"Train accuracy = {train_score*100:.2f} %")
    y_pred = clf.predict(X_train)
    print(classification_report(y_train, y_pred, target_names=["AD", "CN"]))

    X_dev = vectorizer.transform(dev_df["text_lower"])
    X_dev_embed2 = vectorizer_embed.encode(dev_df["text_lower"])
    X_dev_embed = np.array(dev_df["audio_embed"].tolist())
    X_dev = np.concatenate((X_dev.toarray(), X_dev_embed2), axis=1)
    y_dev = dev_df["target"]

    dev_score = clf.score(X_dev, y_dev)
    print(f"Dev accuracy = {dev_score*100:.2f} %")
    y_pred = clf.predict(X_dev)
    print(classification_report(y_dev, y_pred, target_names=["AD", "CN"]))

    test_fname = "../../data/splits/test_text.csv"
    test_df = pd.read_csv(test_fname)
    test_df["text_lower"] = test_df["text"].apply(lambda text: text.lower().strip())
    test_df["audio_embed"] = test_df["ID"].apply(
        lambda file_id: audio_df.loc[file_id]["audio_embed"]
    )

    X_test = vectorizer.transform(test_df["text_lower"])
    X_test_embed2 = vectorizer_embed.encode(test_df["text_lower"])
    X_test_embed = np.array(test_df["audio_embed"].tolist())
    X_test = np.concatenate((X_test.toarray(), X_test_embed2), axis=1)

    test_df["Prediction"] = clf.predict(X_test)
    label2binary = lambda label: (
        "0" if label == "0" or label.lower().strip() == "cn" else "1"
    )
    test_df["Prediction"] = test_df["Prediction"].apply(label2binary)
    output_df = test_df.drop(columns=["text", "text_lower", "audio_embed"])
    output_df.to_csv(
        os.path.join("../../evaluation/", "test_results_task1_audio_embed.csv"),
        index=False,
    )
    print("Completed")


if __name__ == "__main__":
    main()
