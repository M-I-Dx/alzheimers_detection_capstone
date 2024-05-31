import os
import pandas as pd
import librosa
from fastdtw import fastdtw
import wespeaker
import numpy as np


class DataLoader:
    """
    DataLoader is responsible for loading and processing the original and new audio data.

    Attributes:
        dev_csv_path (str): Path to the CSV file containing the original data information.
        original_audio_path (str): Base directory for the original audio files.
        new_audio_ad_path (str): Directory for the new audio files (ad category).
        new_audio_cn_path (str): Directory for the new audio files (cn category).
    """
    def __init__(self, dev_csv_path, original_audio_path, new_audio_ad_path, new_audio_cn_path):
        self.dev_csv_path = dev_csv_path
        self.original_audio_path = original_audio_path
        self.new_audio_ad_path = new_audio_ad_path
        self.new_audio_cn_path = new_audio_cn_path

    def load_original_data(self):
        """
        Load the original data from the CSV file and construct the full file paths.

        Returns:
            pd.DataFrame: DataFrame containing filenames and their corresponding audio directory paths.
        """
        original_filenames = pd.read_csv(self.dev_csv_path).filename.values.tolist()
        original_audio_dirs = pd.read_csv(self.dev_csv_path)["class"].values.tolist()
        original_audio_dirs = [self.original_audio_path + dir + "/" for dir in original_audio_dirs]
        original_audio_dirs = [dir + filename for dir, filename in zip(original_audio_dirs, original_filenames)]
        return pd.DataFrame({"filename": original_filenames, "audio_dir": original_audio_dirs})

    def load_new_data(self):
        """
        Load the new audio data and construct the full file paths for both 'ad' and 'cn' categories.

        Returns:
            pd.DataFrame: DataFrame containing filenames and their corresponding audio directory paths.
        """
        new_ad_filenames = self._get_filenames(self.new_audio_ad_path)
        new_cn_filenames = self._get_filenames(self.new_audio_cn_path)
        new_ad_audio_dirs = [self.new_audio_ad_path + filename for filename in new_ad_filenames]
        new_cn_audio_dirs = [self.new_audio_cn_path + filename for filename in new_cn_filenames]
        return pd.DataFrame({"filename": new_ad_filenames + new_cn_filenames, "audio_dir": new_ad_audio_dirs + new_cn_audio_dirs})

    def _get_filenames(self, path):
        """
        Helper method to get filenames from a directory that end with '.mp3'.

        Args:
            path (str): Directory path to search for audio files.

        Returns:
            list: List of filenames ending with '.mp3'.
        """
        return [filename for filename in os.listdir(path) if filename.endswith(".mp3")]


class SimilarityCalculator:
    """
    SimilarityCalculator calculates the similarity between two audio files using DTW and WeSpeaker methods.

    Attributes:
        model (wespeaker.Wespeaker): WeSpeaker model instance for computing similarity.
    """
    def __init__(self, model_name):
        self.model = wespeaker.load_model(model_name)

    def dtw_similarity(self, file1, file2):
        """
        Calculate the Dynamic Time Warping (DTW) similarity between two audio files.

        Args:
            file1 (str): Path to the first audio file.
            file2 (str): Path to the second audio file.

        Returns:
            float: DTW distance between the two audio files.
        """
        y1, sr1 = librosa.load(file1)
        y2, sr2 = librosa.load(file2)
        distance, _ = fastdtw(y1, y2)
        return distance

    def wespeaker_similarity(self, file1, file2):
        """
        Calculate the WeSpeaker similarity between two audio files.

        Args:
            file1 (str): Path to the first audio file.
            file2 (str): Path to the second audio file.

        Returns:
            float: Similarity score between the two audio files.
        """
        similarity = self.model.compute_similarity(file1, file2)
        return similarity

class ProgressManager:
    """
    ProgressManager handles the saving and loading of processing progress to ensure resumption from the last saved state.

    Attributes:
        progress_file (str): Path to the file storing progress checkpoints.
        similarity_file (str): Path to the file storing similarity results.
    """
    def __init__(self, progress_file, similarity_file):
        self.progress_file = progress_file
        self.similarity_file = similarity_file

    def load_progress(self):
        """
        Load progress from the checkpoint file, if available.

        Returns:
            tuple: A tuple containing the starting indices and the list of similarity results.
        """
        if os.path.exists(self.progress_file):
            progress_df = pd.read_csv(self.progress_file)
            start_idx1 = progress_df.iloc[0]['idx1']
            start_idx2 = progress_df.iloc[0]['idx2']
            similarity_results = pd.read_csv(self.similarity_file).to_dict('records')
        else:
            start_idx1 = 0
            start_idx2 = 0
            similarity_results = []
        return start_idx1, start_idx2, similarity_results

    def save_progress(self, idx1, idx2, similarity_results):
        """
        Save the current progress and similarity results to the files.

        Args:
            idx1 (int): Current index in the original data.
            idx2 (int): Current index in the new data.
            similarity_results (list): List of similarity results to save.
        """
        df = pd.DataFrame(similarity_results)
        df.to_csv(self.similarity_file, index=False)
        progress_checkpoint = pd.DataFrame({'idx1': [idx1], 'idx2': [idx2]})
        progress_checkpoint.to_csv(self.progress_file, index=False)


class SimilarityProcessor:
    """
    SimilarityProcessor coordinates the processing of calculating similarities between original and new audio files.

    Attributes:
        original_data (pd.DataFrame): DataFrame containing the original audio data.
        new_data (pd.DataFrame): DataFrame containing the new audio data.
        calculator (SimilarityCalculator): Instance of SimilarityCalculator for calculating similarities.
        progress_manager (ProgressManager): Instance of ProgressManager for handling progress checkpoints.
    """
    def __init__(self, original_data, new_data, calculator, progress_manager):
        self.original_data = original_data
        self.new_data = new_data
        self.calculator = calculator
        self.progress_manager = progress_manager

    def process(self):
        """
        Process all combinations of original and new audio data, calculating their similarities and saving the progress periodically.
        """
        start_idx1, start_idx2, similarity_results = self.progress_manager.load_progress()
        total = len(self.original_data) * len(self.new_data)
        count = len(similarity_results)

        for idx1 in range(start_idx1, len(self.original_data)):
            row1 = self.original_data.iloc[idx1]
            file1 = row1['audio_dir']
            filename1 = os.path.basename(file1)
            for idx2 in range(start_idx2, len(self.new_data)):
                row2 = self.new_data.iloc[idx2]
                file2 = row2['audio_dir']
                filename2 = os.path.basename(file2)
                dtw_score = self.calculator.dtw_similarity(file1, file2)
                wespeaker_score = self.calculator.wespeaker_similarity(file1, file2)
                similarity_results.append({
                    'original_file': filename1,
                    'new_file': filename2,
                    'dtw_score': dtw_score,
                    'wespeaker_score': wespeaker_score
                })
                count += 1
                print(f"Progress: {count}/{total}")

                if count % 20 == 0:
                    self.progress_manager.save_progress(idx1, idx2, similarity_results)
            start_idx2 = 0

        self.progress_manager.save_progress(len(self.original_data), len(self.new_data), similarity_results)

def main():
    dev_csv_path = "/content/drive/MyDrive/Colab-Notebooks/Capstone/Data/Original/dev.csv"
    original_audio_path = "/content/drive/MyDrive/Colab-Notebooks/Capstone/Data/Original/"
    new_audio_ad_path = "/content/drive/MyDrive/Colab-Notebooks/Capstone/Data/Pitt/Audio/ad/"
    new_audio_cn_path = "/content/drive/MyDrive/Colab-Notebooks/Capstone/Data/Pitt/Audio/cn/"
    model_name = "english"

    data_loader = DataLoader(dev_csv_path, original_audio_path, new_audio_ad_path, new_audio_cn_path)
    original_data = data_loader.load_original_data()
    new_data = data_loader.load_new_data()

    calculator = SimilarityCalculator(model_name)
    progress_file = '/content/drive/MyDrive/Colab-Notebooks/Capstone/Data/progress_checkpoint.csv'
    similarity_file = '/content/drive/MyDrive/Colab-Notebooks/Capstone/Data/similarity_results.csv'
    progress_manager = ProgressManager(progress_file, similarity_file)

    processor = SimilarityProcessor(original_data, new_data, calculator, progress_manager)
    processor.process()

if __name__ == "__main__":
    main()
