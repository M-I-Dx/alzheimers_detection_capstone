import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from summarizer import Summarizer
import joblib


class AlzheimerClassifierTrainer:
    """
    A class for training an Alzheimer's disease classifier.

    Args:
        train_fname (str): Filename of the training dataset.
        dev_fname (str): Filename of the development dataset.
        train_ratio (float, optional): Ratio of training data to use. Defaults to 1.0.
        use_summary (bool, optional): Whether to use text summarization. Defaults to True.
        embed_model_name (str, optional): Name of the embedding model to use. Defaults to "LaBSe".
        concat_vector (bool, optional): Whether to concatenate TF-IDF and embedding vectors. Defaults to True.
        use_embeddings_only (bool, optional): Whether to use only embedding vectors. Defaults to False.
        tuning_score (str, optional): Metric to use for hyperparameter tuning. Defaults to "f1-score".

    Attributes:
        vectorizer (TfidfVectorizer): TF-IDF vectorizer.
        sum_model (Summarizer): Text summarization model.
        vectorizer_embed (SentenceTransformer): Sentence embedding model.
        clf (LogisticRegression): Logistic regression classifier.
    """

    def __init__(
        self,
        train_fname,
        dev_fname,
        train_ratio=1.0,
        use_summary=True,
        embed_model_name="LaBSe",
        concat_vector=True,
        use_embeddings_only=False,
        tuning_score="f1-score",
    ):
        self.train_fname = train_fname
        self.dev_fname = dev_fname
        self.train_ratio = train_ratio
        self.use_summary = use_summary
        self.embed_model_name = embed_model_name
        self.concat_vector = concat_vector
        self.use_embeddings_only = use_embeddings_only
        self.tuning_score = tuning_score
        self.vectorizer = None
        self.sum_model = None
        self.vectorizer_embed = None
        self.clf = None

    def load_data(self):
        """Load and preprocess the training and development datasets."""
        self.train_df = pd.read_csv(self.train_fname)
        self.dev_df = pd.read_csv(self.dev_fname)
        self.train_df = self.train_df.sample(frac=self.train_ratio, random_state=42)
        self.train_df.reset_index(inplace=True)
        self.train_df["text_lower"] = self.train_df["text"].apply(
            lambda text: text.lower().strip()
        )
        self.dev_df["text_lower"] = self.dev_df["text"].apply(
            lambda text: text.lower().strip()
        )

    def summarize_text(self):
        """Summarize the text data if use_summary is True."""
        self.sum_model = Summarizer()
        if self.use_summary:
            summ_f = (
                lambda text: f"{text.lower()}. Summary: {self.sum_model(text).lower()}"
            )
        else:
            summ_f = lambda text: f"{text.lower()}"
        self.train_df["text_lower"] = self.train_df["text"].apply(summ_f)
        self.dev_df["text_lower"] = self.dev_df["text"].apply(summ_f)

    def build_tfidf_vectorizer(self):
        """Build the TF-IDF vectorizer and transform the training data."""
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        corpus = self.train_df["text_lower"]
        self.X_train_tfidf = self.vectorizer.fit_transform(corpus)

    def create_embeddings(self):
        """Create sentence embeddings for the training data."""
        self.vectorizer_embed = SentenceTransformer(self.embed_model_name)
        corpus = self.train_df["text_lower"]
        self.X_train_embed = self.vectorizer_embed.encode(corpus)

    def combine_vectors(self, tfidf_vec, embed_vec):
        """Combine TF-IDF and embedding vectors based on the specified options."""
        if self.concat_vector:
            result = np.concatenate((tfidf_vec.toarray(), embed_vec), axis=1)
        elif self.use_embeddings_only:
            result = embed_vec
        else:
            result = tfidf_vec
        return result

    def train_model(self):
        """Train the logistic regression classifier."""
        self.y_train = self.train_df["target"]
        self.X_train = self.combine_vectors(self.X_train_tfidf, self.X_train_embed)
        self.clf = LogisticRegression(random_state=42, class_weight="balanced").fit(
            self.X_train, self.y_train
        )

    def evaluate_model(self):
        """Evaluate the trained model on the development dataset."""
        X_dev_tfidf = self.vectorizer.transform(self.dev_df["text_lower"])
        y_dev = self.dev_df["target"]
        X_dev_embed = self.vectorizer_embed.encode(self.dev_df["text_lower"])
        X_dev = self.combine_vectors(X_dev_tfidf, X_dev_embed)
        dev_score = self.clf.score(X_dev, y_dev)
        y_pred = self.clf.predict(X_dev)
        print(f"Dev accuracy = {dev_score*100:.2f} %")
        print(classification_report(y_dev, y_pred, target_names=["AD", "CN"]))

    def save_models(self):
        """Save the trained models to disk."""
        joblib.dump(self.vectorizer, "tfidf_vectorizer.pkl")
        joblib.dump(self.sum_model, "sum_model.pkl")
        joblib.dump(self.vectorizer_embed, "embed_vectorizer.pkl")
        joblib.dump(self.clf, "logistic_regression.pkl")

    def train(self):
        """Train the Alzheimer's disease classifier."""
        self.load_data()
        self.summarize_text()
        self.build_tfidf_vectorizer()
        self.create_embeddings()
        self.train_model()
        self.evaluate_model()
        self.save_models()


class AlzheimerClassifierPredictor:
    """
    A class for making predictions using a pre-trained Alzheimer's disease classifier.

    Args:
        use_summary (bool, optional): Whether to use text summarization. Defaults to True.

    Attributes:
        vectorizer (TfidfVectorizer): TF-IDF vectorizer.
        sum_model (Summarizer): Text summarization model.
        vectorizer_embed (SentenceTransformer): Sentence embedding model.
        clf (LogisticRegression): Logistic regression classifier.
    """

    def __init__(self, use_summary=True):
        self.vectorizer = joblib.load("./app/model/tfidf_vectorizer.pkl")
        self.sum_model = joblib.load("./app/model/sum_model.pkl")
        self.vectorizer_embed = joblib.load("./app/model/embed_vectorizer.pkl")
        self.clf = joblib.load("./app/model/logistic_regression.pkl")
        self.use_summary = use_summary

    def preprocess_text(self, text):
        """
        Preprocess the input text.

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        text = text.lower().strip()
        if self.use_summary:
            text = f"{text}. Summary: {self.sum_model(text).lower()}"
        return text

    def combine_vectors(self, tfidf_vec, embed_vec):
        """
        Combine TF-IDF and embedding vectors.

        Args:
            tfidf_vec (scipy.sparse.csr_matrix): TF-IDF vector.
            embed_vec (numpy.ndarray): Embedding vector.

        Returns:
            numpy.ndarray: Combined vector.
        """
        result = np.concatenate((tfidf_vec.toarray(), embed_vec), axis=1)
        return result

    def predict(self, input_strings):
        """
        Make predictions on the input strings.

        Args:
            input_strings (list): List of input strings to make predictions on.

        Returns:
            numpy.ndarray: Predicted labels for the input strings.
        """
        preprocessed_texts = [self.preprocess_text(text) for text in input_strings]
        X_tfidf = self.vectorizer.transform(preprocessed_texts)
        X_embed = self.vectorizer_embed.encode(preprocessed_texts)
        X = self.combine_vectors(X_tfidf, X_embed)
        predictions = self.clf.predict(X)
        return predictions
