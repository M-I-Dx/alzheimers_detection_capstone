import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class TransformerClassifier:
    """
    A classifier that uses a transformer model to extract text embeddings and trains a machine learning model on these embeddings.

    Attributes:
        csv_file (str): Path to the CSV file containing the dataset.
        text_column (str): The column name in the CSV file containing the text data.
        label_column (str): The column name in the CSV file containing the labels.
        model_name (str): The name of the transformer model to use.
        classifier_type (str): The type of classifier to use ('logistic_regression', 'svm', or 'random_forest').
        model (AutoModel): The transformer model used for embedding extraction.
        tokenizer (AutoTokenizer): The tokenizer used with the transformer model.
        classifier: The machine learning classifier used for prediction.
    """
    def __init__(self, csv_file, text_column, label_column, model_name="bert-base-uncased", classifier_type="logistic_regression"):
        """
        Initializes the TransformerClassifier with the given parameters.

        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            text_column (str): The column name in the CSV file containing the text data.
            label_column (str): The column name in the CSV file containing the labels.
            model_name (str): The name of the transformer model to use. Default is "bert-base-uncased".
            classifier_type (str): The type of classifier to use ('logistic_regression', 'svm', or 'random_forest'). Default is "logistic_regression".
        """
        self.csv_file = csv_file
        self.text_column = text_column
        self.label_column = label_column
        self.model_name = model_name
        self.classifier_type = classifier_type
        self.model = None
        self.tokenizer = None
        self.classifier = None

    def load_data(self):
        """
        Loads the data from the CSV file and extracts the texts and labels into lists.
        """
        data = pd.read_csv(self.csv_file)
        self.texts = data[self.text_column].tolist()
        self.labels = data[self.label_column].tolist()

    def preprocess_data(self):
        """
        Preprocesses the data by loading the transformer model and tokenizer, and tokenizing the text data.
        """
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.input_sequences = self.tokenizer(self.texts, padding=True, truncation=True, return_tensors="pt")

    def extract_embeddings(self):
        """
        Extracts embeddings from the transformer model for the tokenized text data.
        """
        with torch.no_grad():
            self.embeddings = self.model(self.input_sequences["input_ids"], attention_mask=self.input_sequences["attention_mask"])[0][:, 0, :]

    def train_classifier(self):
        """
        Trains the specified classifier on the extracted embeddings and evaluates it on a validation set.
        """
        X_train, X_val, y_train, y_val = train_test_split(self.embeddings.numpy(), self.labels, test_size=0.3, random_state=42)
        if self.classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(max_iter=500)
        elif self.classifier_type == "svm":
            self.classifier = SVC()
        elif self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier()
        else:
            raise ValueError("Invalid classifier type. Choose from 'logistic_regression', 'svm', or 'random_forest'.")

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")

    def predict(self, text):
        """
        Predicts the label for a given text using the trained classifier.

        Args:
            text (str): The input text to classify.

        Returns:
            The predicted label for the input text.
        """
        input_sequence = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model(input_sequence["input_ids"], attention_mask=input_sequence["attention_mask"])[0][:, 0, :]
        prediction = self.classifier.predict(embedding.numpy())
        return prediction[0]

    def run(self):
        """
        Executes the complete pipeline: loading data, preprocessing, extracting embeddings, and training the classifier.
        """
        self.load_data()
        self.preprocess_data()
        self.extract_embeddings()
        self.train_classifier()
    

# Usage
# classifier = TransformerClassifier("/content/original.csv", "transcription", "class", classifier_type="logistic_regression")
# classifier.run()