import pandas as pd
from setfit import SetFitModel, sample_dataset, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class SetFitClassifier:
    """
    A classifier class utilizing the SetFit framework for fine-tuning and evaluating models.

    Attributes:
        model (SetFitModel): The pretrained SetFit model.
        model_name (str): The name of the pretrained model.
        train_csv (str): Path to the training dataset CSV file.
        dev_csv (str): Path to the development dataset CSV file.
        test_csv (str): Path to the test dataset CSV file.
        text_column (str): The column name containing text data.
        label_column (str): The column name containing label data.
    """

    def __init__(self, model_name, train_csv, dev_csv, test_csv, text_column, label_column):
        """
        Initializes the SetFitClassifier with the provided model name and dataset paths.

        Args:
            model_name (str): The name of the pretrained model to use.
            train_csv (str): Path to the training dataset CSV file.
            dev_csv (str): Path to the development dataset CSV file.
            test_csv (str): Path to the test dataset CSV file.
            text_column (str): The column name containing text data.
            label_column (str): The column name containing label data.
        """
        self.model = SetFitModel.from_pretrained(model_name)
        self.model_name = model_name
        self.train_csv = train_csv
        self.dev_csv = dev_csv
        self.test_csv = test_csv
        self.text_column = text_column
        self.label_column = label_column

    def load_data(self, num_samples):
        """
        Loads and samples the datasets.

        Args:
            num_samples (int): The number of samples to use from the training dataset.
        """
        dataset = load_dataset('csv', data_files={
            'train': self.train_csv,
            'dev': self.dev_csv,
            'test': self.test_csv
        })
        self.train_dataset = sample_dataset(dataset['train'], label_column=self.label_column, num_samples=num_samples)
        self.dev_dataset = dataset['dev']
        self.test_dataset = dataset['test']

    def finetuning(self, batch_size, num_epochs):
        """
        Fine-tunes the model with the provided batch size and number of epochs.

        Args:
            batch_size (int): The batch size to use for training.
            num_epochs (int): The number of epochs to train for.
        """
        args = TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        args.eval_strategy = args.evaluation_strategy
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            column_mapping={self.text_column: "text", self.label_column: "label"}
        )
        self.trainer.train()

    def evaluate(self):
        """
        Evaluates the model on the development dataset.

        Returns:
            dict: A dictionary containing evaluation metrics (f1, precision, recall, accuracy).
        """
        # Predictions on the dev set
        preds = self.model.predict(self.dev_dataset[self.text_column])
        labels = self.dev_dataset[self.label_column]
        # Compute metrics: Accuracy, Precision, Recall, F1 via sklearn
        f1 = f1_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}

# Example usage
# model_name = "bert-base-uncased"
# csv_file = "./Data/Main Study/training/transcriptions/original.csv"
# text_column = "transcription"
# label_column = "class"
# num_samples = 10

# classifier = SetFitClassifier(model_name, csv_file, text_column, label_column)
# classifier.load_data(num_samples)
# classifier.finetuning(batch_size=2, num_epochs=1)
# classifier.evaluate()

