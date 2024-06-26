# Alzheimer's Classification Web App

This project provides a web application for Alzheimer's classification based on audio recordings of people undergoing the cookie test. The application uses pre-trained models to predict the presence of dementia. This guide will help you set up and run the application, even if you are not familiar with coding.

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)

## Project Structure

```
app/
│
├── model/
│   ├── embed_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── sum_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── model.py
├── README.md
└── requirements.txt
```


- `app/`: Main project directory.
- `app/model/`: Contains the saved model files. Please contact us for the pre-trained model files
   - `embed_vectorizer.pkl`
   - `logistic_regression.pkl`
   - `sum_model.pkl` 
   - `tfidf_vectorizer.pkl`
- `app/app.py`: Main file to run the web application.
- `app/model.py`: Contains the code for model computations and predictions.
- `README.md`: This file.
- `requirements.txt`: List of Python libraries required to run the code.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have a computer with internet access.
- You have installed Python 3.7 or higher. If not, download and install it from [Python's official website](https://www.python.org/).

## Installation

1. **Clone the repository** or **download the zip file**:
   ```sh
   git clone https://github.com/M-I-Dx/alzheimers-detection.git
   cd alzheimers-detection/
   ```
 2. **Navigate to the project directory:**
     ```sh
     cd app
     ```
3. **Install the required libraries:** First, ensure you have pip installed. Then run:
    ```sh
    pip install setuptools-rust
    pip install -r requirements.txt
    pip install -U openai-whisper
    ```
    Note: You may also want to install these packages for the transcription model used in the code:
   ```sh
   # on Ubuntu or Debian
   sudo apt update && sudo apt install ffmpeg
   
   # on Arch Linux
   sudo pacman -S ffmpeg
   
   # on MacOS using Homebrew (https://brew.sh/)
   brew install ffmpeg
   
   # on Windows using Chocolatey (https://chocolatey.org/)
   choco install ffmpeg
   
   # on Windows using Scoop (https://scoop.sh/)
   scoop install ffmpeg
   ```

## Running the Application
To launch the web application, follow these steps:
1. **Navigate to the `app` directory if you are not already there:**
    ```
    cd app
    ```
2. **Run the application:**
    ```
    python app.py
    ```
3. **Open the web application:**
    > Once the application is running, it will provide a local URL (e.g., http://127.0.0.1:7860). Open this URL in your web browser to access the application.

## Usage

1. Upload Audio File:
    In the web application, upload the audio recording of the person undergoing the cookie test.

2. Run Prediction:
    Click the button to run the prediction. The application will process the audio and display the prediction results, indicating whether the person is likely to have dementia.
