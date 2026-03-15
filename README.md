# alzheimers-detection

Capstone project GDRL for Alzheiner's Disease early detection.

Link to blog: https://medium.com/@nguyentranminh/cognitive-assessment-and-role-of-computational-linguistics-a3d1a86eb3c2  
Link to final paper: https://github.com/M-I-Dx/alzheimers-detection/blob/main/GDRL_Final_Report__Draft___V1_.pdf

Team members:
- Mandeep
- Minh Nguyen
- Qihan Li
- Daniel Jimenez

Supervisor / Mentor:
- Jian Zhu


### Objectives of the project
This is a Capstone project, part of Master of Data Science, Computational Linguistics at University of British Columbia, Canada.
The main objective of the project is building models to predict early signs of Alzheimer's Disease in people, based on their speech about a particular picture / drawing called The Cookie Thief. 

There are 2 main tasks:
1. Classification:
  Based on the text transcription (and optionally audio) of a person's speech about the drawing, the model will classify them as AD (Alzheimer's Disease) or CN (Cognitively Normal). 

2. Regression:
  Based on the same input (text and/or audio), the model will predict the MMSE (Mini Mental State Examination) scores of a person.

### Organization of the code repo
The repo are organized into 3 main folders: `app`, `data` and `src`.
- `app`: deployment code with a simple user interface.
- `data`: contains the data necessary for training, intermediate data outputs. Please see README.md in `data` folder for more information how the data is organized.
- `src`: contains the code for data-preprocessing, EDA (Exploratory Data Analysis) and various models (classical_baseline, hybrid, neural_classification, etc). Please see the README.md file in the `src` folder for more information how to run these models. 
