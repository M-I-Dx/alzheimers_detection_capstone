# alzheimers-detection

Capstone project GDRL for Alzheimer's Disease early detection.

Link to blog: https://medium.com/@nguyentranminh/cognitive-assessment-and-role-of-computational-linguistics-a3d1a86eb3c2  
Capstone report: [Final project report](GDRL_Final_Report__Draft___V1_.pdf)

### Publication and citation

This project led to the following paper, published in *PLOS Digital Health* on June 25, 2026. If you use this work, please cite:

Madden KM, Feldman B, Jimenez D, Khokhar M, Li Q, Nguyen M, Zhu J (2026). [Design of an artificial intelligence model to screen spontaneous speech to detect Alzheimer’s Disease](https://doi.org/10.1371/journal.pdig.0001444). *PLOS Digital Health*, 5(6): e0001444.

```bibtex
@article{madden2026speech,
  author = {Madden, Kenneth M. and Feldman, Boris and Jimenez, Daniel and Khokhar, Mandeep and Li, Qihan and Nguyen, Minh and Zhu, Jian},
  title = {Design of an artificial intelligence model to screen spontaneous speech to detect {Alzheimer's Disease}},
  journal = {PLOS Digital Health},
  year = {2026},
  volume = {5},
  number = {6},
  pages = {e0001444},
  doi = {10.1371/journal.pdig.0001444},
  url = {https://doi.org/10.1371/journal.pdig.0001444}
}
```

### Project team

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
