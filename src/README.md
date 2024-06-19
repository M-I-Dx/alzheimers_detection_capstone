## Code Execution Instructions

In order to run our code, please follow the following instruction:
1. Install standard data science environment (Python, Numpy, Pandas, Jupyter Notebook, etc)
2. Install SBert text embeddings and Bert summarizer libraries with specific versions as follow:
    - `pip install sentence-transformers==2.7.0`
    - `pip install bert-extractive-summarizer==0.10.1`
3. Clone our repo by
    - `git clone https://github.com/M-I-Dx/alzheimers-detection.git`
4. Download DementiaBank ADReSS dataset and Pitt dataset, place them in the data folder with structure the same as in the final report

6. Run data preprocessing code:
    - Original dataset:
        - `/src/data preprocessing/original/01_process_test_set.ipynb`
    - Pitt dataset:
        - `/src/data_preprocessing/pitt/00_split_data.ipynb`
        - `/src/data_preprocessing/pitt/01_preprocessing_pitt.ipynb`
        - `/src/data_preprocessing/pitt/02_data_splits_pitt.ipynb`
7. Run the EDA code:
    - Original dataset
        - `/src/EDA/original/01_EDA_data_splits.ipynb`
    - Pitt dataset
        - `/src/EDA/pitt/01_eda_pitt.ipynb`
8. Classical (Tfidf) baseline model:
    - Original dataset:
        - `/src/classical_baseline/original/01_classical_baseline_original.ipynb`
    - Pitt dataset:
        - `/src/classical_baseline/pitt/01_classical_baseline_Pitt.ipynb`
9. Neural classifier model:
    - Original dataset:
        - `/src/neural_classification/original/01_neural_classification_original.ipynb`
    - Pitt dataset:
        - `/src/neural_classification/pitt/01_neural_classification_pitt.ipynb`
10. Hybrid model:
    - Original dataset:
        - `/src/hybrid/original/01_hybrid_original.ipynb`
    - Pitt dataset:
        - `/src/hybrid/pitt/01_hybrid_pitt.ipynb`
11. Evaluation code (for original dataset)
    - go to `/src/evaluation/` folder
        - run `python evaluate.py task1 gold.csv test_results_task1_classical.csv` for classical model
        output
        - run `python evaluate.py task1 gold.csv test_results_task1_neural.csv` for neural classifer model
        output
        - run `python evaluate.py task1 gold.csv test_results_task1_hybrid.csv` for hybrid model
        output
