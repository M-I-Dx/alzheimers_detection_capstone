# this is the evaluation script for task 1 (classification)
# arg0: gold filename
# arg1: prediction filename

import sys
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, classification_report

def get_fnames():
    """
        Get the filenames from arguments
        Return gold (ground truth) and sys (prediction) filenames
    """
    # print("Arguments =", sys.argv)
    if len(sys.argv) < 3:
        print("Not enough information, please run evaluate.py with following arguments:")
        print("evaluate_py gold_file prediction_file")
        print("Example: python evaluate_py task1_gold.csv test_results_task1.csv")

    assert len(sys.argv) == 3
        
    gold_fname = sys.argv[1]
    sys_fname = sys.argv[2]
    
    return (gold_fname, sys_fname)

def load_data(fname):
    """
        Load data from filename to a data frame
        Return dataframe
    """
    result = None
    print(f"Loading data from {fname}...", end="") 
    try:
        result = pd.read_csv(fname)
        print("done")
        pass
    except Exception as e:
        print("Error loading file. Error =", e)
        
    return result

def evaluate(gold_df, sys_df):
    """
    """
    print("Gold len =", len(gold_df))
    print("Prediction len =", len(sys_df))

    result = None
    if len(gold_df) != len(sys_df):
        print("Error. The gold and prediction sets should have the same length.")

    gold = gold_df["target"]
    sys = sys_df["Prediction"]

    accuracy = accuracy_score(gold, sys)
    recall = recall_score(gold, sys)

    print("Classification report:")
    print(classification_report(gold, sys))
    
    return (accuracy, recall)
    
    # return result

def main():
    gold_fname, sys_fname = get_fnames()
    gold_df = load_data(gold_fname)
    sys_df = load_data(sys_fname)
    accuracy, recall = evaluate(gold_df, sys_df)
    print("Accuracy =", accuracy)
    print("Recall =", recall)
    

if __name__ == '__main__':
    main()