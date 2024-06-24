This is `data` folder for the Alzheimer's project.

Because of license restriction, we are not able to upload the data here. Please register with DementiaBank to download the ADReSS and Pitt datasets.

The `data` folder contains several sub-folders. It is important to copy your downloaded data according to the structure below for the code to run properly. If needed, please update the code to the corresponding folders containing the datasets.


- `Ground Truth for test sets`: contains the gold labels for test set (`task1.csv` and `task2.csv`) from the data organizer. 
- `pitt`: contains the intermediate outputs for the Pitt datasets (e.g. `pitt_train.csv`, `pitt_dev.csv`, etc). It's a blank folder at the beginning.
- `splits`: contains the intermediate outputs for the splits of the ADReSS dataset (e.g. `train_50pc.csv`, `dev_50pc.csv`, `test_text.csv`, etc). It's a blank folder at the beginning.
- `TalkBank downloads`: contains several extra datasets, the most important one is `Pitt`
  - `Pitt`: contains the Pitt dataset, in 2 subfolders `Control` and `Dementia`
    - `Control/cookie`: contains data (`.cha` files) for Control (or normal) group
    - `Dementia/cookie`: contains data (`.cha` files) for the Dementia (or possible Alzheimer's Disease) group
- `Test Dataset`: contains testset for the ADReSS dataset, including audio and segmentation
- `Training Dataset`: contains train set for the ADReSS dataset, including audio files and segmentation
- `Transcribed Data-2024-Feb-13`: contains transcription files (`.txt` for both train and test set for ADReSS dataset). This is a very important folder. 
  - `Diagnosis Training Data`: transcriptions for Train set
    - `ad transcribed`: text files for AD (Alzheimer's Disease) group
    - `cn transcribed`: text files for CN (Cognitively Normal) group
  - `Diagnosis Testing Data`: transcriptions for test set
  
