# Human Activity Recognition Using Hidden Markov Models (HMM)

This project implements a **Human Activity Recognition** system using the **Hidden Markov Model (HMM)** algorithm.  
It classifies human activities based on time-series sensor data (accelerometer and gyroscope) by learning the statistical dependencies between temporal observations.

The workflow includes **data preprocessing**, **feature extraction**, **model training**, and **performance evaluation** on test datasets.

---

## Project Structure

Data/
│
├── extracted_datasets/
│ ├── extracted_train_features.csv
│ └── extracted_test_features.csv
│
├── merged_datasets/
│ ├── combined_train_data.csv
│ └── combined_test_data.csv
│
├── Test_unprocessed_data/
└── Train_unprocessed_data/

evaluation_unseen_results.csv
extraction.py
Hidden Markov Model(4).pdf
HMM_Implementation.py
merge_acc_gyr.py
README.md
trained_hmm_discrete.pkl
trained_scaler.pkl
trained_supervised_hmm.pkl


---

## Implementation Overview

### 1. Data Preprocessing
- Conversion and merging of accelerometer and gyroscope sensor readings.  
- Cleaning inconsistent or missing data.  
- Generation of feature-rich datasets (time and frequency domain features).  
- Final datasets:  
  - `combined_train_data.csv`  
  - `combined_test_data.csv`

### 2. Model Training
- Implementation of **Hidden Markov Models (HMM)** using `hmmlearn`.  
- Training performed with the **Baum–Welch algorithm** for parameter estimation.  
- Feature normalization and state sequence modeling for each activity class.

### 3. Evaluation Metrics
After training, the model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- Classification Report  

These metrics measure how effectively the model recognizes human activities across the test samples.

---

## Technologies & Libraries Used

- **Python 3.10+**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **hmmlearn**
- **Matplotlib / Seaborn**
- **Joblib**

---

## Team Contributions

| Team Member | Contributions |
|--------------|---------------|
| **Christine** | • Data Preprocessing (format conversion, merging sensors, cleaning)<br>• Feature Extraction (time-domain & frequency-domain features)<br>• Data Exploration & Statistical Analysis<br>• Data Visualization (signal plots, PCA, correlation matrices, frequency analysis)<br>• Report Documentation |
| **Jean** | • Model Implementation (HMM architecture setup)<br>• Model Training (Baum–Welch algorithm, hyperparameter tuning)<br>• Model Evaluation (Viterbi decoding, confusion matrix, performance metrics)<br>• Report Documentation |

---

## Results Summary

The trained HMM achieved strong classification accuracy across activities such as **walking**, **standing**, **sitting**, **jumping**, and **still**.  
The confusion matrix and F1-scores demonstrate that the model effectively captures temporal dynamics from the sensor signals.

---

## How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/AkotoChristine/Hidden-Markov-Model.git
   cd HIDDEN-MARKOV-MODEL


2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the data extraction and merging scripts**
   ```bash
   python extraction.py
   python merge_acc_gyr.py
4. **Train and evaluate the model**
   ```bash
   python HMM_Implementation.py
5. View Results
    Console output will display accuracy, F1-score, and confusion matrix.
