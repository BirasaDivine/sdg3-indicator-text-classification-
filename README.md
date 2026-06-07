# SDG 3 Indicator Text Classification
### Multi-Label NLP Classification | Group 12

This repository contains the full pipeline for predicting which Sustainable Development Goal 3 (SDG 3) health indicators are relevant to a given text document. The system is built as a multi-label classification pipeline trained on approximately 3,000 real-world documents from tenders, grants, contracts, and news articles in the international development sector.

The best configuration, a soft probability ensemble with per-label threshold tuning, achieved a validation Hamming Loss of **0.0421**.

---

## Demo Video

demo video link: https://drive.google.com/file/d/1jMQZZ-LJmRRkBDrdRCqDXkb6B_kDPUnr/view?usp=sharing(#)

---

## Group Members and Contributions

| Member | Notebook | Responsibility |
|---|---|---|
| Birasa Divine | `text_classification.ipynb` | Dataset preprocessing and feature engineering |
| Birasa Divine | `Exp8_9.ipynb` | Experiments 8 to 10: hyperparameter sweep, ensemble learning, and final inference |
| Emmanuella Briggs | `SDG_text_classification(1_4).ipynb` | Experiments 1 to 4: TF-IDF baselines with Logistic Regression, SVM, combined features, and class balancing |
| Leny Pascal Ihirwe | `SDG3_Text_Classification_Leny.ipynb` | Experiments 5 to 7: threshold tuning, class imbalance handling, and DistilBERT fine-tuning |

The test set preprocessing is handled in `test_preprocessing.ipynb`, which applies the same cleaning and vectorization pipeline from Birassa's notebook to the unlabeled test set.

---

## Repository Structure

```
sdg3-indicator-text-classification/
│
├── text_classification.ipynb          # Preprocessing and feature engineering (Birassa)
├── SDG_text_classification(1_4).ipynb # Experiments 1 to 4 (Emmanuella)
├── SDG3_Text_Classification_Leny.ipynb # Experiments 5 to 7 (Leny)
├── Exp8_9.ipynb                       # Experiments 8 to 10 (Birasa)
├── test_preprocessing.ipynb           # Test set preprocessing
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## How to Reproduce

### Step 1 -- Clone the repository

```bash
git clone https://github.com/BirasaDivine/sdg3-indicator-text-classification-.git
```

### Step 2 -- Set up the environment

All notebooks are designed to run on **Google Colab** with a standard CPU runtime. No GPU is required except for `SDG3_Text_Classification_Leny.ipynb` which fine-tunes DistilBERT and requires a T4 GPU runtime.

To enable GPU in Colab: Runtime > Change runtime type > T4 GPU

Install dependencies by running the first cell in any notebook, or manually:

```bash
pip install -r requirements.txt
```

### Step 3 -- Upload the datasets

Upload the following files to `/content/` in your Colab session:

- `Devex_train.csv` -- training set with labels
- `Devex_test_questions.csv` -- test set without labels

Both files are available from the course assignment page on Canvas.

### Step 4 -- Run the notebooks in order

The notebooks must be run in the following order because each one saves processed data to Google Drive that the next one loads:

1. `text_classification.ipynb` -- cleans text, builds TF-IDF and SBERT features, saves all processed matrices to `/content/drive/MyDrive/SDG_Assignment/processed_data/`
2. `test_preprocessing.ipynb` -- applies the same pipeline to the test set and saves test features to the same Drive folder
3. `SDG_text_classification(1_4).ipynb` -- loads processed data, runs Experiments 1 to 4, saves best model to `/content/drive/MyDrive/SDG_Assignment/models/`
4. `SDG3_Text_Classification_Leny.ipynb` -- loads processed data, runs Experiments 5 to 7
5. `Exp8_9.ipynb` -- loads processed data and saved models, runs Experiments 8 to 10 and generates final test predictions

### Step 5 -- Mount Google Drive

Every notebook after the first requires Google Drive to be mounted. Each notebook contains a Drive mount cell at the top. When prompted, sign in with the Google account that has access to the shared `SDG_Assignment` folder.

---

## Evaluation Metric

The primary evaluation metric is **Hamming Loss** -- the fraction of label predictions that are incorrect, averaged across all 27 indicators and all documents. Lower is better, with 0.0 representing perfect classification.

Macro F1 is reported alongside Hamming Loss in all experiments to capture model behaviour on rare indicators, which Hamming Loss alone can obscure.

---

## Experiment Summary

| Experiment | Description | Hamming Loss |
|---|---|---|
| Exp 10 | Soft Ensemble + Per-Label Thresholds | **0.0421** |
| Exp 9 | Majority Vote Ensemble | 0.0442 |
| Exp 5 | LR + Per-Label Tuned Thresholds | 0.0444 |
| Exp 3 | Word+Char TF-IDF + Linear SVM | 0.0449 |
| Exp 8 | LinearSVC C Sweep (best C=1.0) | 0.0449 |
| Exp 2 | TF-IDF Word + Linear SVM | 0.0464 |
| Exp 4 | Linear SVM + class_weight=balanced | 0.0483 |
| Exp 6 | Balanced LR + Tuned Thresholds | 0.0529 |
| Exp 1 | TF-IDF Word + Logistic Regression (Baseline) | 0.0589 |
| Exp 7 | DistilBERT Fine-tuned | 0.0669 |

---

## Key Dependencies

- Python 3.10+
- scikit-learn
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- sentence-transformers
- transformers
- torch
- joblib
- nltk

---

## Notes

- All random seeds are fixed at 42 across every notebook for reproducibility.
- The processed data folder in Google Drive must be shared with all team members before running notebooks 2 through 5.
- The DistilBERT experiment in `SDG3_Text_Classification_Leny.ipynb` requires a GPU runtime and takes approximately 15 to 20 minutes to complete.
- Test predictions are saved to `/content/drive/MyDrive/SDG_Assignment/results/` as a CSV file upon completion of `Exp8_9.ipynb`.
