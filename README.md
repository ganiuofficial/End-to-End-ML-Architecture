# BAN6440 Final Project — Otomoto Customer Churn Prediction Pipeline

## End-to-End ML Architecture \& Interoperability Proposal

## Nexford University | May 2026

**Author:** Ganiu Olalekan Mustapha

\---

## Project Overview

This project designs and demonstrates a production-minded end-to-end machine
learning pipeline that solves a real business problem for Otomoto, a telecom
operator, using the Teleconnect customer dataset.

The business decision: which customers should Otomoto's marketing team contact
with a retention offer before they churn? The pipeline produces a churn
probability score for every active customer, ranked from highest to lowest risk,
enabling targeted, cost-efficient intervention.

**Verified output metrics (from actual run):**

|Metric|LR Baseline|ANN / RMSProp|ANN 5-Fold CV Mean ± SD|
|-|-|-|-|
|Accuracy|80.70%|78.35%|80.00% ± 0.84%|
|Precision (Churn)|65.84%|59.72%|65.49% ± 3.06%|
|Recall (Churn)|56.68%|56.68%|52.71% ± 4.95%|
|F1-Score (Churn)|60.92%|58.16%|58.19% ± 2.76%|
|AUC-ROC|84.16%|83.65%|83.89% ± 1.28%|
|Best val\_loss|N/A|0.478|—|
|Training time|0.18s|35.3s|—|

**Key findings:**

* Logistic Regression outperforms the ANN on all metrics on this 7k-row
tabular dataset — expected and documented honestly in the report.
* ANN best threshold = 0.30 (F1 = 62.17%, Recall = 76.47%) — at this
threshold the ANN catches more churners than the baseline, making it
the deployment recommendation when recall is the priority metric.
* Top churn risk features: Fiber optic internet, high TotalCharges,
StreamingMovies/TV subscriptions.
* Top retention features: Long tenure, high MonthlyCharges (counter-
intuitive — investigated in report), two-year contract, OnlineSecurity.

\---

## Project Structure

```
otomoto\\\_final\\\_project/
|
|-- otomoto\\\_final\\\_pipeline.py        <- Main Python pipeline (8 sections)
|-- requirements.txt                 <- Pinned library versions
|-- README.md                        <- This file
|-- teleconnect.csv                  <- Dataset (provided by assignment)
|-- BAN6440\\\_FinalProject\\\_Report.docx <- Full written report (10-15 pages)
|-- Model\\\_Card.docx                  <- Model card (1-2 pages)
|-- architecture\\\_diagram.png         <- Pipeline diagram (separate file)
|
|-- otomoto\\\_final\\\_outputs/           <- Created automatically on run
    |-- 01\\\_data\\\_overview.png         <- EDA: churn distribution, charges, tenure, contract
    |-- 02\\\_model\\\_comparison\\\_table.png<- Baseline vs ANN summary (green = recall row)
    |-- 03\\\_training\\\_history.png      <- ANN loss \\\& accuracy curves
    |-- 04\\\_cross\\\_validation.png      <- 5-fold CV metrics per fold
    |-- 05\\\_threshold\\\_analysis.png    <- Precision/recall trade-off, t=0.20-0.70
    |-- 06\\\_feature\\\_importance\\\_roc.png<- LR coefficients + overlaid ROC curves
    |-- 07\\\_calibration.png           <- Calibration curve + Brier scores
    |-- 08\\\_confusion\\\_matrices.png    <- Baseline vs ANN confusion matrices
    |-- data\\\_quality\\\_report.json     <- Before/after cleansing quality gate
    |-- metrics\\\_full.json            <- All metrics, CV summary, best threshold
    |-- model\\\_comparison.csv         <- Comparison table as CSV
    |-- otomoto\\\_churn\\\_model.keras    <- Saved ANN model weights
    |-- terminal\\\_output.txt          <- Full terminal output from the run
```

\---

## Dataset

**Name:** Teleconnect Customer Churn Dataset
**Provided by:** BAN6440 Final Project assignment
**Records:** 7,043 customers | 20 raw columns → 30 encoded features
**Target:** Churn (Yes=1 / No=0)
**Class balance:** 26.5% churned (1,869) / 73.5% retained (5,174)

**Before cleansing:**

* Rows: 7,043 | Columns: 21 (including customerID)
* Null values: 0 | Whitespace in TotalCharges: 11 | Duplicates: 0

**After cleansing:**

* Rows: 7,043 | Encoded features: 30 | Null values: 0
* Quality gate: PASSED (assert zero NaN before training)

\---

## Requirements

### Python Version

Python 3.9 or higher recommended.

### Install Dependencies

Open VS Code integrated terminal (Ctrl + `) and run:

&#x20;   pip install -r requirements.txt



Or install manually:

&#x20;   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn



Note for Windows — if TensorFlow fails:

&#x20;   pip install tensorflow-cpu



\---

## How to Run in VS Code

### Step 1 - Open the project folder

&#x20;   File -> Open Folder -> select the otomoto\_final\_project folder



### Step 2 - Ensure teleconnect.csv is in the same folder as the .py file

### Step 3 - Open integrated terminal

&#x20;   Terminal -> New Terminal   (or Ctrl + `)



### Step 4 - Install dependencies

&#x20;   pip install -r requirements.txt



### Step 5 - Run the pipeline

&#x20;   python otomoto\_final\_pipeline.py



### Step 6 - Expected runtime

Approximately 210 seconds (3.5 minutes) on CPU.
The pipeline runs 8 sections sequentially and prints progress to terminal.

### Step 7 - Expected terminal output (condensed)

&#x20;   =================================================================
BAN6440 Final Project | Otomoto Customer Churn Pipeline
Baseline: Logistic Regression | Primary: ANN / RMSProp
Dataset : teleconnect.csv (7,043 customers)
=================================================================

&#x20;   SECTION 1: DATA LOADING \\\& QUALITY AUDIT
       Raw shape         : (7043, 21)
       Churn=Yes         : 1869 (26.5%)
       Churn=No          : 5174 (73.5%)
       BEFORE cleansing:
          total\\\_rows     : 7043 | null\\\_count : 0
          whitespace\\\_totalcharges : 11 | duplicate\\\_rows : 0

    SECTION 2: PREPROCESSING PIPELINE
       TotalCharges NaN  : 11 imputed with median ($1397.47)
       Outlier cells     : 0 in \\\[tenure, MonthlyCharges, TotalCharges]
       AFTER cleansing:
          total\\\_cols : 31 | feature\\\_count : 30 | null\\\_count : 0
       \\\[OK] Data quality gate passed — zero NaN in feature matrix
       Train set : (5634, 30) | Test set : (1409, 30)

    SECTION 3: BASELINE — LOGISTIC REGRESSION
       Training time  : 0.18s
       Accuracy       : 80.70% | Precision : 65.84%
       Recall         : 56.68% | F1        : 60.92% | AUC : 84.16%

    SECTION 4: PRIMARY MODEL — ANN / RMSProp
       Epochs run     : 57 (best weights restored)
       Training time  : 35.3s
       Accuracy       : 78.35% | Precision : 59.72%
       Recall         : 56.68% | F1        : 58.16% | AUC : 83.65%
       Best val\\\_loss  : 0.478

    SECTION 5: CROSS-VALIDATION (5-FOLD STRATIFIED)
       Fold 1: acc=0.7941  rec=0.5418  f1=0.5827  auc=0.8360
       Fold 2: acc=0.7879  rec=0.4849  f1=0.5482  auc=0.8191
       Fold 3: acc=0.8119  rec=0.5719  f1=0.6173  auc=0.8348
       Fold 4: acc=0.8004  rec=0.5819  f1=0.6073  auc=0.8568
       Fold 5: acc=0.8055  rec=0.4548  f1=0.5540  auc=0.8477
       accuracy : 80.00% +/- 0.84%  (95% CI: 78.35% to 81.64%)
       recall   : 52.71% +/- 4.95%  (95% CI: 43.01% to 62.40%)
       f1       : 58.19% +/- 2.76%  (95% CI: 52.78% to 63.60%)
       auc      : 83.89% +/- 1.28%  (95% CI: 81.39% to 86.39%)

    SECTION 6: THRESHOLD ANALYSIS
       ANN best threshold (max F1) : 0.30
       At t=0.30 — Precision: 52.38%  Recall: 76.47%  F1: 62.17%

    SECTION 7: FEATURE IMPORTANCE
       Top churn risk: Fiber optic (+0.78), TotalCharges (+0.50)
       Top retention: tenure (-1.22), MonthlyCharges (-0.92),
                      Contract\\\_Two year (-0.59)

    SECTION 8: CALIBRATION \\\& MODEL SAVE
       Brier score (Logistic Regression): 0.1383
       Brier score (ANN / RMSProp)      : 0.1411
       Total pipeline time              : 210s

    \\\[OK] Final pipeline completed successfully.





### Step 8 - View output files

All plots and saved files appear in the otomoto\_final\_outputs/ folder.

\---

## Pipeline Architecture (8 Sections)

|Section|Task|Report Section|Key Output|
|-|-|-|-|
|1|Data Loading \& Quality Audit|2|01\_data\_overview.png|
|2|Preprocessing Pipeline|2|data\_quality\_report.json|
|3|Logistic Regression Baseline|3|02\_model\_comparison\_table.png|
|4|ANN / RMSProp Primary Model|3|03\_training\_history.png|
|5|5-Fold Cross-Validation|3|04\_cross\_validation.png|
|6|Threshold Analysis|5|05\_threshold\_analysis.png|
|7|Feature Importance + ROC|5|06\_feature\_importance\_roc.png|
|8|Calibration + Model Save|5 \& 7|07\_calibration.png, .keras model|

\---

## Key Design Decisions

**Why Logistic Regression as baseline:**
Interpretable coefficients map directly to business recommendations.
On 7k-row tabular data, LR often matches or outperforms ANNs — this is
confirmed by the results (LR F1=60.92% vs ANN F1=58.16%).

**Why RMSProp for the ANN:**
Teleconnect has 26 binary one-hot columns creating sparse gradient signals.
At batch\_size=64 with 26.5% churn rate, only \~17 minority-class examples
appear per batch — high gradient variance for the churn signal. RMSProp's
per-parameter adaptive learning rate directly addresses this heterogeneity
(Tieleman \& Hinton, 2012).

**Why threshold = 0.30 for deployment:**
At t=0.50, ANN recall = 56.68% (same as LR). At t=0.30, recall rises to
76.47% — catching significantly more churners. The commercial asymmetry:
a false negative (missed churner) = full lifetime revenue loss. A false
positive (loyal customer receives retention offer) = cost of the discount
only. This asymmetry justifies the lower threshold.

**Why pos\_label=1 is explicit in every metric call:**
Churn=Yes is encoded as 1. Without explicit pos\_label=1, sklearn defaults
may compute metrics for the wrong class. This was the critical error in
Module 5 that cost 3.5 marks — corrected here throughout.

\---

## Output Files Explained

|File|What it shows|
|-|-|
|01\_data\_overview.png|Churn distribution, monthly charges, tenure,|
||contract type — 4-panel EDA|
|02\_model\_comparison\_table.png|LR vs ANN metrics table (recall row highlighted)|
|03\_training\_history.png|ANN loss \& accuracy curves across 57 epochs|
|04\_cross\_validation.png|All metrics across 5 CV folds|
|05\_threshold\_analysis.png|Precision/recall/F1 vs threshold (0.20 to 0.70)|
|06\_feature\_importance\_roc.png|Top 15 LR coefficients + overlaid ROC curves|
|07\_calibration.png|Calibration curve: predicted prob vs actual rate|
|08\_confusion\_matrices.png|LR vs ANN confusion matrices (pos\_label=1)|
|data\_quality\_report.json|Before/after cleansing quality gate evidence|
|metrics\_full.json|All metrics, CV summary, best threshold|
|model\_comparison.csv|Model comparison table as CSV|
|otomoto\_churn\_model.keras|Saved ANN model (deploy-ready)|
|terminal\_output.txt|Full terminal output from the pipeline run|



\---

## Troubleshooting

**ModuleNotFoundError: No module named 'tensorflow'**

&#x20;   pip install tensorflow



**FileNotFoundError: teleconnect.csv not found**

&#x20;   Ensure teleconnect.csv is in the SAME folder as
otomoto\_final\_pipeline.py. The code reads:
df = pd.read\_csv('teleconnect.csv')
No path prefix — file must be in the working directory.



**Pipeline runs but Logistic Regression F1 is 0%**

&#x20;   Check that Churn is encoded as Yes=1 before the split.
The preprocessing step assigns:
df = df.assign(Churn=(df\['Churn'] == 'Yes').astype(int))
pos\_label=1 is passed to all metric calls explicitly.



**Plots not appearing as pop-up windows**

&#x20;   Expected. The code uses matplotlib Agg backend which saves
plots as PNG files. Check otomoto\_final\_outputs/.



**TensorFlow GPU/CPU warnings in terminal**

&#x20;   Harmless. Suppressed by:
os.environ\['TF\_CPP\_MIN\_LOG\_LEVEL'] = '3'
(already in the code)



\---

## References

Kingma, D. P., \& Ba, J. (2015). Adam: A method for stochastic
optimization. ICLR 2015. https://arxiv.org/abs/1412.6980

Tieleman, T., \& Hinton, G. (2012). RMSProp: Lecture 6e, Neural
Networks for Machine Learning. COURSERA.

Ioffe, S., \& Szegedy, C. (2015). Batch normalization: Accelerating
deep network training by reducing internal covariate shift. ICML 2015.

Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python.
Journal of Machine Learning Research, 12, 2825-2830.

\---

BAN6440 Business Analytics | Nexford University | May 2026

