# =============================================================================
# BAN6440 Final Project: End-to-End ML Pipeline - Otomoto Customer Churn
# =============================================================================
# Business Problem : Predict which Otomoto telecom customers will churn
#                    within the next billing cycle to enable targeted
#                    retention marketing campaigns.
# Dataset          : teleconnect.csv - 7,043 customers, 20 raw columns -> 30 encoded features
# Models           : Logistic Regression (baseline) + ANN / RMSProp (primary)
# Author           : Ganiu Olalekan Mustapha | Nexford University | May 2026
#
# SECTION MAP (maps to final project report sections):
#   Section 1  -> Report Section 2: Data Strategy & Quality Audit
#   Section 2  -> Report Section 2: Preprocessing Pipeline
#   Section 3  -> Report Section 3: Baseline Model
#   Section 4  -> Report Section 3: ANN Primary Model
#   Section 5  -> Report Section 3: Cross-Validation
#   Section 6  -> Report Section 5: Threshold Analysis & Decision Impact
#   Section 7  -> Report Section 5: Feature Importance
#   Section 8  -> Report Section 7: Calibration, Model Save, Deployment Artefacts
#
# pos_label NOTE (lesson from Module 5):
#   Churn=Yes is encoded as 1. pos_label=1 is passed explicitly to ALL
#   sklearn metric calls. This ensures metrics measure churn detection,
#   not retention performance.
#
# OPTIMISER JUSTIFICATION (lesson from Module 6):
#   RMSProp is selected for the ANN because the Teleconnect dataset contains
#   26 binary one-hot encoded columns creating sparse gradient signals during
#   mini-batch training. At batch_size=64, with only 26.5% churn rate, the
#   expected number of minority-class examples per batch is ~17, creating
#   high gradient variance for the churn signal. RMSProp's per-parameter
#   adaptive learning rate directly addresses this by scaling down noisy
#   gradients from sparse features and scaling up stable gradients from
#   informative dense features (Tieleman & Hinton, 2012).
# =============================================================================

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, json, time
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration   import calibration_curve, CalibratedClassifierCV
from sklearn.metrics       import (accuracy_score, precision_score,
                                   recall_score, f1_score, roc_auc_score,
                                   confusion_matrix, classification_report,
                                   roc_curve, brier_score_loss)

tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = "otomoto_final_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1: DATA LOADING & QUALITY AUDIT
# Produces a before/after quality report showing the cleansing impact.
# This is required evidence for Report Section 2.
# =============================================================================
def load_and_audit(path: str = "teleconnect.csv"):
    print("\n" + "="*65)
    print("  SECTION 1: DATA LOADING & QUALITY AUDIT")
    print("="*65)

    df = pd.read_csv(path)
    print(f"\n   Raw shape         : {df.shape}")
    print(f"   Churn=Yes         : {(df['Churn']=='Yes').sum()} ({(df['Churn']=='Yes').mean()*100:.1f}%)")
    print(f"   Churn=No          : {(df['Churn']=='No').sum()} ({(df['Churn']=='No').mean()*100:.1f}%)")

    # --- BEFORE quality snapshot ---
    before = {
        'total_rows'      : len(df),
        'total_cols'      : len(df.columns),
        'null_count'      : df.isnull().sum().sum(),
        'whitespace_totalcharges': (pd.to_numeric(df['TotalCharges'],
                                    errors='coerce').isnull().sum()),
        'duplicate_rows'  : df.duplicated().sum(),
        'churn_rate_pct'  : round((df['Churn']=='Yes').mean()*100, 2),
    }
    print(f"\n   BEFORE cleansing:")
    for k, v in before.items():
        print(f"      {k:35s}: {v}")

    _plot_quality_audit(df)
    return df, before


def _plot_quality_audit(df):
    """EDA overview: churn balance, monthly charges, tenure, contract type."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Otomoto Teleconnect - Pre-Cleansing Data Overview",
                 fontweight='bold', fontsize=13)

    # Churn distribution
    counts = [(df['Churn']=='No').sum(), (df['Churn']=='Yes').sum()]
    bars = axes[0,0].bar(['No Churn','Churn'], counts,
                         color=['#185FA5','#D85A30'], edgecolor='white', width=0.5)
    for bar, count in zip(bars, counts):
        axes[0,0].text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+30,
                       f"{count}\n({count/sum(counts)*100:.1f}%)",
                       ha='center', fontsize=10)
    axes[0,0].set_title("Class Distribution"); axes[0,0].grid(axis='y', alpha=0.3)

    # Monthly charges by churn
    axes[0,1].hist(df[df['Churn']=='No']['MonthlyCharges'],
                   bins=30, alpha=0.65, color='#185FA5', label='No Churn')
    axes[0,1].hist(df[df['Churn']=='Yes']['MonthlyCharges'],
                   bins=30, alpha=0.65, color='#D85A30', label='Churn')
    axes[0,1].set_title("Monthly Charges by Churn")
    axes[0,1].set_xlabel("Monthly Charges ($)")
    axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    # Tenure by churn
    axes[1,0].hist(df[df['Churn']=='No']['tenure'],
                   bins=30, alpha=0.65, color='#185FA5', label='No Churn')
    axes[1,0].hist(df[df['Churn']=='Yes']['tenure'],
                   bins=30, alpha=0.65, color='#D85A30', label='Churn')
    axes[1,0].set_title("Tenure (months) by Churn")
    axes[1,0].set_xlabel("Tenure (months)")
    axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    # Contract type by churn
    ct = df.groupby(['Contract','Churn']).size().unstack(fill_value=0)
    ct.plot(kind='bar', ax=axes[1,1], color=['#185FA5','#D85A30'],
            edgecolor='white', alpha=0.87)
    axes[1,1].set_title("Contract Type by Churn")
    axes[1,1].set_xlabel("Contract Type")
    axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=15)
    axes[1,1].legend(['No Churn','Churn']); axes[1,1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_data_overview.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   EDA overview      -> {path}")


# =============================================================================
# SECTION 2: PREPROCESSING PIPELINE
# Full cleansing with before/after quality gate.
# All decisions documented with explicit justification.
# =============================================================================
def preprocess(df: pd.DataFrame, before: dict):
    print("\n" + "="*65)
    print("  SECTION 2: PREPROCESSING PIPELINE")
    print("="*65)

    # Step 1: Drop non-informative identifier
    df = df.drop(columns=['customerID'])

    # Step 2: TotalCharges - whitespace -> NaN -> impute with median
    # Using df.assign() to avoid pandas 2.x Copy-on-Write silent failure
    df = df.assign(TotalCharges=pd.to_numeric(df['TotalCharges'], errors='coerce'))
    median_tc = df['TotalCharges'].median()
    n_missing = df['TotalCharges'].isnull().sum()
    df = df.assign(TotalCharges=df['TotalCharges'].fillna(median_tc))
    print(f"\n   TotalCharges NaN  : {n_missing} imputed with median (${median_tc:.2f})")

    # Step 3: Encode target - Churn Yes=1 (positive/minority), No=0
    df = df.assign(Churn=(df['Churn'] == 'Yes').astype(int))

    # Step 4: Binary Yes/No -> 1/0
    for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
        df = df.assign(**{col: (df[col] == 'Yes').astype(int)})
    df = df.assign(gender=(df['gender'] == 'Male').astype(int))

    # Step 5: One-hot encode multi-category columns (drop_first avoids dummy trap)
    ohe_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                'Contract','PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Step 6: Convert bool dtype -> int (prevents NaN after StandardScaler)
    df = df.astype({c: int for c in df.select_dtypes('bool').columns})

    # Step 7: IQR outlier audit - flag but retain (customer data clinically relevant)
    numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
    outlier_count = 0
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_count += ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
    print(f"   Outlier cells     : {outlier_count} in {numeric_cols} - retained "
          f"(extreme values are diagnostically informative for churn prediction)")

    # AFTER quality snapshot
    after = {
        'total_rows'    : len(df),
        'total_cols'    : len(df.columns),
        'null_count'    : df.isnull().sum().sum(),
        'feature_count' : df.shape[1] - 1,
        'churn_rate_pct': round(df['Churn'].mean()*100, 2),
    }
    print(f"\n   AFTER cleansing:")
    for k, v in after.items():
        print(f"      {k:35s}: {v}")

    # Data quality gate - assert zero nulls before training
    assert df.isnull().sum().sum() == 0, "QUALITY GATE FAILED: NaN in feature matrix"
    print(f"\n   [OK] Data quality gate passed - zero NaN in feature matrix")

    _save_quality_report(before, after)

    X = df.drop('Churn', axis=1).values.astype(np.float32)
    y = df['Churn'].values.astype(np.float32)
    feature_names = df.drop('Churn', axis=1).columns.tolist()

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # StandardScaler - fitted on train only, no leakage
    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X_train)
    X_test   = scaler.transform(X_test)

    print(f"\n   Train set         : {X_train.shape} (stratified 80%)")
    print(f"   Test set          : {X_test.shape}  (stratified 20%)")
    print(f"   Feature names     : {len(feature_names)} encoded features")
    print(f"   Scaling           : StandardScaler (fitted on train only)")

    return X_train, X_test, y_train, y_test, scaler, feature_names


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)

def _save_quality_report(before, after):
    report = {'before': before, 'after': after,
              'timestamp': datetime.now().isoformat()}
    path = os.path.join(OUTPUT_DIR, "data_quality_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, cls=NpEncoder)
    print(f"\n   Quality report    -> {path}")


# =============================================================================
# SECTION 3: BASELINE MODEL - LOGISTIC REGRESSION
# Required by rubric: "baselines included; justify any added complexity."
# Logistic regression is the appropriate baseline for binary classification:
#   - Interpretable coefficients map directly to business recommendations
#   - Low variance: stable on 7k-row tabular dataset
#   - Fast to train: provides performance floor in seconds
# =============================================================================
def train_baseline(X_train, X_test, y_train, y_test):
    print("\n" + "="*65)
    print("  SECTION 3: BASELINE - LOGISTIC REGRESSION")
    print("="*65)

    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    t0 = time.time()
    lr.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.50).astype(int)

    metrics = _compute_metrics("Logistic Regression (Baseline)",
                                y_test, y_pred_lr, y_prob_lr)
    metrics['time_s']  = round(elapsed, 2)
    metrics['epochs']  = 'N/A'
    metrics['val_loss'] = 'N/A'

    print(f"\n   Training time     : {elapsed:.2f}s")
    print(f"   Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"   Precision (Churn) : {metrics['precision']*100:.2f}%  (pos_label=1)")
    print(f"   Recall (Churn)    : {metrics['recall']*100:.2f}%   (pos_label=1)")
    print(f"   F1 (Churn)        : {metrics['f1']*100:.2f}%")
    print(f"   AUC-ROC           : {metrics['auc']*100:.2f}%")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred_lr,
                                target_names=['No Churn','Churn'],
                                zero_division=0))

    return lr, y_prob_lr, metrics


# =============================================================================
# SECTION 4: PRIMARY MODEL - ANN with RMSProp
# Architecture: 30 -> 64 -> 32 -> 16 -> 1 (funnel, progressive abstraction)
# RMSProp selected for dataset-specific reasons documented in header.
# =============================================================================
def build_ann(input_dim: int, optimiser) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name='input'),

        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Activation('relu', name='relu_1'),
        layers.Dropout(0.30, name='dropout_1'),

        layers.Dense(32, kernel_regularizer=regularizers.l2(0.001), name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Activation('relu', name='relu_2'),
        layers.Dropout(0.30, name='dropout_2'),

        layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), name='dense_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.Activation('relu', name='relu_3'),
        layers.Dropout(0.20, name='dropout_3'),

        layers.Dense(1, activation='sigmoid', name='output'),
    ], name='Otomoto_ANN_RMSProp')

    model.compile(
        optimizer = optimiser,
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
    )
    return model


def train_ann(X_train, X_test, y_train, y_test):
    print("\n" + "="*65)
    print("  SECTION 4: PRIMARY MODEL - ANN / RMSProp")
    print("="*65)

    # RMSProp: lr=0.001, rho=0.9
    # Dataset-specific justification in file header
    optimiser = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    model = build_ann(X_train.shape[1], optimiser)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0)
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs           = 150,
        batch_size       = 64,
        validation_split = 0.15,
        callbacks        = callbacks,
        verbose          = 0
    )
    elapsed = time.time() - t0
    epochs_run = len(history.history['loss'])

    print(f"\n   Epochs run        : {epochs_run}  (best weights restored)")
    print(f"   Training time     : {elapsed:.1f}s")

    y_prob_ann = model.predict(X_test, verbose=0).flatten()
    y_pred_ann = (y_prob_ann >= 0.50).astype(int)

    metrics = _compute_metrics("ANN / RMSProp", y_test, y_pred_ann, y_prob_ann)
    metrics['time_s']   = round(elapsed, 1)
    metrics['epochs']   = epochs_run
    metrics['val_loss'] = round(min(history.history['val_loss']), 4)

    print(f"   Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"   Precision (Churn) : {metrics['precision']*100:.2f}%  (pos_label=1)")
    print(f"   Recall (Churn)    : {metrics['recall']*100:.2f}%   (pos_label=1)")
    print(f"   F1 (Churn)        : {metrics['f1']*100:.2f}%")
    print(f"   AUC-ROC           : {metrics['auc']*100:.2f}%")
    print(f"   Best val_loss     : {metrics['val_loss']}")

    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred_ann,
                                target_names=['No Churn','Churn'],
                                zero_division=0))

    _plot_training_history(history)
    return model, y_prob_ann, metrics, history


def _plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ANN Training History - RMSProp Optimiser (Otomoto)",
                 fontweight='bold')
    ep = range(1, len(history.history['loss']) + 1)
    ax1.plot(ep, history.history['loss'],     color='#185FA5', lw=2, label='Train')
    ax1.plot(ep, history.history['val_loss'], color='#D85A30', lw=2,
             linestyle='--', label='Validation')
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, history.history['accuracy'],     color='#185FA5', lw=2, label='Train')
    ax2.plot(ep, history.history['val_accuracy'], color='#D85A30', lw=2,
             linestyle='--', label='Validation')
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Training history  -> {path}")


# =============================================================================
# SECTION 5: CROSS-VALIDATION
# 5-fold stratified CV on the ANN to estimate generalisation variance.
# Provides confidence-interval-equivalent evidence for the rubric.
# =============================================================================
def cross_validate_ann(X_train, y_train, input_dim: int):
    print("\n" + "="*65)
    print("  SECTION 5: CROSS-VALIDATION (5-FOLD STRATIFIED)")
    print("="*65)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        opt = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        m   = build_ann(input_dim, opt)
        m.fit(Xtr, ytr, epochs=80, batch_size=64, verbose=0,
              validation_data=(Xval, yval),
              callbacks=[EarlyStopping(patience=10,
                                       restore_best_weights=True)])
        yp   = m.predict(Xval, verbose=0).flatten()
        ypred = (yp >= 0.50).astype(int)

        fold_m = {
            'fold'     : fold,
            'accuracy' : round(accuracy_score(yval, ypred), 4),
            'precision': round(precision_score(yval, ypred, pos_label=1, zero_division=0), 4),
            'recall'   : round(recall_score(yval, ypred,    pos_label=1, zero_division=0), 4),
            'f1'       : round(f1_score(yval, ypred,        pos_label=1, zero_division=0), 4),
            'auc'      : round(roc_auc_score(yval, yp), 4),
        }
        cv_results.append(fold_m)
        print(f"   Fold {fold}: acc={fold_m['accuracy']:.4f}  "
              f"prec={fold_m['precision']:.4f}  rec={fold_m['recall']:.4f}  "
              f"f1={fold_m['f1']:.4f}  auc={fold_m['auc']:.4f}")

    print(f"\n   ---------------- Summary (mean +/- std) ----------------")
    cv_summary = {}
    for metric in ['accuracy','precision','recall','f1','auc']:
        vals = [r[metric] for r in cv_results]
        mean, std = np.mean(vals), np.std(vals)
        cv_summary[metric] = {'mean': round(mean,4), 'std': round(std,4)}
        print(f"   {metric:10s}: {mean*100:.2f}% +/- {std*100:.2f}% "
          f"(95% CI ~ {(mean-1.96*std)*100:.2f}% to {(mean+1.96*std)*100:.2f}%)")

    _plot_cv_results(cv_results)
    return cv_results, cv_summary


def _plot_cv_results(cv_results):
    metrics = ['accuracy','precision','recall','f1','auc']
    colours = ['#185FA5','#1D9E75','#D85A30','#BA7517','#639922']
    folds   = [r['fold'] for r in cv_results]
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, c in zip(metrics, colours):
        vals = [r[m] for r in cv_results]
        ax.plot(folds, vals, marker='o', color=c, lw=2, label=m.capitalize())
    ax.set_xlabel("Fold"); ax.set_ylabel("Score")
    ax.set_ylim(0.4, 1.0); ax.set_xticks(folds)
    ax.set_title("5-Fold Cross-Validation - ANN/RMSProp (Otomoto)",
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_cross_validation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   CV plot           -> {path}")


# =============================================================================
# SECTION 6: THRESHOLD ANALYSIS & DECISION IMPACT
# Sweeps thresholds 0.20–0.70.
# Marketing context: recall is the priority metric.
#   False negative = churner not contacted = full lifetime revenue loss.
#   False positive = loyal customer receives unnecessary retention offer.
#   Asymmetric cost structure justifies threshold below 0.50.
# =============================================================================
def threshold_analysis(y_test, y_prob_lr, y_prob_ann):
    print("\n" + "="*65)
    print("  SECTION 6: THRESHOLD ANALYSIS & DECISION IMPACT")
    print("="*65)

    thresholds = np.arange(0.20, 0.71, 0.05)
    results_ann = []
    results_lr  = []

    for t in thresholds:
        for y_prob, store, name in [
            (y_prob_ann, results_ann, "ANN"),
            (y_prob_lr,  results_lr,  "LR")
        ]:
            ypred = (y_prob >= t).astype(int)
            store.append({
                'threshold' : round(float(t), 2),
                'precision' : round(precision_score(y_test, ypred,
                                    pos_label=1, zero_division=0), 4),
                'recall'    : round(recall_score(y_test, ypred,
                                    pos_label=1, zero_division=0), 4),
                'f1'        : round(f1_score(y_test, ypred,
                                    pos_label=1, zero_division=0), 4),
                'accuracy'  : round(accuracy_score(y_test, ypred), 4),
            })

    # Best threshold by F1 for ANN
    best_t_row = max(results_ann, key=lambda r: r['f1'])
    print(f"\n   ANN best threshold (max F1) : {best_t_row['threshold']}")
    print(f"   At best threshold:")
    print(f"      Precision : {best_t_row['precision']*100:.2f}%")
    print(f"      Recall    : {best_t_row['recall']*100:.2f}%")
    print(f"      F1        : {best_t_row['f1']*100:.2f}%")

    _plot_threshold_analysis(thresholds, results_ann, results_lr, best_t_row)
    return results_ann, results_lr, best_t_row


def _plot_threshold_analysis(thresholds, results_ann, results_lr, best_row):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Threshold Analysis - Precision / Recall Trade-off (Otomoto)",
                 fontweight='bold')

    for ax, results, title in [(ax1, results_ann, "ANN / RMSProp"),
                                (ax2, results_lr,  "Logistic Regression (Baseline)")]:
        ts = [r['threshold'] for r in results]
        ax.plot(ts, [r['precision'] for r in results], 'o-',
                color='#185FA5', lw=2, label='Precision')
        ax.plot(ts, [r['recall'] for r in results], 's-',
                color='#D85A30', lw=2, label='Recall (priority)')
        ax.plot(ts, [r['f1'] for r in results], '^-',
                color='#1D9E75', lw=2, label='F1')
        if title.startswith("ANN"):
            ax.axvline(best_row['threshold'], color='grey',
                       linestyle=':', lw=1.5, label=f"Best F1 t={best_row['threshold']}")
        ax.axvline(0.50, color='black', linestyle='--', lw=1, alpha=0.5,
                   label='Default t=0.50')
        ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_threshold_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Threshold plot    -> {path}")


# =============================================================================
# SECTION 7: FEATURE IMPORTANCE & ROC COMPARISON
# Feature importance from Logistic Regression coefficients (interpretable).
# ROC curves for both models overlaid.
# =============================================================================
def feature_importance_and_roc(lr_model, y_test, y_prob_lr,
                                 y_prob_ann, feature_names):
    print("\n" + "="*65)
    print("  SECTION 7: FEATURE IMPORTANCE & ROC COMPARISON")
    print("="*65)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Importance & ROC Curves - Otomoto Churn Model",
                 fontweight='bold')

    # Feature importance from LR coefficients
    coefs = lr_model.coef_[0]
    top_n = 15
    top_idx  = np.argsort(np.abs(coefs))[-top_n:]
    top_names = [feature_names[i] for i in top_idx]
    top_coefs = coefs[top_idx]
    colours_fi = ['#D85A30' if c > 0 else '#185FA5' for c in top_coefs]

    ax1.barh(range(top_n), top_coefs, color=colours_fi, edgecolor='white', alpha=0.87)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_names, fontsize=9)
    ax1.axvline(0, color='black', lw=0.8)
    ax1.set_xlabel("Logistic Regression Coefficient")
    ax1.set_title(f"Top {top_n} Feature Importances\n"
                  f"(orange = churn risk ↑, blue = churn risk ↓)")
    ax1.grid(axis='x', alpha=0.3)

    # Print top churner risk factors
    churn_risk = [(feature_names[i], coefs[i])
                  for i in np.argsort(coefs)[-5:]][::-1]
    print(f"\n   Top 5 churn risk features (positive LR coefficient):")
    for name, coef in churn_risk:
        print(f"      {name:45s}: {coef:+.4f}")

    no_churn_risk = [(feature_names[i], coefs[i])
                     for i in np.argsort(coefs)[:5]]
    print(f"\n   Top 5 retention features (negative LR coefficient):")
    for name, coef in no_churn_risk:
        print(f"      {name:45s}: {coef:+.4f}")

    # ROC comparison
    for y_prob, label, colour in [
        (y_prob_lr,  f"Logistic Regression (AUC={roc_auc_score(y_test,y_prob_lr):.3f})",
         '#185FA5'),
        (y_prob_ann, f"ANN / RMSProp (AUC={roc_auc_score(y_test,y_prob_ann):.3f})",
         '#D85A30'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax2.plot(fpr, tpr, color=colour, lw=2, label=label)

    ax2.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
    ax2.fill_between([0,1],[0,1], alpha=0.04, color='grey')
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves - Baseline vs ANN")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_feature_importance_roc.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Feature importance + ROC -> {path}")


# =============================================================================
# SECTION 8: CALIBRATION, COMPARISON TABLE, MODEL SAVE
# =============================================================================
def calibration_and_save(model, lr_model, y_test, y_prob_lr,
                          y_prob_ann, metrics_lr, metrics_ann,
                          cv_summary, best_threshold_row):
    print("\n" + "="*65)
    print("  SECTION 8: CALIBRATION & MODEL SAVE")
    print("="*65)

    # Calibration plot
    fig, ax = plt.subplots(figsize=(7, 6))
    for y_prob, label, colour in [
        (y_prob_lr,  'Logistic Regression', '#185FA5'),
        (y_prob_ann, 'ANN / RMSProp',       '#D85A30'),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker='o', color=colour, lw=2, label=label)
        bs = brier_score_loss(y_test, y_prob)
        print(f"   Brier score ({label:22s}): {bs:.4f}  (lower = better calibration)")

    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Perfect calibration')
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Churn Rate)")
    ax.set_title("Calibration Curve - Predicted Probability vs Actual Churn Rate",
                 fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_calibration.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Calibration plot  -> {path}")

    # Confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Confusion Matrices - Baseline vs ANN (pos_label=1 = Churn=Yes)",
                 fontweight='bold')
    for ax, y_prob, title in [
        (ax1, y_prob_lr,  "Logistic Regression (Baseline)"),
        (ax2, y_prob_ann, "ANN / RMSProp (Primary)"),
    ]:
        cm = confusion_matrix(y_test, (y_prob >= 0.50).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred No (0)','Pred Churn (1)'],
                    yticklabels=['True No (0)','True Churn (1)'],
                    annot_kws={'size': 13})
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrices -> {path}")

    # Full comparison table plot
    _plot_comparison_table(metrics_lr, metrics_ann, cv_summary)

    # Save full metrics JSON
    all_metrics = {
        'baseline_lr'     : {k: v for k, v in metrics_lr.items()
                              if k != 'y_prob'},
        'ann_rmsprop'     : {k: v for k, v in metrics_ann.items()
                              if k != 'y_prob'},
        'cv_summary'      : cv_summary,
        'best_threshold'  : best_threshold_row,
        'run_timestamp'   : datetime.now().isoformat(),
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_full.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, cls=NpEncoder)
    print(f"   Metrics JSON      -> {metrics_path}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "otomoto_churn_model.keras")
    model.save(model_path)
    print(f"   Model saved       -> {model_path}")

    # Save comparison CSV
    rows = [
        {'Model': 'Logistic Regression (Baseline)',
         **{k: v for k, v in metrics_lr.items() if k not in ('name','y_prob')}},
        {'Model': 'ANN / RMSProp (Primary)',
         **{k: v for k, v in metrics_ann.items() if k not in ('name','y_prob')}},
    ]
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"   Comparison CSV    -> {csv_path}")


def _plot_comparison_table(metrics_lr, metrics_ann, cv_summary):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')

    col_labels = ['Metric', 'LR Baseline', 'ANN / RMSProp', 'ANN CV Mean +/- SD']
    rows_data = [
        ['Accuracy',
         f"{metrics_lr['accuracy']*100:.2f}%",
         f"{metrics_ann['accuracy']*100:.2f}%",
         f"{cv_summary['accuracy']['mean']*100:.2f}% +/- {cv_summary['accuracy']['std']*100:.2f}%"],
        ['Precision (Churn)',
         f"{metrics_lr['precision']*100:.2f}%",
         f"{metrics_ann['precision']*100:.2f}%",
         f"{cv_summary['precision']['mean']*100:.2f}% +/- {cv_summary['precision']['std']*100:.2f}%"],
        ['Recall (Churn) ★',
         f"{metrics_lr['recall']*100:.2f}%",
         f"{metrics_ann['recall']*100:.2f}%",
         f"{cv_summary['recall']['mean']*100:.2f}% +/- {cv_summary['recall']['std']*100:.2f}%"],
        ['F1-Score (Churn)',
         f"{metrics_lr['f1']*100:.2f}%",
         f"{metrics_ann['f1']*100:.2f}%",
         f"{cv_summary['f1']['mean']*100:.2f}% +/- {cv_summary['f1']['std']*100:.2f}%"],
        ['AUC-ROC',
         f"{metrics_lr['auc']*100:.2f}%",
         f"{metrics_ann['auc']*100:.2f}%",
         f"{cv_summary['auc']['mean']*100:.2f}% +/- {cv_summary['auc']['std']*100:.2f}%"],
        ['Training time',
         f"{metrics_lr['time_s']}s",
         f"{metrics_ann['time_s']}s",
         '-'],
    ]

    tbl = ax.table(cellText=rows_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1F4E79')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight recall row (row 3, index 2 in data = row index 3 in table)
    for j in range(len(col_labels)):
        tbl[3, j].set_facecolor('#FFF2CC')

    ax.set_title("Model Comparison - Baseline vs Primary (★ = priority marketing metric)",
                 fontweight='bold', pad=20, fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_model_comparison_table.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Comparison table  -> {path}")


# =============================================================================
# HELPER: compute metrics dict
# =============================================================================
def _compute_metrics(name, y_test, y_pred, y_prob):
    return {
        'name'      : name,
        'accuracy'  : round(accuracy_score(y_test, y_pred), 4),
        'precision' : round(precision_score(y_test, y_pred,
                            pos_label=1, zero_division=0), 4),
        'recall'    : round(recall_score(y_test, y_pred,
                            pos_label=1, zero_division=0), 4),
        'f1'        : round(f1_score(y_test, y_pred,
                            pos_label=1, zero_division=0), 4),
        'auc'       : round(roc_auc_score(y_test, y_prob), 4),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  BAN6440 Final Project | Otomoto Customer Churn Pipeline")
    print("  Baseline: Logistic Regression | Primary: ANN / RMSProp")
    print("  Dataset : teleconnect.csv (7,043 customers)")
    print("=" * 65)

    t_total = time.time()

    # 1. Load & audit
    df, before = load_and_audit('teleconnect.csv')

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df, before)

    # 3. Baseline
    lr_model, y_prob_lr, metrics_lr = train_baseline(
        X_train, X_test, y_train, y_test)

    # 4. ANN
    ann_model, y_prob_ann, metrics_ann, history = train_ann(
        X_train, X_test, y_train, y_test)

    # 5. Cross-validation
    cv_results, cv_summary = cross_validate_ann(
        X_train, y_train, X_train.shape[1])

    # 6. Threshold analysis
    results_ann_thresh, results_lr_thresh, best_t = threshold_analysis(
        y_test, y_prob_lr, y_prob_ann)

    # 7. Feature importance + ROC
    feature_importance_and_roc(
        lr_model, y_test, y_prob_lr, y_prob_ann, feature_names)

    # 8. Calibration + save
    calibration_and_save(
        ann_model, lr_model, y_test, y_prob_lr, y_prob_ann,
        metrics_lr, metrics_ann, cv_summary, best_t)

    total_time = time.time() - t_total
    print(f"\n   Total pipeline time : {total_time:.0f}s")
    print(f"   All outputs in      : {OUTPUT_DIR}/")

    print("\n" + "=" * 65)
    print("  [OK] Final pipeline completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
