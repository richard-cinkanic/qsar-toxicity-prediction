# QSAR Toxicity Prediction

Prediction of molecular activity and toxicity from chemical structure using machine learning and molecular fingerprints.

## Overview

This project was developed as part of the *Artificial Intelligence in Life Sciences* course. The objective was to predict molecular activity across 11 biological tasks using molecular SMILES representations and machine learning models.

The challenge involved partially missing labels, class imbalance, and varying task difficulty. Performance was evaluated using the Area Under the ROC Curve (AUC) on a hidden test set.

The workflow consists of:

1. Dataset exploration
2. Molecular feature generation
3. Fingerprint extraction using RDKit
4. Model training
5. Ensemble construction
6. Cross-validation
7. Submission generation and evaluation

## Requirements

* Python 3.11
* RDKit
* scikit-learn
* pandas
* NumPy

## Dataset

The dataset contains molecular SMILES strings together with activity labels for 11 prediction tasks.

Label encoding:

* +1 = Active molecule
* -1 = Inactive molecule
* 0 = Unknown / missing label

Key challenges:

* Missing labels across tasks
* Class imbalance
* Different task difficulty levels
* Multi-task prediction problem

## Feature Engineering

Molecules were converted into numerical feature vectors using RDKit.

Feature representation:

* Morgan fingerprints (radius 2, 2048 bits)
* MACCS structural keys (167 bits)

These feature sets were concatenated into a single 2215-dimensional molecular representation.

Feature extraction pipeline:

```text
SMILES Strings
      ↓
RDKit Molecule Parsing
      ↓
Morgan Fingerprints (2048)
      +
MACCS Keys (167)
      ↓
2215-Dimensional Feature Vector
```

## Model Architecture

The final solution uses a weighted ensemble of three machine learning models.

```text
ExtraTreesClassifier          60%
LogisticRegression            25%
HistGradientBoosting          15%
                ↓
       Weighted Averaging
                ↓
      Final Predictions
```

### ExtraTreesClassifier

* 1200 trees
* Balanced class weights
* Nonlinear decision boundaries
* Best individual validation performance

### Logistic Regression

* Balanced class weights
* Strong linear baseline
* Helps reduce overfitting

### HistGradientBoosting

* Gradient-boosted decision trees
* Captures complex feature interactions
* Efficient on high-dimensional data

## Training Configuration

* 5-fold cross-validation
* Ensemble prediction averaging
* Hyperparameter tuning through validation experiments
* Task-specific handling of missing labels

Cross-validation was used to improve robustness and reduce overfitting while providing more reliable performance estimates.

## Technologies

* Python
* RDKit
* scikit-learn
* pandas
* NumPy

## Skills Demonstrated

* Machine Learning
* Ensemble Learning
* QSAR Modeling
* Molecular Feature Engineering
* Cross-Validation
* Model Evaluation
* Data Preprocessing
* Scientific Computing

## Results

Final public leaderboard score:

* Public leaderboard AUC: 0.757

Additional project statistics:

* 11 prediction tasks
* 2215 molecular features per molecule
* 5-fold cross-validation
* Weighted ensemble model

The final solution combined molecular fingerprints with ensemble learning techniques to achieve strong predictive performance on unseen molecules.

## Repository Structure

```text
src/
├── explore_data.py
├── generate_features.py
├── train_models.py
├── validate_models.py
├── validate_cv_extratrees.py
└── evaluate_submission_auc.py
```

## How to Run

### 1. Generate Molecular Features

```bash
python src/generate_features.py
```

This creates:

```text
X_train.npy
X_test.npy
```

using Morgan fingerprints and MACCS keys.

### 2. Train Models and Generate Predictions

```bash
python src/train_models.py
```

This trains the ensemble model and generates:

```text
final_submission.csv
```

### 3. Validate Models

```bash
python src/validate_models.py
```

Evaluates the ensemble using 5-fold cross-validation.

### 4. Evaluate ExtraTrees Separately

```bash
python src/validate_cv_extratrees.py
```

Computes cross-validation performance for the ExtraTrees model.

### 5. Evaluate Submission AUC

```bash
python src/evaluate_submission_auc.py --submission submission.csv --target target.csv
```

Computes the average AUC across all prediction tasks while handling missing labels.

## Notes

The original dataset, generated feature matrices, and competition submissions are not included in this repository.

This repository focuses on the implementation of molecular feature engineering, ensemble learning, and model evaluation methods used in the challenge.

Final solution:
- Morgan fingerprints (radius 2, 2048 bits)
- MACCS keys (167 bits)
- Weighted 3-model ensemble
- 5-fold cross-validation
