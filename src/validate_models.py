import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def compute_task_auc(y_true, y_score):

    known_mask = y_true != 0

    y_true_known = y_true[known_mask]
    y_score_known = y_score[known_mask]

    y_true_known = (y_true_known == 1).astype(int)

    if len(np.unique(y_true_known)) < 2:
        return None

    return roc_auc_score(y_true_known, y_score_known)


def get_models():
    return [
        ExtraTreesClassifier(
            n_estimators=1200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2
        ),

        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            C=0.5,
            random_state=42
        ),

        HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42
        )
    ]


def prepare_task_data(X, train_data, task):

    labels = train_data[task].values
    known_samples = labels != 0

    X_task = X[known_samples]

    y_task = labels[known_samples]
    y_task = (y_task == 1).astype(int)

    return X_task, y_task


if __name__ == "__main__":

    train_data = pd.read_csv("data_train.csv")
    X = np.load("X_train.npy")

    task_columns = [f"task{i}" for i in range(1, 12)]

    # weights = [0.45, 0.35, 0.20]
    weights = [0.60, 0.25, 0.15]

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    task_aucs = []

    for task in task_columns:

        print(f"\nEvaluating {task}...")

        oof_predictions = np.zeros(len(train_data))

        for fold, (train_index, val_index) in enumerate(kfold.split(X), start=1):

            print(f"  Fold {fold}/5")

            X_train = X[train_index]
            X_val = X[val_index]

            train_fold = train_data.iloc[train_index].reset_index(drop=True)

            X_task, y_task = prepare_task_data(
                X_train,
                train_fold,
                task
            )

            fold_prediction = np.zeros(X_val.shape[0])

            for model, weight in zip(get_models(), weights):

                model.fit(X_task, y_task)

                probabilities = model.predict_proba(X_val)[:, 1]

                fold_prediction += weight * probabilities

            oof_predictions[val_index] = fold_prediction

        auc = compute_task_auc(
            train_data[task].values,
            oof_predictions
        )

        task_aucs.append(auc)

        print(f"{task}: AUC = {auc:.4f}")

    print(f"\nMean CV ensemble AUC: {np.mean(task_aucs):.4f}")
