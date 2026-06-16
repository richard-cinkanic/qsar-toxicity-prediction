import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier


def compute_task_auc(y_true, y_score):
    known_mask = y_true != 0

    y_true_known = y_true[known_mask]
    y_score_known = y_score[known_mask]

    y_true_known = (y_true_known == 1).astype(int)

    if len(np.unique(y_true_known)) < 2:
        return None

    return roc_auc_score(y_true_known, y_score_known)


def prepare_task_data(X, train_data, task):
    labels = train_data[task].values
    known_samples = labels != 0

    X_task = X[known_samples]
    y_task = (labels[known_samples] == 1).astype(int)

    return X_task, y_task


if __name__ == "__main__":

    train_data = pd.read_csv("data_train.csv")
    X = np.load("X_train.npy")

    task_columns = [f"task{i}" for i in range(1, 12)]

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

            model = ExtraTreesClassifier(
                n_estimators=1200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                max_features="sqrt"
            )

            model.fit(X_task, y_task)

            oof_predictions[val_index] = model.predict_proba(X_val)[:, 1]

        auc = compute_task_auc(
            train_data[task].values,
            oof_predictions
        )

        task_aucs.append(auc)

        print(f"{task}: AUC = {auc:.4f}")

    print(f"\nMean CV ExtraTrees AUC: {np.mean(task_aucs):.4f}")
