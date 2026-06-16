import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


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


def predict_task_cv(X, X_test, train_data, task):
    # ExtraTrees, LogisticRegression, HistGradientBoosting
    weights = [0.60, 0.25, 0.15]

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    test_prediction = np.zeros(X_test.shape[0])

    for fold, (train_index, _) in enumerate(kfold.split(X), start=1):

        print(f"  Fold {fold}/5")

        X_fold = X[train_index]
        train_fold = train_data.iloc[train_index].reset_index(drop=True)

        X_task, y_task = prepare_task_data(
            X_fold,
            train_fold,
            task
        )

        fold_prediction = np.zeros(X_test.shape[0])

        for model, weight in zip(get_models(), weights):

            model.fit(X_task, y_task)

            probabilities = model.predict_proba(X_test)[:, 1]

            fold_prediction += weight * probabilities

        test_prediction += fold_prediction / 5

    return test_prediction


if __name__ == "__main__":

    train_data = pd.read_csv("data_train.csv")
    sample_submission = pd.read_csv("sample_submission.csv")

    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")

    task_columns = [f"task{i}" for i in range(1, 12)]

    predictions_df = sample_submission.copy()

    for task in task_columns:

        print(f"\nTraining {task}...")

        predictions_df[task] = predict_task_cv(
            X_train,
            X_test,
            train_data,
            task
        )

    predictions_df.to_csv(
        "final_submission.csv",
        index=False
    )

    print("\nSaved final_submission.csv")
