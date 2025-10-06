import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference, performance_on_categorical_slice, compute_model_metrics

def test_one():
    """
    # process_data returns NumPy arrays (types) and an 80/20 split gives expected sizes.

    """
    df = pd.DataFrame(
        [
            [39, 77516, 13, 0, 0, 40, "Private", "Bachelors", "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", "United-States", "<=50K"],
            [50, 83311, 13, 0, 0, 13, "Self-emp-not-inc", "Bachelors", "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", "United-States", ">50K"],
            [38, 215646, 9,  0, 0, 40, "Private", "HS-grad", "Divorced", "Handlers-cleaners", "Not-in-family", "White", "Male", "United-States", "<=50K"],
            [28, 338409, 13, 0, 0, 40, "Private", "Bachelors", "Married-civ-spouse", "Prof-specialty", "Wife", "Black", "Female", "United-States", ">50K"],
            [37, 284582, 14, 0, 0, 60, "Local-gov", "Masters", "Never-married", "Prof-specialty", "Not-in-family", "White", "Female", "United-States", ">50K"],
        ],
        columns=[
            "age","fnlgt","education_num","capital_gain","capital_loss","hours_per_week",
            "workclass","education","marital-status","occupation","relationship","race","sex",
            "native-country","salary",
        ],
    )
    cats = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["salary"])
    assert isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame)
    assert len(train_df) == 4 and len(test_df) == 1  # 80/20 of 5 rows

    X, y, enc, lb = process_data(train_df, categorical_features=cats, label="salary", training=True)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0] == len(train_df)
    assert enc is not None and lb is not None
    pass


def test_two():
    """
    # train_model returns RandomForestClassifier and inference gives a binary prediction.

    """
    df = pd.DataFrame(
        [
            [39, 77516, 13, 0, 0, 40, "Private", "Bachelors", "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", "United-States", "<=50K"],
            [50, 83311, 13, 0, 0, 13, "Self-emp-not-inc", "Bachelors", "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", "United-States", ">50K"],
            [38, 215646, 9,  0, 0, 40, "Private", "HS-grad", "Divorced", "Handlers-cleaners", "Not-in-family", "White", "Male", "United-States", "<=50K"],
        ],
        columns=[
            "age","fnlgt","education_num","capital_gain","capital_loss","hours_per_week",
            "workclass","education","marital-status","occupation","relationship","race","sex",
            "native-country","salary",
        ],
    )
    cats = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

    Xtr, ytr, enc, lb = process_data(df, categorical_features=cats, label="salary", training=True)
    model = train_model(Xtr, ytr)
    assert isinstance(model, RandomForestClassifier)

    Xte, yte, _, _ = process_data(df.iloc[[0, 2]], categorical_features=cats, label="salary",
                                  training=False, encoder=enc, lb=lb)
    preds = inference(model, Xte)
    assert preds.shape[0] == Xte.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})
    pass


def test_three():
    """
    # compute_model_metrics returns expected precision/recall/F1 on a known case.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])  # TP=1, FP=0, FN=1  -> P=1.0, R=0.5, F1=0.6667
    p, r, f1 = compute_model_metrics(y_true, y_pred)
    assert pytest.approx(p, rel=1e-6) == 1.0
    assert pytest.approx(r, rel=1e-6) == 0.5
    assert pytest.approx(f1, rel=1e-6) == 2 * p * r / (p + r)
    pass