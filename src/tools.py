"""Three tool wrappers around thesis evaluation functions."""

from typing import TypedDict

import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class FidelityResult(TypedDict):
    """Per-column fidelity scores plus a summary, as returned by the fidelity tools.

    ``per_column`` maps each column name to its score, or ``None`` when the score
    is undefined (e.g. a zero pooled standard deviation, or a NaN distance).
    ``mean`` is the average over the defined scores, or ``None`` when none are defined.
    """

    per_column: dict[str, float | None]
    mean: float | None
    n_columns: int


class UtilityResult(TypedDict):
    """Train-on-synthetic vs train-on-real utility benchmark, returned by ``tstr_xgboost``."""

    tstr_accuracy: float
    trtr_accuracy: float
    utility_gap: float
    target_column: str
    n_test_samples: int
    n_classes: int


def js_divergence_categorical(real_csv_path: str, synth_csv_path: str) -> FidelityResult:
    """
    Compute Jensen-Shannon distance for each categorical column.
    
    Args:
        real_csv_path: path to the real dataset CSV
        synth_csv_path: path to the synthetic dataset CSV
    
    Returns:
        {
            "per_column": {column_name: js_distance, ...},
            "mean": float,
            "n_columns": int
        }
    """

    real_data = pd.read_csv(real_csv_path)
    synthetic_data = pd.read_csv(synth_csv_path)

    categorical_features = real_data.select_dtypes(include=['object', 'category', 'string']).columns

    per_column: dict[str, float | None] = {}

    for feature in categorical_features :
        categories = real_data[feature].unique()
        n_real = real_data[feature].shape[0]
        n_synthetic = synthetic_data[feature].shape[0]

        real_count = (
            real_data[feature]
            .value_counts()
            .reindex(categories, fill_value=1e-7)
        )

        synthetic_count = (
            synthetic_data[feature]
            .value_counts()
            .reindex(categories, fill_value=1e-7)
        )


        real_prob = np.array(real_count/n_real)
        synthetic_prob = np.array(synthetic_count/n_synthetic)
        js_distance = float(jensenshannon(real_prob, synthetic_prob, base=2))
        per_column[feature] = None if np.isnan(js_distance) else js_distance

    return _build_results(per_column)

def cohen_d_numerical(real_csv_path: str, synth_csv_path: str) -> FidelityResult:
    """
    Compute absolute Cohen's d effect size for each numerical column.

    Args:
        real_csv_path: path to the real dataset CSV
        synth_csv_path: path to the synthetic dataset CSV

    Returns:
        {
            "per_column": {column_name: cohen_d, ...},
            "mean": float,
            "n_columns": int
        }
    """

    real_data = pd.read_csv(real_csv_path)
    synthetic_data = pd.read_csv(synth_csv_path)

    numerical_features = real_data.select_dtypes(exclude=['object', 'category', 'string']).columns

    per_column: dict[str, float | None] = {}

    for feature in numerical_features :
        group1 = real_data[feature].to_numpy()
        group2 = synthetic_data[feature].to_numpy()

        #calculate mean and variances
        means1 = np.mean(group1)
        means2 = np.mean(group2)
        variance1 = np.var(group1, ddof=1)
        variance2 = np.var(group2, ddof=1)

        # Calculate pooledstan
        n1 = len(group1)
        n2 = len(group2)
        pooled_std = np.sqrt(((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2))

        #cohen s d
        cohen_d = float(abs( (means1 - means2) / pooled_std )) if pooled_std != 0 else None
        per_column[feature] = cohen_d

    return _build_results(per_column)

def tstr_xgboost(
    real_csv_path: str,
    synth_csv_path: str,
    target_column: str,
    test_csv_path: str | None = None,
) -> UtilityResult:
    """
    Train-on-synthetic, test-on-real (TSTR) utility benchmark using XGBoost.
    Compares against train-on-real, test-on-real (TRTR) baseline.

    Args:
        real_csv_path: real training dataset CSV
        synth_csv_path: synthetic dataset CSV  
        target_column: name of the column to predict (classification)
        test_csv_path: optional held-out test set CSV. If None, splits real_csv_path 80/20.

    Returns:
        {
            "tstr_accuracy": float,
            "trtr_accuracy": float,
            "utility_gap": float,    # trtr - tstr (positive = synthetic loses utility)
            "target_column": str,
            "n_test_samples": int,
            "n_classes": int,
        }
    """

    real_data = pd.read_csv(real_csv_path)
    synthetic_data = pd.read_csv(synth_csv_path)


    if test_csv_path is None :
        real_train, real_test = train_test_split(real_data, test_size=0.2,random_state=42)
    else :
        real_train, real_test = real_data, pd.read_csv(test_csv_path)
    
    feature_columns = [feature for feature in real_data.columns if feature != target_column]  


    real_train_features, real_train_target = real_train[feature_columns], real_train[target_column]
    synth_features, synth_target = synthetic_data[feature_columns], synthetic_data[target_column]
    real_test_features, real_test_target = real_test[feature_columns], real_test[target_column]

    #Preprocess one hot encoding

    n_train = real_train_features.shape[0]
    n_test = real_test_features.shape[0]

    combined_frame = pd.concat([real_train_features, real_test_features, synth_features], axis=0, ignore_index=True)
    
    combined_frame_encoded = pd.get_dummies(combined_frame)

    real_train_encoded = combined_frame_encoded.iloc[:n_train].copy()
    real_test_encoded = combined_frame_encoded.iloc[n_train:n_train+n_test].copy()
    synth_encoded = combined_frame_encoded.iloc[n_train+n_test:].copy()

    # Encode the target

    le = LabelEncoder()
    le.fit(pd.concat([real_train_target, real_test_target, synth_target], ignore_index=True))

    target_train_enc = le.transform(real_train_target)
    target_test_enc = le.transform(real_test_target)
    target_synth_enc = le.transform(synth_target)

    # training and prediction
    trtr_classifier = XGBClassifier()
    tstr_classifier = XGBClassifier()

    trtr_classifier.fit(real_train_encoded, target_train_enc)
    trtr_predicted = trtr_classifier.predict(real_test_encoded)

    tstr_classifier.fit(synth_encoded, target_synth_enc)
    tstr_predicted = tstr_classifier.predict(real_test_encoded)

    # score
    trtr_accuracy = accuracy_score(trtr_predicted, target_test_enc)
    tstr_accuracy = accuracy_score(tstr_predicted, target_test_enc)

    return {
        "tstr_accuracy": float(tstr_accuracy),
        "trtr_accuracy": float(trtr_accuracy),
        "utility_gap": float(trtr_accuracy - tstr_accuracy),    # trtr - tstr (positive = synthetic loses utility)
        "target_column": target_column,
        "n_test_samples": n_test,
        "n_classes": int(real_train[target_column].nunique()),
        }

def _build_results(per_column: dict[str, float | None]) -> FidelityResult:

    valid_values: list[float] = [value for value in per_column.values() if value is not None]
    mean: float | None = float(np.mean(valid_values)) if valid_values else None

    return {
        "per_column": per_column,
        "mean": mean,
        "n_columns": len(per_column)
        }



if __name__ == "__main__" :
    js_result = js_divergence_categorical(
        "examples/real_sample.csv",
        "examples/synth_sample.csv",
    )

    print("JS divergence:", js_result)

    cohen_result = cohen_d_numerical(
        "examples/real_sample.csv",
        "examples/synth_sample.csv",
    )
    print("Cohen's d:   ", cohen_result)

    tstr_result = tstr_xgboost(
        "examples/real_sample.csv",
        "examples/synth_sample.csv",
        target_column="target"
    )
    print("tstr :   ", tstr_result)

