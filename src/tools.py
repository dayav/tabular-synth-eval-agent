"""Four tool wrappers around thesis evaluation functions: three fidelity/utility metrics and one privacy metric."""

from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
        # Reindex on the union of levels: a level that exists only in the synthetic
        # file is a real distributional difference and must not be dropped.
        categories = pd.unique(
            pd.concat([real_data[feature], synthetic_data[feature]], ignore_index=True).dropna()
        )
        n_real = real_data[feature].shape[0]
        n_synthetic = synthetic_data[feature].shape[0]

        real_count = (
            real_data[feature]
            .value_counts()
            .reindex(categories, fill_value=0.0)
        )

        synthetic_count = (
            synthetic_data[feature]
            .value_counts()
            .reindex(categories, fill_value=0.0)
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

class PrivacyResult(TypedDict):
    """HEOM nearest-neighbour privacy metrics, returned by ``identifiability_heom``.

    ``identifiability`` is the fraction of real records whose nearest synthetic record is
    strictly closer than their nearest other real record (the identifiability score of
    Yoon et al. 2020, computed in HEOM space). For an honest sample of the same size as
    the real data it sits near 0.5 by symmetry; a verbatim copy scores 1.0. The synthetic
    set is subsampled (seeded) to the real size before this is computed so that the
    0.5 reference holds whatever the size ratio.

    ``epsilon_any`` is the thesis filter statistic: the fraction of synthetic rows that
    fall strictly inside at least one real record's k-nearest-neighbour ball. Lower is
    better, but its level for an honest sample is high and data-dependent (about 0.65 on
    the golden cases), so it is reported as supporting evidence, not thresholded.

    ``exact_match_fraction`` is the fraction of synthetic rows at HEOM distance zero from
    some real row, i.e. verbatim copies.
    """

    identifiability: float
    n_real_identified: int
    n_synthetic_used_for_identifiability: int
    epsilon_any: float
    n_unsafe_synthetic: int
    n_synthetic: int
    n_real: int
    k: int
    exact_matches: int
    exact_match_fraction: float
    median_synth_to_real_distance: float
    median_real_knn_radius: float
    numerical_columns: list[str]
    categorical_columns: list[str]
    n_rows_dropped_for_missing_values: int


def _heom_encode(
    real: pd.DataFrame, synth: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorise both frames so that Euclidean distance equals the HEOM distance.

    Numeric columns are min-max scaled on the real data (a difference of one full real
    range counts 1). Each categorical column becomes a one-hot block scaled by 1/sqrt(2),
    so two rows with different categories are at distance exactly 1 on that column and
    rows with the same category at distance 0. A category unseen in the real data maps to
    an all-zero block. This is the encoder of the thesis pipeline (privacy_sampling/heom.py).
    """
    scale = 1.0 / np.sqrt(2.0)
    blocks_real: list[np.ndarray] = []
    blocks_synth: list[np.ndarray] = []
    if num_cols:
        scaler = MinMaxScaler().fit(real[num_cols].to_numpy(dtype=np.float64))
        blocks_real.append(scaler.transform(real[num_cols].to_numpy(dtype=np.float64)))
        blocks_synth.append(scaler.transform(synth[num_cols].to_numpy(dtype=np.float64)))
    for col in cat_cols:
        levels = pd.unique(real[col].astype(object))
        index = {level: i for i, level in enumerate(levels)}
        for frame, blocks in ((real, blocks_real), (synth, blocks_synth)):
            block = np.zeros((len(frame), len(levels)), dtype=np.float64)
            codes = frame[col].astype(object).map(index)
            mask = codes.notna().to_numpy()
            block[np.nonzero(mask)[0], codes[mask].astype(int).to_numpy()] = scale
            blocks.append(block)
    if not blocks_real:
        raise ValueError("no usable columns to compute HEOM distances")
    return np.hstack(blocks_real), np.hstack(blocks_synth)


def identifiability_heom(
    real_csv_path: str,
    synth_csv_path: str,
    k: int = 2,
    exclude_columns: list[str] | None = None,
) -> PrivacyResult:
    """
    HEOM nearest-neighbour privacy metrics of the synthetic dataset with respect to the real one.

    For every real record x_i, r_i is the HEOM distance to its k-th nearest neighbour
    among the real records (k=2 means its nearest other record).
    - identifiability: fraction of real records whose nearest synthetic record is closer
      than r_i (honest sample of equal size: about 0.5; verbatim copy: 1.0). Higher is worse.
    - epsilon_any: fraction of synthetic rows that sit inside at least one real ball, i.e.
      closer to some real record than that record's own neighbourhood (the thesis filter
      statistic). Lower is better; its honest-sample level is high and data-dependent.
    - exact_match_fraction: fraction of synthetic rows identical to some real row.

    Args:
        real_csv_path: path to the real dataset CSV
        synth_csv_path: path to the synthetic dataset CSV
        k: neighbour rank that defines each real record's radius (default 2, the thesis setting)
        exclude_columns: columns to leave out of the distance (identifiers, free text)

    Returns:
        PrivacyResult (see its docstring).
    """
    if k < 2:
        raise ValueError("k must be at least 2: k=1 would make every real radius zero")
    real_data = pd.read_csv(real_csv_path)
    synthetic_data = pd.read_csv(synth_csv_path)
    drop = set(exclude_columns or [])
    columns = [c for c in real_data.columns if c not in drop and c in synthetic_data.columns]
    if not columns:
        raise ValueError("the two files share no usable columns")
    real = real_data[columns]
    synth = synthetic_data[columns]

    n_before = len(real) + len(synth)
    real = real.dropna()
    synth = synth.dropna()
    dropped = n_before - len(real) - len(synth)
    if len(real) < k:
        raise ValueError(f"need at least {k} complete real rows, got {len(real)}")
    if len(synth) == 0:
        raise ValueError("no complete synthetic rows")

    categorical = set(real.select_dtypes(include=["object", "category", "string"]).columns)
    cat_cols = [c for c in columns if c in categorical]
    num_cols = [c for c in columns if c not in categorical]

    x_real, x_synth = _heom_encode(real, synth, num_cols, cat_cols)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_real)
    real_dists, _ = knn.kneighbors(x_real)
    radii = real_dists[:, k - 1]
    r_max = float(radii.max())

    nn1_dists, _ = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(x_real).kneighbors(x_synth)
    nearest = nn1_dists[:, 0]
    exact = int(np.sum(nearest <= 1e-9))

    # identifiability (real side): nearest synthetic closer than nearest other real, on a
    # size-matched synthetic subsample so the honest-sample reference stays at 0.5
    n_used = min(len(synth), len(real))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(synth), size=n_used, replace=False) if n_used < len(synth) else np.arange(len(synth))
    real_to_synth, _ = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(x_synth[pick]).kneighbors(x_real)
    identified = int(np.sum(real_to_synth[:, 0] < real_dists[:, 1]))

    unsafe = 0
    if r_max > 0:
        rad = NearestNeighbors(radius=r_max, metric="euclidean").fit(x_real)
        dist_lists, idx_lists = rad.radius_neighbors(x_synth, radius=r_max, return_distance=True)
        for dists, idxs in zip(dist_lists, idx_lists, strict=True):
            if len(idxs) and bool(np.any(dists < radii[idxs])):
                unsafe += 1

    return {
        "identifiability": identified / len(real),
        "n_real_identified": identified,
        "n_synthetic_used_for_identifiability": int(n_used),
        "epsilon_any": unsafe / len(synth),
        "n_unsafe_synthetic": unsafe,
        "n_synthetic": int(len(synth)),
        "n_real": int(len(real)),
        "k": k,
        "exact_matches": exact,
        "exact_match_fraction": exact / len(synth),
        "median_synth_to_real_distance": float(np.median(nearest)),
        "median_real_knn_radius": float(np.median(radii)),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "n_rows_dropped_for_missing_values": int(dropped),
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

