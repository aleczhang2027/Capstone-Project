"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for NFL coverage classification (pff_passCoverage_collapsed).
build_model() must return an sklearn-compatible estimator.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

NUMERIC_FEATURES = [
    "x", "y", "s", "a", "dis", "o", "dir", "quarter", "down", "yardsToGo",
    "gameClock", "absoluteYardlineNumber", "yardlineNumber", "playClockAtSnap",
    "preSnapHomeScore", "preSnapVisitorScore", "preSnapHomeTeamWinProbability",
    "preSnapVisitorTeamWinProbability", "expectedPoints",
]

CATEGORICAL_FEATURES = [
    "event", "sideofball", "yardlineSide", "possessionTeam",
    "defensiveTeam", "offenseFormation", "receiverAlignment",
]

ENGINEERED = [
    "score_diff", "is_redzone", "down_x_yards", "wp_diff",
    "is_long_yardage", "is_short_yardage", "two_min_warning",
    "field_zone", "score_sign",
]

ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["score_diff"] = X["preSnapHomeScore"] - X["preSnapVisitorScore"]
        X["wp_diff"] = X["preSnapHomeTeamWinProbability"] - X["preSnapVisitorTeamWinProbability"]
        X["is_redzone"] = (X["absoluteYardlineNumber"] <= 20).astype(float)
        X["down_x_yards"] = X["down"] * X["yardsToGo"]
        X["is_long_yardage"] = (X["yardsToGo"] >= 8).astype(float)
        X["is_short_yardage"] = (X["yardsToGo"] <= 2).astype(float)
        X["two_min_warning"] = (X["gameClock"] <= 120).astype(float)
        ayl = X["absoluteYardlineNumber"]
        X["field_zone"] = np.where(ayl >= 75, 0, np.where(ayl >= 40, 1, 2)).astype(float)
        X["score_sign"] = np.sign(X["score_diff"]).astype(float)
        return X


def build_model():
    hgb_pre = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    cat_idx = list(range(len(CATEGORICAL_FEATURES)))
    hgb_pipe = Pipeline([
        ("eng", FeatureEngineer()),
        ("pre", hgb_pre),
        ("clf", HistGradientBoostingClassifier(
            max_iter=750, learning_rate=0.03, max_depth=9,
            min_samples_leaf=8, l2_regularization=0.08,
            class_weight="balanced", categorical_features=cat_idx,
            random_state=42,
        )),
    ])

    rf_pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("qt", QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=42)),
            ]), ALL_NUMERIC),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ]
    )
    rf_pipe = Pipeline([
        ("eng", FeatureEngineer()),
        ("pre", rf_pre),
        ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )),
    ])

    return VotingClassifier(
        estimators=[("hgb", hgb_pipe), ("rf", rf_pipe)],
        voting="soft",
    )
