import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL_COLS = ["possessionTeam", "defensiveTeam", "offenseFormation", "receiverAlignment"]

ENGINEERED = [
    "score_diff", "is_redzone", "down_x_yards", "wp_diff",
    "is_long_yardage", "is_short_yardage", "two_min_warning",
    "field_zone", "score_sign",
]


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


def build_model() -> Pipeline:
    # Categorical context columns get ordinal-encoded.
    # All tracking columns (off_p*/def_p*) + numeric context pass through untouched.
    # HGB handles NaN natively so no imputation needed for missing players.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="passthrough",
    )
    return Pipeline([
        ("eng", FeatureEngineer()),
        ("pre", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=8,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=42,
        )),
    ])
