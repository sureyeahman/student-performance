"""Preprocessing: the ColumnTransformer used at the start of every pipeline."""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode text columns."""
    numeric = StandardScaler()
    categorical = OneHotEncoder(drop="if_binary", handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )
