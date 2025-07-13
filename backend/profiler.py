import pandas as pd
import numpy as np

def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": [],
        "correlations": {},
    }

    # Column-level profiling
    for col in df.columns:
        series = df[col]
        col_profile = {
            "name": col,
            "dtype": str(series.dtype),
            "n_missing": int(series.isnull().sum()),
            "n_unique": int(series.nunique(dropna=True)),
            "sample_values": series.dropna().unique()[:5].tolist(),
        }

        # Type-specific analysis
        if pd.api.types.is_numeric_dtype(series):
            col_profile.update({
                "type": "numerical",
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "skew": float(series.skew())
            })
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == object:
            col_profile.update({
                "type": "categorical",
                "top": series.mode().iloc[0] if not series.mode().empty else None,
                "freq_top": int(series.value_counts().iloc[0]) if not series.value_counts().empty else 0
            })
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_profile.update({
                "type": "datetime",
                "min": str(series.min()),
                "max": str(series.max())
            })
        else:
            col_profile.update({
                "type": "other"
            })

        profile["columns"].append(col_profile)

    # Correlation matrix for numerical columns
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        corr = num_cols.corr().to_dict()
        profile["correlations"] = corr

    return profile
