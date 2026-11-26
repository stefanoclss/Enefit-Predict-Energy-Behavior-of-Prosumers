"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: row_summary_stats
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('row_summary_stats',)
PRIORITY_SIGNAL_COLUMNS: list[str] = ['target_lag_168h', 'target_lag_mean_4', 'target_lag_336h', 'target_lag_48h', 'target_lag_96h', 'hw_shortwave_radiation']


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit, human-readable engineered features one transformation per block."""
    out = df.copy()
    if not ACTIVE_FEATURE_OPS:
        return out

    fallback_cols = [c for c in PRIORITY_SIGNAL_COLUMNS if c in out.columns]
    if len(fallback_cols) < 2:
        for col in out.columns:
            if col not in fallback_cols and not str(col).endswith(("_isna", "_target_enc")):
                fallback_cols.append(str(col))
            if len(fallback_cols) >= 4:
                break

    # Row-wise summary moments across predictors
    if fallback_cols:
        out['row_numeric_mean'] = out[fallback_cols].mean(axis=1)
        out['row_numeric_std'] = out[fallback_cols].std(axis=1).fillna(0.0)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
