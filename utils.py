import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the dataframe for fuel burn anomaly analysis.
    Ensures the presence of required columns, sorts by time,
    and derives burn rate if needed.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]  # Normalize column names

    # Ensure 'time_s' column exists
    if "time_s" not in df.columns:
        for alt in ["time", "t", "seconds"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "time_s"})
                break
    if "time_s" not in df.columns:
        raise ValueError("CSV must contain a 'time_s' column.")

    # Sort by time
    df = df.sort_values("time_s").reset_index(drop=True)

    # Derive burn rate if missing
    if "burn_rate_kg_s" not in df.columns:
        if "fuel_mass_kg" in df.columns:
            t = df["time_s"].values
            m = df["fuel_mass_kg"].values
            dm_dt = np.gradient(m, t)  # rate of change of mass
            burn_rate = -dm_dt         # burn rate is negative slope
            df["burn_rate_kg_s"] = burn_rate
        else:
            raise ValueError("Need either 'burn_rate_kg_s' or 'fuel_mass_kg' to compute burn rate.")

    # Add default stage column if missing
    if "stage" not in df.columns:
        df["stage"] = 1

    # Clean NaNs and infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "burn_rate_kg_s"])
    return df


def compute_expected_curve(df: pd.DataFrame, smoothing_window: int = 8) -> pd.DataFrame:
    """
    Compute the nominal burn rate curve per stage using rolling median
    and rolling mean smoothing.
    """
    out = df.copy()
    win = max(int(smoothing_window), 1)
    pieces = []

    for stage_val, g in out.groupby("stage", sort=True):
        g = g.sort_values("time_s").copy()
        med = g["burn_rate_kg_s"].rolling(window=win, min_periods=1, center=True).median()
        exp = med.rolling(window=win, min_periods=1, center=True).mean()
        g["expected_burn_rate"] = exp
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True).sort_values("time_s").reset_index(drop=True)
    out["residual"] = out["burn_rate_kg_s"] - out["expected_burn_rate"]
    out["zscore_stage"] = out.groupby("stage")["residual"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )
    return out


def detect_anomalies(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
    use_iforest: bool = True,
    contamination: float = 0.02
) -> pd.DataFrame:
    """
    Detect anomalies based on Z-score and optionally IsolationForest.
    """
    out = df.copy()

    # Z-score detection
    z_flag = out["zscore_stage"].abs() >= z_thresh

    # IsolationForest detection
    if use_iforest:
        iso_flags = np.zeros(len(out), dtype=bool)
        for stage_val, g in out.groupby("stage", sort=True):
            X = np.column_stack([g["time_s"].values, g["residual"].values])
            if len(g) >= 10:
                clf = IsolationForest(
                    n_estimators=200,
                    contamination=min(max(contamination, 1e-6), 0.5),
                    random_state=42,
                )
                y = clf.fit_predict(X)  # -1 = anomaly
                stage_flags = (y == -1)
            else:
                stage_flags = np.zeros(len(g), dtype=bool)
            iso_flags[g.index] = stage_flags
    else:
        iso_flags = np.zeros(len(out), dtype=bool)

    out["is_anomaly"] = z_flag | iso_flags
    return out


def summarize_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary dataframe of detected anomalies.
    """
    anom = df[df["is_anomaly"]].copy()
    if anom.empty:
        return pd.DataFrame(columns=["time_s", "stage", "actual_burn", "expected_burn", "deviation_%", "zscore"])

    anom["deviation_%"] = 100.0 * (
        (anom["burn_rate_kg_s"] - anom["expected_burn_rate"]) /
        anom["expected_burn_rate"].replace(0, np.nan)
    )

    anom = anom[[
        "time_s", "stage", "burn_rate_kg_s",
        "expected_burn_rate", "deviation_%", "zscore_stage"
    ]]
    anom.columns = ["time_s", "stage", "actual_burn", "expected_burn", "deviation_%", "zscore"]

    return anom.sort_values("time_s").reset_index(drop=True)
