
"""
Alarm Prediction ML Pipeline
----------------------------
Reads per-(H3 cell, time period) features (crash_count, Gi*, hotspot_type, etc.),
merges trend features, builds lag/rolling features per cell, and trains an
ML model to predict the alarm type ("New Hotspot", "Intensifying Hotspot",
"Persistent Hotspot", "Diminishing Hotspot", "No Alarm") for the latest time period.

Usage:
    python alarm_ml_pipeline.py --hotspot /path/hotspot_analysis.csv \
        --trends /path/trend_analysis.csv \
        --alarms /path/spatiotemporal_alarms.csv \
        --out_csv /path/predicted_alarms_latest_period.csv \
        --out_json /path/predicted_alarms_latest_period.json
"""

import argparse
import pandas as pd
import numpy as np
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def parse_time_period(tp):
    m = re.search(r"Week[_\s]*(\d+)[_\s]+(\d{2})-(\d{2})", str(tp))
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    nums = re.findall(r"\d+", str(tp))
    if len(nums) >= 3:
        return int(nums[0]), int(nums[1]), int(nums[2])
    return None, None, None

def build_pipeline():
    categorical = [
        "hotspot_type", "trend_significance", "trend_direction",
    ]
    numeric = [
        "crash_count","gi_star","gi_p_value","is_hot_sig","is_cold_sig",
        "crash_count_lag1","crash_count_lag2",
        "gi_star_lag1","gi_star_lag2",
        "is_hot_sig_lag1","is_hot_sig_lag2",
        "is_cold_sig_lag1","is_cold_sig_lag2",
        "crash_rolling_mean_2","crash_rolling_sum_2",
        "gi_star_rolling_mean_2","hot_sig_rolling_sum_3",
        "trend_slope","trend_p_value","total_crashes"
    ]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric),
        ],
        remainder="drop"
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42
    )
    return Pipeline(steps=[("prep", pre), ("clf", clf)]), categorical, numeric

def main(args):
    hotspot = pd.read_csv(args.hotspot)
    trends = pd.read_csv(args.trends)
    alarms = pd.read_csv(args.alarms)

    hotspot.columns = [c.strip().lower() for c in hotspot.columns]
    trends.columns = [c.strip().lower() for c in trends.columns]
    alarms.columns = [c.strip().lower() for c in alarms.columns]

    hotspot = hotspot.rename(columns={
        "h3_cell": "h3_cell_id",
        "time_bin": "time_period",
        "center_lat": "centroid_lat",
        "center_lon": "centroid_lon",
    })
    alarms = alarms.rename(columns={
        "h3_cell_id": "h3_cell_id",
        "time_period": "time_period",
        "alarm_type": "alarm_type",
        "crash_count": "crash_count",
        "gi_star_score": "gi_star",
    })

    hotspot[["week_num","mm","dd"]] = hotspot["time_period"].apply(lambda x: pd.Series(parse_time_period(x)))
    alarms[["week_num","mm","dd"]] = alarms["time_period"].apply(lambda x: pd.Series(parse_time_period(x)))

    time_order = (
        hotspot[["time_period","week_num","mm","dd"]]
        .drop_duplicates()
        .sort_values(by=["week_num","mm","dd"])
        .reset_index(drop=True)
    )
    time_order["t_idx"] = np.arange(len(time_order))
    hotspot = hotspot.merge(time_order[["time_period","t_idx"]], on="time_period", how="left")
    alarms = alarms.merge(time_order[["time_period","t_idx"]], on="time_period", how="left")

    alarm_lookup = alarms.set_index(["h3_cell_id","time_period"])["alarm_type"].to_dict()
    hotspot["alarm_type"] = hotspot.apply(
        lambda r: alarm_lookup.get((r["h3_cell_id"], r["time_period"]), "No Alarm"), axis=1
    )

    hotspot = hotspot.sort_values(["h3_cell_id","t_idx"]).reset_index(drop=True)
    hotspot["is_hot_sig"] = ((hotspot.get("gi_p_value", 1.0) < 0.05) & (hotspot.get("gi_star", 0.0) > 0)).astype(int)
    hotspot["is_cold_sig"] = ((hotspot.get("gi_p_value", 1.0) < 0.05) & (hotspot.get("gi_star", 0.0) < 0)).astype(int)

    def add_lags(g, cols, lags=(1,2)):
        for col in cols:
            for L in lags:
                g[f"{col}_lag{L}"] = g[col].shift(L)
        return g

    hotspot = hotspot.groupby("h3_cell_id", group_keys=False).apply(
        add_lags, cols=["crash_count","gi_star","is_hot_sig","is_cold_sig"], lags=(1,2)
    )

    def add_rolls(g):
        g["crash_rolling_mean_2"] = g["crash_count"].rolling(2, min_periods=1).mean()
        g["crash_rolling_sum_2"] = g["crash_count"].rolling(2, min_periods=1).sum()
        g["gi_star_rolling_mean_2"] = g["gi_star"].rolling(2, min_periods=1).mean()
        g["hot_sig_rolling_sum_3"] = g["is_hot_sig"].rolling(3, min_periods=1).sum()
        return g

    hotspot = hotspot.groupby("h3_cell_id", group_keys=False).apply(add_rolls)

    features = hotspot.merge(trends.rename(columns={"h3_cell":"h3_cell_id"}), on="h3_cell_id", how="left")

    t_max = features["t_idx"].max()
    t_train_max = t_max - 1
    train_df = features[features["t_idx"] <= t_train_max].copy()
    test_df  = features[features["t_idx"] == t_max].copy()

    train_df = train_df.dropna(subset=["crash_count_lag1","gi_star_lag1","is_hot_sig_lag1","is_cold_sig_lag1"])

    model, categorical, numeric = build_pipeline()
    X_train = train_df[categorical + numeric]
    y_train = train_df["alarm_type"].astype("category")
    X_test  = test_df[categorical + numeric]
    y_test  = test_df["alarm_type"].astype("category")

    # Ensure all numeric columns exist
    for col in numeric:
        if col not in X_train.columns:
            X_train[col] = np.nan
            X_test[col] = np.nan

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
        proba_cols = list(model.named_steps["clf"].classes_)
    except Exception:
        y_proba = None
        proba_cols = []

    test_pred = test_df.copy()
    test_pred["pred_alarm_type"] = y_pred
    if y_proba is not None:
        proba_df = pd.DataFrame(y_proba, columns=proba_cols)
        test_pred = pd.concat([test_pred.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
        test_pred["pred_alarm_confidence"] = test_pred[proba_cols].max(axis=1)
    else:
        test_pred["pred_alarm_confidence"] = np.nan

    keep_cols = [
        "h3_cell_id","centroid_lat","centroid_lon","time_period",
        "crash_count","gi_star","hotspot_type","pred_alarm_type","pred_alarm_confidence"
    ]
    proba_keep = [c for c in proba_cols if c in test_pred.columns]
    out = test_pred[keep_cols + proba_keep].copy()

    action_map = {
        "New Hotspot": "Investigate spike in incidents; dispatch a field team.",
        "Intensifying Hotspot": "Increase patrols/resources and consider temporary traffic calming.",
        "Persistent Hotspot": "Coordinate with city safety team for targeted interventions.",
        "Diminishing Hotspot": "Monitor closely; consider reducing dedicated resources.",
        "No Alarm": "No action needed beyond routine monitoring."
    }
    out["Recommended_Action"] = out["pred_alarm_type"].map(action_map).fillna("Review context and determine action.")

    out.to_csv(args.out_csv, index=False)
    out.to_json(args.out_json, orient="records", lines=False)
    print(f"Saved:\n  CSV: {args.out_csv}\n  JSON: {args.out_json}\n  Latest period: {out['time_period'].iloc[0] if len(out) else 'N/A'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotspot", required=True)
    parser.add_argument("--trends", required=True)
    parser.add_argument("--alarms", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    main(args)
