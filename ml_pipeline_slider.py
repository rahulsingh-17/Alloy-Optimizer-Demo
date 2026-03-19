"""
================================================================
  ML PIPELINE — SLIDER UI  (index.html)
================================================================
  This script trains the machine learning models used by the
  SLIDER-based web UI (index.html).

  HOW IT FITS IN:
  ───────────────
    1. Run this script  →  produces  ml_data.json
    2. Open index.html in a browser (via local server)
    3. index.html fetches ml_data.json and lets users drag
       dual range sliders (min ↔ max) per property to find
       the best Cu alloy composition.

  WHAT THIS SCRIPT DOES:
  ───────────────────────
    Step 1 — Load & preprocess Cu_alloys_database.csv
    Step 2 — Train one GradientBoostingRegressor per property:
                • Hardness (HV)
                • Electrical Conductivity (%IACS)
                • Ultimate Tensile Strength (MPa)
                • Yield Strength (MPa)
    Step 3 — Evaluate each model (R², MAE, 5-fold CV)
    Step 4 — Run trained models on ALL 1,831 records to get
              predicted property values for every alloy
    Step 5 — Export everything to ml_data.json

  ml_data.json STRUCTURE:
  ────────────────────────
    {
      "records": [
        {
          "formula":     "Cu-2Ni-0.5Si",
          "alloy_class": "Cu-Ni-Si alloys",
          "comp":  { "Cu": 97.5, "Ni": 2.0, "Si": 0.5 },
          "proc":  { "Tss_K": 1073, "tss_h": 1, "CR": 0,
                     "Tag_K": 723, "tag_h": 2, "aging": 1 },
          "pred":  { "hardness": 245.3, "conductivity": 42.1,
                     "uts": 720.5, "yield_strength": 610.2 },
          "actual":{ "hardness": 240.0, "conductivity": null, ... }
        }, ...
      ],
      "perf": {
        "hardness":      { "r2": 0.86, "mae": 20.5, "n": 1614, ... },
        "conductivity":  { ... },
        ...
      },
      "proc_medians": { "Tss_K": 1073, ... }
    }

  REQUIREMENTS:
  ─────────────
    pip install pandas numpy scikit-learn

  RUN:
  ────
    python ml_pipeline_slider.py

  FILES NEEDED IN SAME FOLDER:
    Cu_alloys_database.csv

  FILES PRODUCED:
    ml_data.json   ← loaded by index.html (slider UI)
================================================================
"""

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────

DATA_PATH   = "Cu_alloys_database.csv"
OUTPUT_PATH = "ml_data.json"            # loaded by index.html

# 27 element columns (wt%)
ELEMENT_COLS = [
    "Cu", "Al", "Ag", "B",  "Be", "Ca", "Co", "Ce", "Cr", "Fe",
    "Hf", "La", "Mg", "Mn", "Mo", "Nb", "Nd", "Ni", "P",  "Pb",
    "Pr", "Si", "Sn", "Ti", "V",  "Zn", "Zr"
]

# Processing parameter columns
PROCESS_COLS = [
    "Tss (K)",           # Solution treatment temperature
    "tss (h)",           # Solution treatment time
    "CR reduction (%)",  # Cold-rolling reduction
    "Tag (K)",           # Aging temperature
    "tag (h)",           # Aging time
]

# Encoded categoricals (created during preprocessing)
CAT_COLS = ["Aging_enc", "SecProc_enc"]

ALL_FEATURES = ELEMENT_COLS + PROCESS_COLS + CAT_COLS

# Target properties → short keys used in JSON
TARGETS = {
    "Hardness (HV)":                   "hardness",
    "Electrical conductivity (%IACS)": "conductivity",
    "Ultimate tensile strength (MPa)": "uts",
    "Yield strength (MPa)":            "yield_strength",
}

# Gradient Boosting hyperparameters
GBR_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.85,
    min_samples_leaf=3,
    random_state=42,
)


# ──────────────────────────────────────────────────────────
#  STEP 1 — LOAD & PREPROCESS
# ──────────────────────────────────────────────────────────

def load_data(path: str):
    """
    Load CSV, encode categorical columns,
    impute missing process parameters with column medians.

    Returns
    -------
    df           : preprocessed DataFrame
    proc_medians : dict {col_name: median_value}
    """
    print(f"\n{'─'*60}")
    print("  STEP 1 — Loading & preprocessing")
    print(f"{'─'*60}")

    df = pd.read_csv(path, sep=";", encoding="latin1")
    print(f"  ✓ {len(df)} records loaded")
    print(f"  ✓ Alloy classes: {df['Alloy class'].value_counts().to_dict()}")

    # Encode: Aging (Y/N) → 1/0
    df["Aging_enc"] = (df["Aging"] == "Y").astype(int)

    # Encode: Secondary thermo-mechanical process (Y/N) → 1/0
    df["SecProc_enc"] = (
        df["Secondary thermo-mechanical process"] == "Y"
    ).astype(int)

    # Impute missing process columns with column median
    proc_medians = {}
    for col in PROCESS_COLS:
        median_val = df[col].median()
        proc_medians[col] = float(median_val)
        n_missing = df[col].isna().sum()
        df[col]   = df[col].fillna(median_val)
        if n_missing:
            print(f"  ⚠  Imputed {n_missing} NaN in '{col}' → median={median_val:.2f}")

    print("\n  Target property coverage:")
    for col in TARGETS:
        n   = df[col].notna().sum()
        pct = n / len(df) * 100
        print(f"    {col:<42} {n:>4} records ({pct:.0f}%)")

    return df, proc_medians


# ──────────────────────────────────────────────────────────
#  STEP 2 — TRAIN MODELS
# ──────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame) -> dict:
    """
    Train one GradientBoostingRegressor per target property.
    Includes 5-fold cross-validation for honest evaluation.

    Returns
    -------
    trained : dict { key → {model, r2, mae, cv_r2, n, y_min, y_max, ...} }
    """
    print(f"\n{'─'*60}")
    print("  STEP 2 — Training ML models")
    print(f"  Params: {GBR_PARAMS}")
    print(f"{'─'*60}\n")

    trained = {}

    for col, key in TARGETS.items():
        mask = df[col].notna()
        n    = int(mask.sum())

        if n < 30:
            print(f"  ⚠  Skipping '{key}': only {n} samples")
            continue

        X = df.loc[mask, ALL_FEATURES].copy()
        y = df.loc[mask, col].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        model = GradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X_tr, y_tr)

        # Test-set evaluation
        y_pred = model.predict(X_te)
        r2  = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)

        # 5-fold cross-validation
        cv = cross_val_score(
            GradientBoostingRegressor(**GBR_PARAMS),
            X, y,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring="r2",
        )

        # Top-5 feature importances
        fi   = dict(zip(ALL_FEATURES, model.feature_importances_))
        top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]

        trained[key] = {
            "model":    model,
            "col":      col,
            "r2":       round(r2, 4),
            "mae":      round(float(mae), 2),
            "cv_r2":    round(float(cv.mean()), 4),
            "cv_std":   round(float(cv.std()), 4),
            "n":        n,
            "y_min":    round(float(y.min()), 1),
            "y_max":    round(float(y.max()), 1),
            "y_mean":   round(float(y.mean()), 2),
            "feat_imp": fi,
        }

        print(f"  [{key}]")
        print(f"    n={n}   R²={r2:.4f}   MAE={mae:.2f}")
        print(f"    CV R² = {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"    Top features: {', '.join(f[0] for f in top5)}\n")

    return trained


# ──────────────────────────────────────────────────────────
#  STEP 3 — PREDICT ON FULL DATASET
# ──────────────────────────────────────────────────────────

def predict_all(df: pd.DataFrame, trained: dict) -> dict:
    """
    Use every trained model to predict its property for every record.

    Returns
    -------
    preds : dict { key → np.array of length len(df) }
    """
    print(f"{'─'*60}")
    print(f"  STEP 3 — Predicting on all {len(df)} records")
    print(f"{'─'*60}")

    X_all = pd.DataFrame(df[ALL_FEATURES].values, columns=ALL_FEATURES)
    preds = {}

    for key, info in trained.items():
        preds[key] = info["model"].predict(X_all)
        print(f"  ✓ {key}")

    return preds


# ──────────────────────────────────────────────────────────
#  STEP 4 — EXPORT JSON
# ──────────────────────────────────────────────────────────

def export_json(
    df: pd.DataFrame,
    trained: dict,
    preds: dict,
    proc_medians: dict,
    output_path: str,
):
    """
    Build ml_data.json consumed by index.html (slider UI).
    """
    print(f"\n{'─'*60}")
    print(f"  STEP 4 — Exporting → {output_path}")
    print(f"{'─'*60}")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):

        # Composition: include Cu always, other elements only if > 0
        comp = {"Cu": round(float(row["Cu"]), 3)}
        for el in ELEMENT_COLS:
            if el != "Cu" and float(row[el]) > 0:
                comp[el] = round(float(row[el]), 4)

        # Processing params
        proc = {
            "Tss_K": round(float(row["Tss (K)"]),         1),
            "tss_h": round(float(row["tss (h)"]),          2),
            "CR":    round(float(row["CR reduction (%)"]), 1),
            "Tag_K": round(float(row["Tag (K)"]),          1)
                     if not pd.isna(row["Tag (K)"])
                     else round(proc_medians["Tag (K)"],   1),
            "tag_h": round(float(row["tag (h)"]),          2)
                     if not pd.isna(row["tag (h)"])
                     else round(proc_medians["tag (h)"],   2),
            "aging": int(row["Aging_enc"]),
        }

        # ML model predictions (all 4 properties for every record)
        pred = {
            key: round(float(preds[key][i]), 1)
            for key in preds
        }

        # Actual measured values (None if not measured)
        actual = {}
        for col, key in TARGETS.items():
            v = row[col]
            actual[key] = None if pd.isna(v) else round(float(v), 1)

        records.append({
            "formula":     row["Alloy formula"],
            "alloy_class": row["Alloy class"],
            "comp":        comp,
            "proc":        proc,
            "pred":        pred,
            "actual":      actual,
        })

    # Performance summary (no model objects — not JSON-serialisable)
    perf = {
        key: {
            "r2":     info["r2"],
            "mae":    info["mae"],
            "cv_r2":  info["cv_r2"],
            "cv_std": info["cv_std"],
            "n":      info["n"],
            "y_min":  info["y_min"],
            "y_max":  info["y_max"],
            "y_mean": info["y_mean"],
        }
        for key, info in trained.items()
    }

    # Clean process-median keys for JSON
    proc_med_clean = {
        col.replace(" ", "_")
           .replace("(", "")
           .replace(")", "")
           .replace("%", "pct"): round(v, 2)
        for col, v in proc_medians.items()
    }

    export = {
        "records":      records,
        "perf":         perf,
        "proc_medians": proc_med_clean,
    }

    with open(output_path, "w") as f:
        json.dump(export, f, separators=(",", ":"))

    size_kb = len(json.dumps(export, separators=(",", ":"))) // 1024
    print(f"  ✓ {len(records)} records exported")
    print(f"  ✓ File size : {size_kb} KB")
    print(f"  ✓ Saved to  : {output_path}")


# ──────────────────────────────────────────────────────────
#  STEP 5 — FEATURE IMPORTANCE REPORT
# ──────────────────────────────────────────────────────────

def report_feature_importances(trained: dict):
    print(f"\n{'─'*60}")
    print("  Feature Importance Report (top 10 per property)")
    print(f"{'─'*60}")

    for key, info in trained.items():
        print(f"\n  [{info['col']}]")
        for feat, imp in sorted(
            info["feat_imp"].items(), key=lambda x: -x[1]
        )[:10]:
            bar = "█" * int(imp * 80)
            print(f"    {feat:<22} {bar:<35} {imp:.4f}")


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  ML Pipeline — Slider UI (index.html)")
    print("═" * 60)

    df, proc_medians = load_data(DATA_PATH)
    trained          = train_models(df)
    preds            = predict_all(df, trained)
    export_json(df, trained, preds, proc_medians, OUTPUT_PATH)
    report_feature_importances(trained)

    print("\n" + "═" * 60)
    print("  Done!")
    print(f"  ✓ ml_data.json ready for  index.html")
    print("  ✓ Serve with: python -m http.server 8080")
    print("  ✓ Open: http://localhost:8080/index.html")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
