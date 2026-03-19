"""
================================================================
  ML PIPELINE — NLP UI  (alloy_nlp_ui.html)
================================================================
  This script trains the machine learning models used by the
  NATURAL LANGUAGE web UI (alloy_nlp_ui.html).

  HOW IT FITS IN:
  ───────────────
    1. Run this script  →  re-embeds ml_data into alloy_nlp_ui.html
    2. Open alloy_nlp_ui.html in any browser (no server needed)
    3. User types plain-English queries like:
         "high strength alloy for springs"
         "hardness above 300 HV and conductivity above 50"
       The built-in JavaScript NLP engine parses the text,
       extracts property targets, and the ML data (baked into
       the HTML) is searched to find the best composition.

  HOW THE NLP + ML WORK TOGETHER:
  ─────────────────────────────────
    Python (this script):
      • Trains GradientBoostingRegressor models offline
      • Predicts all 4 properties for all 1,831 records
      • Embeds the predictions into the HTML file

    JavaScript (inside alloy_nlp_ui.html):
      • NLP engine parses the user's text → extracts targets
        (keywords, intensity words, numbers, operators, context)
      • Weighted scorer ranks all embedded records
      • Top matches displayed with composition + processing info

  WHAT THIS SCRIPT DOES:
  ───────────────────────
    Step 1 — Load & preprocess Cu_alloys_database.csv
    Step 2 — Train one GBR model per target property
    Step 3 — Predict on all 1,831 records
    Step 4 — Inject the predictions into alloy_nlp_ui.html
              by replacing the ML_DATA placeholder block

  REQUIREMENTS:
  ─────────────
    pip install pandas numpy scikit-learn

  RUN:
  ────
    python ml_pipeline_nlp.py

  FILES NEEDED IN SAME FOLDER:
    Cu_alloys_database.csv
    alloy_nlp_ui_template.html   (or existing alloy_nlp_ui.html)

  FILES PRODUCED / UPDATED:
    alloy_nlp_ui.html   ← open this in browser directly
================================================================
"""

import json
import re
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

DATA_PATH    = "Cu_alloys_database.csv"
OUTPUT_HTML  = "alloy_nlp_ui.html"     # opened directly in browser
OUTPUT_JSON  = "ml_data_nlp.json"      # intermediate (also saved for inspection)

ELEMENT_COLS = [
    "Cu", "Al", "Ag", "B",  "Be", "Ca", "Co", "Ce", "Cr", "Fe",
    "Hf", "La", "Mg", "Mn", "Mo", "Nb", "Nd", "Ni", "P",  "Pb",
    "Pr", "Si", "Sn", "Ti", "V",  "Zn", "Zr"
]

PROCESS_COLS = [
    "Tss (K)", "tss (h)", "CR reduction (%)", "Tag (K)", "tag (h)"
]

CAT_COLS     = ["Aging_enc", "SecProc_enc"]
ALL_FEATURES = ELEMENT_COLS + PROCESS_COLS + CAT_COLS

TARGETS = {
    "Hardness (HV)":                   "hardness",
    "Electrical conductivity (%IACS)": "conductivity",
    "Ultimate tensile strength (MPa)": "uts",
    "Yield strength (MPa)":            "yield_strength",
}

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
    print(f"\n{'─'*60}")
    print("  STEP 1 — Loading & preprocessing")
    print(f"{'─'*60}")

    df = pd.read_csv(path, sep=";", encoding="latin1")
    print(f"  ✓ {len(df)} records, {len(df.columns)} columns")
    print(f"  ✓ Classes: {df['Alloy class'].value_counts().to_dict()}")

    df["Aging_enc"]   = (df["Aging"] == "Y").astype(int)
    df["SecProc_enc"] = (
        df["Secondary thermo-mechanical process"] == "Y"
    ).astype(int)

    proc_medians = {}
    for col in PROCESS_COLS:
        med = df[col].median()
        proc_medians[col] = float(med)
        n_na = df[col].isna().sum()
        df[col] = df[col].fillna(med)
        if n_na:
            print(f"  ⚠  {n_na} NaN in '{col}' imputed → {med:.2f}")

    print("\n  Target coverage:")
    for col in TARGETS:
        n   = df[col].notna().sum()
        print(f"    {col:<42} {n:>4} ({n/len(df)*100:.0f}%)")

    return df, proc_medians


# ──────────────────────────────────────────────────────────
#  STEP 2 — TRAIN MODELS
# ──────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame) -> dict:
    """
    Train GBR per property.
    The NLP UI uses predicted values to score alloys against
    the targets extracted from the user's text query.
    """
    print(f"\n{'─'*60}")
    print("  STEP 2 — Training ML models (GradientBoostingRegressor)")
    print(f"{'─'*60}\n")

    trained = {}

    for col, key in TARGETS.items():
        mask = df[col].notna()
        n    = int(mask.sum())
        if n < 30:
            continue

        X = df.loc[mask, ALL_FEATURES].copy()
        y = df.loc[mask, col].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        r2     = r2_score(y_te, y_pred)
        mae    = mean_absolute_error(y_te, y_pred)

        cv = cross_val_score(
            GradientBoostingRegressor(**GBR_PARAMS),
            X, y,
            cv=KFold(5, shuffle=True, random_state=42),
            scoring="r2",
        )

        fi   = dict(zip(ALL_FEATURES, model.feature_importances_))
        top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]

        trained[key] = {
            "model":  model,
            "col":    col,
            "r2":     round(r2, 4),
            "mae":    round(float(mae), 2),
            "cv_r2":  round(float(cv.mean()), 4),
            "cv_std": round(float(cv.std()), 4),
            "n":      n,
            "y_min":  round(float(y.min()), 1),
            "y_max":  round(float(y.max()), 1),
            "y_mean": round(float(y.mean()), 2),
        }

        print(f"  [{key}]")
        print(f"    n={n}  R²={r2:.4f}  MAE={mae:.2f}")
        print(f"    CV R² = {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"    Top features: {', '.join(f[0] for f in top5)}\n")

    return trained


# ──────────────────────────────────────────────────────────
#  STEP 3 — PREDICT ON ALL RECORDS
# ──────────────────────────────────────────────────────────

def predict_all(df: pd.DataFrame, trained: dict) -> dict:
    """
    Generate predictions for every record.
    These predictions are what the NLP UI searches through when
    the user types a query — it scores every record against the
    extracted property targets.
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
#  STEP 4 — BUILD EXPORT DICT
# ──────────────────────────────────────────────────────────

def build_export(
    df: pd.DataFrame,
    trained: dict,
    preds: dict,
    proc_medians: dict,
) -> dict:
    """
    Build the dictionary that gets embedded into alloy_nlp_ui.html
    as a JavaScript constant: const ML_DATA = { ... };

    The NLP UI JavaScript reads:
      ML_DATA.records[i].pred.hardness      → predicted hardness
      ML_DATA.records[i].comp               → element composition
      ML_DATA.perf.hardness.r2              → model R²
    """
    records = []

    for i, (_, row) in enumerate(df.iterrows()):
        comp = {"Cu": round(float(row["Cu"]), 3)}
        for el in ELEMENT_COLS:
            if el != "Cu" and float(row[el]) > 0:
                comp[el] = round(float(row[el]), 4)

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

        pred = {k: round(float(preds[k][i]), 1) for k in preds}

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

    perf = {
        key: {
            "r2":    info["r2"],
            "mae":   info["mae"],
            "cv_r2": info["cv_r2"],
            "n":     info["n"],
            "y_min": info["y_min"],
            "y_max": info["y_max"],
        }
        for key, info in trained.items()
    }

    proc_med_clean = {
        col.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct"): round(v, 2)
        for col, v in proc_medians.items()
    }

    return {
        "records":      records,
        "perf":         perf,
        "proc_medians": proc_med_clean,
    }


# ──────────────────────────────────────────────────────────
#  STEP 5 — INJECT INTO HTML
# ──────────────────────────────────────────────────────────

def inject_into_html(export: dict, html_path: str):
    """
    Replace the ML_DATA constant inside alloy_nlp_ui.html
    with the freshly trained model's predictions.

    Looks for:
        const ML_DATA = { ... };
    and replaces the entire block with the new data.

    If the HTML doesn't exist yet, saves the JSON separately
    and prints instructions.
    """
    print(f"\n{'─'*60}")
    print(f"  STEP 5 — Injecting into {html_path}")
    print(f"{'─'*60}")

    js_data    = json.dumps(export, separators=(",", ":"))
    new_block  = f"const ML_DATA = {js_data};"

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Replace existing ML_DATA block
        # Pattern matches: const ML_DATA = <anything>;
        pattern  = r"const ML_DATA\s*=\s*\{.*?\};"
        new_html = re.sub(pattern, new_block, html, count=1, flags=re.DOTALL)

        if new_html == html:
            print("  ⚠  Could not find 'const ML_DATA = {...};' block in HTML.")
            print("      Saving JSON separately instead.")
            _save_json(export)
            return

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(new_html)

        size_kb = len(new_html) // 1024
        print(f"  ✓ Injected {len(export['records'])} records into {html_path}")
        print(f"  ✓ File size: {size_kb} KB")

    except FileNotFoundError:
        print(f"  ⚠  {html_path} not found — saving JSON only.")
        _save_json(export)


def _save_json(export: dict):
    """Fallback: save as standalone JSON file."""
    with open(OUTPUT_JSON, "w") as f:
        json.dump(export, f, separators=(",", ":"))
    size_kb = len(json.dumps(export, separators=(",", ":"))) // 1024
    print(f"  ✓ Saved {OUTPUT_JSON} ({size_kb} KB)")
    print(f"  ℹ  Manually embed it into alloy_nlp_ui.html:")
    print(f"     Replace:  const ML_DATA = {{...}};")
    print(f"     With:     const ML_DATA = <contents of {OUTPUT_JSON}>;")


# ──────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE REPORT
# ──────────────────────────────────────────────────────────

def feature_report(trained: dict):
    print(f"\n{'─'*60}")
    print("  Feature Importances (top 8 per property)")
    print(f"{'─'*60}")
    for key, info in trained.items():
        print(f"\n  [{info['col']}]")
        fi_sorted = sorted(
            zip(ALL_FEATURES, info["model"].feature_importances_),
            key=lambda x: -x[1],
        )[:8]
        for feat, imp in fi_sorted:
            bar = "█" * int(imp * 75)
            print(f"    {feat:<22} {bar:<35} {imp:.4f}")


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  ML Pipeline — NLP UI (alloy_nlp_ui.html)")
    print("═" * 60)

    df, proc_medians = load_data(DATA_PATH)
    trained          = train_models(df)
    preds            = predict_all(df, trained)
    export           = build_export(df, trained, preds, proc_medians)

    # Save standalone JSON (useful for inspection)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(export, f, separators=(",", ":"))
    print(f"\n  ✓ Also saved: {OUTPUT_JSON}")

    # Inject into HTML
    inject_into_html(export, OUTPUT_HTML)

    feature_report(trained)

    print("\n" + "═" * 60)
    print("  Done!")
    print(f"  ✓ Open alloy_nlp_ui.html directly in any browser")
    print("  ✓ No server needed — all data is embedded in the HTML")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
