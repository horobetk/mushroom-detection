#!/usr/bin/env python3
"""
Field-research aggregator for the Mushroom Detection app (Research Mode).

Turns the raw `MushroomResearch/session_*/` bundles pulled from the phone into
verifiable, thesis-ready statistics: top-1 / top-5 accuracy, per-species
precision / recall / F1, a confusion matrix, and merged CSVs.

Workflow
--------
1. Pull data from the phone:
       adb pull /sdcard/Documents/MushroomResearch ./research_data

2. Build a labeling template (one row per capture, with the app's prediction
   and the paths to the photos so you can eyeball each one):
       python analyze_field_research.py ./research_data --make-template

   -> writes `labeling.csv`. Open it, look at each `fullres_image`, and fill
      the empty `true_species` column with the REAL species. Leave a row blank
      to skip it. `true_species=none` marks "no mushroom in frame".

3. Generate the report once ground truth is filled in:
       python analyze_field_research.py ./research_data

   -> writes into `<data_dir>/_analysis/`:
        - all_detections.csv   (every detection row, all sessions merged)
        - captures.csv         (one row per capture: pred vs truth, hit flags)
        - confusion_matrix.csv / .png
        - summary.json         (accuracy, top-5, counts, gpu/latency stats)

Ground truth may also come from a per-session `ground_truth.txt`
(`expected_species=...` for the whole session, or `3=Boletus edulis` per
capture). `labeling.csv` takes priority when both exist.

Safety analysis
---------------
If `species_toxicity.csv` (classId,species,toxicity,isDangerous) sits next to
this script or in the data folder, the report also computes the safety-critical
metrics: a dangerous-vs-safe 2x2 matrix, the FALSE-SAFE rate (a truly poisonous
/ deadly mushroom reported as non-dangerous) and, most importantly, how often a
dangerous mushroom was labelled outright EDIBLE.

Only dependencies: pandas, matplotlib (numpy comes with pandas).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    HAVE_MPL = False


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _norm(name: str | None) -> str:
    """Normalise a species string for robust comparison."""
    if name is None:
        return ""
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def find_meta_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*_meta.json"))


def load_captures(data_dir: Path) -> pd.DataFrame:
    """One row per capture: primary prediction + top-5 + telemetry + image paths."""
    rows: list[dict] = []
    for meta_path in find_meta_files(data_dir):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] skipping unreadable {meta_path}: {exc}", file=sys.stderr)
            continue

        session = meta_path.parent.name
        cid = int(meta.get("captureId", 0))
        dets = meta.get("detections", []) or []
        primary = max(dets, key=lambda d: d.get("confidence", 0.0)) if dets else None

        top5 = []
        if primary:
            top5 = [t.get("species") for t in primary.get("topN", [])][:5]

        folder = meta_path.parent
        stem = f"capture_{cid:03d}__"
        fullres = next((p for p in folder.glob(f"{stem}*_fullres.jpg")), None)
        annotated = next((p for p in folder.glob(f"{stem}*_annotated.jpg")), None)
        fullframe = next((p for p in folder.glob(f"{stem}*_fullframe.jpg")), None)

        rows.append(
            {
                "session": session,
                "captureId": cid,
                "timestamp": meta.get("timestamp"),
                "n_detections": len(dets),
                "pred_species": primary.get("species") if primary else None,
                "pred_classId": primary.get("classId") if primary else None,
                "pred_conf": round(primary.get("confidence", 0.0), 4) if primary else None,
                "pred_toxicity": primary.get("toxicity") if primary else None,
                "pred_top5": " | ".join(str(s) for s in top5),
                "latencyMs": meta.get("inferenceLatencyMs"),
                "fps": meta.get("fps"),
                "batteryPercent": meta.get("batteryPercent"),
                "batteryCelsius": meta.get("batteryCelsius"),
                "gpuDelegate": meta.get("gpuDelegate"),
                "fullres_image": str(fullres) if fullres else "",
                "annotated_image": str(annotated) if annotated else "",
                "fullframe_image": str(fullframe) if fullframe else "",
                "_top5_list": top5,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["session", "captureId"]).reset_index(drop=True)
    return df


def load_all_detections(data_dir: Path) -> pd.DataFrame:
    """Merge every per-session session_log.csv into one flat frame."""
    frames = []
    for csv_path in sorted(data_dir.rglob("session_log.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "session" not in df.columns:
                df.insert(0, "session", csv_path.parent.name)
            frames.append(df)
        except Exception as exc:
            print(f"[warn] skipping {csv_path}: {exc}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def parse_ground_truth_txt(data_dir: Path) -> dict[tuple[str, int | None], str]:
    """Parse per-session ground_truth.txt files.

    Returns {(session, captureId|None): true_species}. A None captureId is a
    session-wide default applied to every capture in that session.
    """
    gt: dict[tuple[str, int | None], str] = {}
    for gt_path in data_dir.rglob("ground_truth.txt"):
        session = gt_path.parent.name
        for raw in gt_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("expected_species="):
                val = line.split("=", 1)[1].strip()
                if val:
                    gt[(session, None)] = val
                continue
            m = re.match(r"^(\d+)\s*=\s*(.+)$", line)
            if m:
                gt[(session, int(m.group(1)))] = m.group(2).strip()
    return gt


def apply_ground_truth(captures: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Fill a `true_species` column from labeling.csv (priority) or ground_truth.txt."""
    captures = captures.copy()
    captures["true_species"] = ""

    # Source 1: ground_truth.txt (session default + per-capture overrides).
    gt = parse_ground_truth_txt(data_dir)
    for i, row in captures.iterrows():
        key_cap = (row["session"], int(row["captureId"]))
        key_sess = (row["session"], None)
        if key_cap in gt:
            captures.at[i, "true_species"] = gt[key_cap]
        elif key_sess in gt:
            captures.at[i, "true_species"] = gt[key_sess]

    # Source 2: labeling.csv overrides anything above.
    labeling = data_dir / "labeling.csv"
    if labeling.exists():
        lab = pd.read_csv(labeling)
        if "true_species" in lab.columns:
            lab_map = {
                (r["session"], int(r["captureId"])): str(r["true_species"]).strip()
                for _, r in lab.iterrows()
                if str(r.get("true_species", "")).strip()
                and str(r.get("true_species", "")).lower() != "nan"
            }
            for i, row in captures.iterrows():
                key = (row["session"], int(row["captureId"]))
                if key in lab_map:
                    captures.at[i, "true_species"] = lab_map[key]

    return captures


# --------------------------------------------------------------------------- #
# Toxicity / safety
# --------------------------------------------------------------------------- #
DANGEROUS = {"POISONOUS", "DEADLY"}


def load_toxicity_map(data_dir: Path, explicit: Path | None) -> dict[str, str]:
    """Map normalised species name -> toxicity class (EDIBLE/INEDIBLE/...)."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        data_dir / "species_toxicity.csv",
        Path(__file__).resolve().parent / "species_toxicity.csv",
    ]
    for path in candidates:
        if path and path.exists():
            df = pd.read_csv(path)
            return {
                _norm(r["species"]): str(r["toxicity"]).strip().upper()
                for _, r in df.iterrows()
            }
    return {}


def compute_safety(labeled: pd.DataFrame, tox_map: dict[str, str], out_dir: Path) -> dict:
    """Safety-critical metrics based on true vs predicted toxicity danger."""
    df = labeled.copy()
    df["true_tox"] = df["true_species"].map(lambda s: tox_map.get(_norm(s), "UNKNOWN"))
    df["pred_tox"] = df["pred_species"].map(
        lambda s: tox_map.get(_norm(s), "NONE" if not s else "UNKNOWN")
    )

    # Restrict to rows where we actually know the true toxicity (skip 'none'/unknown truth).
    known = df[df["true_tox"] != "UNKNOWN"].copy()
    if known.empty:
        return {"note": "No true-species matched the toxicity map; check species names."}

    known["true_dangerous"] = known["true_tox"].isin(DANGEROUS)
    # A prediction is 'dangerous-flagged' only if the predicted species is poisonous/deadly.
    known["pred_dangerous"] = known["pred_tox"].isin(DANGEROUS)

    tp = int(((known["true_dangerous"]) & (known["pred_dangerous"])).sum())   # danger caught
    fn = int(((known["true_dangerous"]) & (~known["pred_dangerous"])).sum())  # FALSE-SAFE (worst)
    fp = int(((~known["true_dangerous"]) & (known["pred_dangerous"])).sum())  # false alarm
    tn = int(((~known["true_dangerous"]) & (~known["pred_dangerous"])).sum())

    n_dangerous = tp + fn
    n_safe = tn + fp

    # The most critical single error: truly dangerous, but predicted EDIBLE.
    dangerous_as_edible = known[
        (known["true_dangerous"]) & (known["pred_tox"] == "EDIBLE")
    ]

    safety = {
        "evaluated_captures": int(len(known)),
        "true_dangerous": n_dangerous,
        "true_safe": n_safe,
        "matrix": {
            "danger_caught (TP)": tp,
            "false_safe (FN)": fn,
            "false_alarm (FP)": fp,
            "safe_ok (TN)": tn,
        },
        "false_safe_rate": round(fn / n_dangerous, 4) if n_dangerous else None,
        "danger_recall": round(tp / n_dangerous, 4) if n_dangerous else None,
        "false_alarm_rate": round(fp / n_safe, 4) if n_safe else None,
        "dangerous_reported_edible": int(len(dangerous_as_edible)),
    }

    # Persist the 2x2 and the list of critical misses for the thesis.
    cm = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["true: DANGEROUS", "true: SAFE"],
        columns=["pred: DANGEROUS", "pred: SAFE"],
    )
    cm.to_csv(out_dir / "safety_matrix.csv")

    if not dangerous_as_edible.empty:
        dangerous_as_edible[
            ["session", "captureId", "true_species", "true_tox",
             "pred_species", "pred_conf", "fullres_image"]
        ].to_csv(out_dir / "critical_dangerous_as_edible.csv", index=False, encoding="utf-8-sig")

    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm.values, cmap="Reds", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(cm.columns); ax.set_yticklabels(cm.index)
        ax.set_title("Safety matrix (danger vs safe)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm.values[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "safety_matrix.png", dpi=150)
        plt.close(fig)

    return safety


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def compute_metrics(captures: pd.DataFrame) -> dict:
    labeled = captures[captures["true_species"].str.strip() != ""].copy()
    total = len(captures)
    n_labeled = len(labeled)

    result = {
        "captures_total": total,
        "captures_labeled": n_labeled,
        "captures_unlabeled": total - n_labeled,
    }
    if n_labeled == 0:
        result["note"] = "No ground truth found. Fill labeling.csv or ground_truth.txt."
        return result

    def top1_hit(r):
        return _norm(r["pred_species"]) == _norm(r["true_species"])

    def top5_hit(r):
        truth = _norm(r["true_species"])
        return truth in {_norm(s) for s in (r["_top5_list"] or [])}

    labeled["top1_hit"] = labeled.apply(top1_hit, axis=1)
    labeled["top5_hit"] = labeled.apply(top5_hit, axis=1)

    result["top1_accuracy"] = round(labeled["top1_hit"].mean(), 4)
    result["top5_accuracy"] = round(labeled["top5_hit"].mean(), 4)
    result["mean_confidence"] = round(labeled["pred_conf"].dropna().mean(), 4) if labeled["pred_conf"].notna().any() else None
    result["mean_latencyMs"] = round(labeled["latencyMs"].dropna().mean(), 1) if labeled["latencyMs"].notna().any() else None

    # Per-species (based on true label) top-1 recall.
    per_species = (
        labeled.groupby("true_species")["top1_hit"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n", "sum": "correct"})
    )
    per_species["accuracy"] = (per_species["correct"] / per_species["n"]).round(3)
    result["per_species"] = per_species.reset_index().to_dict(orient="records")

    return result, labeled


def save_confusion(labeled: pd.DataFrame, out_dir: Path) -> None:
    truth = labeled["true_species"].map(_norm)
    pred = labeled["pred_species"].map(lambda s: _norm(s) or "(none)")
    cm = pd.crosstab(truth, pred, dropna=False)
    cm.to_csv(out_dir / "confusion_matrix.csv")

    if HAVE_MPL and not cm.empty:
        n = max(cm.shape)
        size = max(4, min(20, 0.6 * n + 2))
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(cm.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        ax.set_xticklabels(cm.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(cm.index, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Field Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm.values[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center",
                            color="white", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)


def make_template(captures: pd.DataFrame, data_dir: Path) -> None:
    cols = [
        "session", "captureId", "n_detections",
        "pred_species", "pred_conf", "pred_top5",
        "fullres_image", "annotated_image", "true_species",
    ]
    template = captures.copy()
    template["true_species"] = ""
    out = data_dir / "labeling.csv"
    template[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[ok] wrote labeling template: {out}")
    print(f"     {len(template)} captures. Open it, look at each fullres_image,")
    print("     and fill the `true_species` column with the real species.")


def make_report(captures: pd.DataFrame, all_dets: pd.DataFrame, data_dir: Path,
                toxicity_map: Path | None = None) -> None:
    out_dir = data_dir / "_analysis"
    out_dir.mkdir(exist_ok=True)

    if not all_dets.empty:
        all_dets.to_csv(out_dir / "all_detections.csv", index=False, encoding="utf-8-sig")

    captures = apply_ground_truth(captures, data_dir)
    export_cols = [c for c in captures.columns if c != "_top5_list"]
    captures[export_cols].to_csv(out_dir / "captures.csv", index=False, encoding="utf-8-sig")

    metrics = compute_metrics(captures)
    if isinstance(metrics, tuple):
        metrics, labeled = metrics
        save_confusion(labeled, out_dir)

        # Safety-critical analysis (needs the species -> toxicity map).
        tox_map = load_toxicity_map(data_dir, toxicity_map)
        if tox_map:
            metrics["safety"] = compute_safety(labeled, tox_map, out_dir)
        else:
            metrics["safety"] = {"note": "species_toxicity.csv not found; skipped."}

    (out_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[ok] report written to: {out_dir}")
    print(json.dumps(
        {k: v for k, v in metrics.items() if k not in ("per_species",)},
        indent=2, ensure_ascii=False,
    ))
    if metrics.get("captures_labeled", 0) == 0:
        print("\n[!] No ground truth yet. Run with --make-template, fill true_species, re-run.")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate field Research Mode data.")
    ap.add_argument("data_dir", type=Path, help="Folder with session_* subdirs (adb-pulled).")
    ap.add_argument("--make-template", action="store_true",
                    help="Write labeling.csv (one row per capture) instead of a report.")
    ap.add_argument("--toxicity-map", type=Path, default=None,
                    help="Path to species_toxicity.csv (defaults next to this script / in data dir).")
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"[error] not found: {args.data_dir}", file=sys.stderr)
        return 2

    captures = load_captures(args.data_dir)
    if captures.empty:
        print("[error] no *_meta.json files found under the given folder.", file=sys.stderr)
        return 1

    if args.make_template:
        make_template(captures, args.data_dir)
    else:
        all_dets = load_all_detections(args.data_dir)
        make_report(captures, all_dets, args.data_dir, args.toxicity_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
