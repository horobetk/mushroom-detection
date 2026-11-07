#!/usr/bin/env python3
"""H2 safety matrix on the TFLite FP16 model using the app's post-processing."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import (  # noqa: E402
    CONF_EDIBLE,
    CONF_NEUTRAL_MIN,
    CONF_TOXIC,
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
)
from test_h2_baseline import (  # noqa: E402
    GLOBAL_TAU,
    GT_LABELS,
    IOU_MATCH,
    LOOKALIKE_PAIRS,
    PRED_LABELS,
    add_cell,
    cfpr,
    empty_matrix,
    far_edible,
    iou_xyxy,
    load_class_names,
    load_gt_boxes,
    load_registry,
    name_to_id,
    pred_asymmetric,
    pred_global,
    resolve_dataset_root,
    save_heatmap,
)
from test_tflite_parity import find_tflite, make_interpreter  # noqa: E402

# Must match FrameCropper / MushroomDetector on Android.
PAD_VALUE = 136
NMS_IOU = 0.45
MAX_NMS_INPUT = 50
MAX_DETECTIONS = 20


def letterbox_app(rgb: np.ndarray, size: int):
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    content_w, content_h = w * scale, h * scale
    pad_x, pad_y = (size - content_w) / 2.0, (size - content_h) / 2.0
    x1, y1 = int(round(pad_x)), int(round(pad_y))
    x2, y2 = int(round(pad_x + content_w)), int(round(pad_y + content_h))
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    out[y1:y2, x1:x2] = cv2.resize(rgb, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    return out, scale, pad_x, pad_y


def decode_app(raw: np.ndarray, class_thr: np.ndarray):
    t = np.squeeze(raw)
    if t.shape[0] > t.shape[1]:
        t = t.T
    scores = t[4:]
    best = scores.argmax(axis=0)
    best_s = scores[best, np.arange(t.shape[1])]
    keep = np.where(best_s >= class_thr[best])[0]
    if keep.size == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    keep = keep[np.argsort(-best_s[keep], kind="stable")][:MAX_NMS_INPUT]
    cx, cy, w, h = t[0, keep], t[1, keep], t[2, keep], t[3, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float64)
    return boxes, best_s[keep].astype(np.float64), best[keep]


def nms_app(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    order = list(range(len(scores)))
    kept: list[int] = []
    while order:
        i = order.pop(0)
        kept.append(i)
        if len(kept) >= MAX_DETECTIONS:
            break
        order = [j for j in order if iou_xyxy(boxes[i], boxes[j]) <= NMS_IOU]
    return kept


def to_image_coords(boxes, size, scale, pad_x, pad_y, img_w, img_h):
    if boxes.size == 0:
        return boxes
    b = boxes.copy()
    if float(np.abs(b).max()) <= 1.5:
        b *= size
    b[:, [0, 2]] = ((b[:, [0, 2]] - pad_x) / scale).clip(0, img_w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - pad_y) / scale).clip(0, img_h)
    return b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tflite", type=Path, default=None)
    p.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--toxicity-csv", type=Path, default=_TESTS_DIR / "toxicity_registry.csv")
    p.add_argument("--global-tau", type=float, default=GLOBAL_TAU)
    p.add_argument("--n", type=int, default=0, help="random subset size (0 = full test split)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tflite = find_tflite(args.tflite)
    if tflite is None:
        print("[ERR] TFLite FP16 not found. Pass --tflite path-to-file.tflite")
        sys.exit(2)

    args.output.mkdir(parents=True, exist_ok=True)
    names = load_class_names(args.data)
    tox = load_registry(args.toxicity_csv)
    n2i = name_to_id(names)
    class_thr = np.array(
        [CONF_TOXIC if tox.get(c) == "TOXIC" else CONF_NEUTRAL_MIN for c in range(len(names))],
        dtype=np.float32,
    )

    root, cfg = resolve_dataset_root(args.data)
    images_dir = Path(cfg.get("test", "images/test"))
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    labels_dir = root / "labels" / "test"
    files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if args.n and args.n < len(files):
        files = sorted(random.Random(args.seed).sample(files, args.n))

    interp = make_interpreter(tflite)
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    size = int(in_d["shape"][1])
    print(f"[INFO] tflite={tflite} ({tflite.stat().st_size / 1e6:.2f} MB)")
    print(f"[INFO] input={list(in_d['shape'])} output={list(out_d['shape'])}")
    print(f"[INFO] images={len(files)} classes={len(names)} toxic={int((class_thr == CONF_TOXIC).sum())}")

    pair_ids = [(n2i[a], n2i[b], a, b) for a, b in LOOKALIKE_PAIRS if a in n2i and b in n2i]
    pair_counts = defaultdict(int)
    pair_gt = defaultdict(int)
    mat_a, mat_g = empty_matrix(), empty_matrix()
    n_gt = 0
    n_pred = 0
    invoke_ms = []
    t0 = time.perf_counter()

    for k, img_path in enumerate(files, 1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]
        lb, scale, pad_x, pad_y = letterbox_app(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)
        x = (lb.astype(np.float32) / 255.0)[None, ...]
        interp.set_tensor(in_d["index"], x.astype(in_d["dtype"]))
        ti = time.perf_counter()
        interp.invoke()
        invoke_ms.append(1000.0 * (time.perf_counter() - ti))
        raw = interp.get_tensor(out_d["index"]).astype(np.float32)

        boxes, scores, clss = decode_app(raw, class_thr)
        keep = nms_app(boxes, scores) if scores.size else []
        boxes = to_image_coords(boxes[keep], size, scale, pad_x, pad_y, w0, h0) if keep else boxes[:0]
        preds = [
            {"box": tuple(boxes[i]), "cls": int(clss[keep[i]]), "conf": float(scores[keep[i]]), "used": False}
            for i in range(len(keep))
        ]
        n_pred += len(preds)

        for gt_cls, gt_box in load_gt_boxes(labels_dir / f"{img_path.stem}.txt", w0, h0):
            n_gt += 1
            gt_row = tox.get(gt_cls, "INEDIBLE")
            best_i, best_iou = -1, IOU_MATCH
            for i, pr in enumerate(preds):
                if pr["used"]:
                    continue
                s = iou_xyxy(gt_box, pr["box"])
                if s > best_iou:
                    best_i, best_iou = i, s
            if best_i < 0:
                add_cell(mat_a, gt_row, "NEUTRAL")
                add_cell(mat_g, gt_row, "NEUTRAL")
                continue
            preds[best_i]["used"] = True
            pcls, pconf = preds[best_i]["cls"], preds[best_i]["conf"]
            ptox = tox.get(pcls, "INEDIBLE")
            add_cell(mat_a, gt_row, pred_asymmetric(ptox, pconf))
            add_cell(mat_g, gt_row, pred_global(ptox, pconf, args.global_tau))
            for ia, ib, na, nb in pair_ids:
                if gt_cls == ia:
                    pair_gt[na] += 1
                    if pcls == ib:
                        pair_counts[f"{na} -> {nb}"] += 1
                elif gt_cls == ib:
                    pair_gt[nb] += 1
                    if pcls == ia:
                        pair_counts[f"{nb} -> {na}"] += 1

        if k % 500 == 0 or k == len(files):
            el = time.perf_counter() - t0
            eta = el / k * (len(files) - k)
            print(
                f"[INFO] {k}/{len(files)}  CFPR so far={cfpr(mat_a):.4f}/{cfpr(mat_g):.4f}  "
                f"eta={eta / 60:.1f} min"
            )

    ti_toxic = GT_LABELS.index("TOXIC")
    ti_edible = GT_LABELS.index("EDIBLE")
    pj = PRED_LABELS.index("JADALNY")
    pw = PRED_LABELS.index("OSTRZEZENIE")
    summary = {
        "backend": "tflite-interpreter, app pipeline (letterbox 136, per-class 0.18/0.20, top-50, agnostic NMS 0.45, max 20)",
        "tflite": str(tflite),
        "data": str(args.data),
        "n_test_images": len(files),
        "subset_seed": args.seed if args.n else None,
        "n_gt_instances": n_gt,
        "n_predictions": n_pred,
        "invoke_ms_mean_cpu": float(np.mean(invoke_ms)) if invoke_ms else None,
        "conf_toxic": CONF_TOXIC,
        "conf_edible": CONF_EDIBLE,
        "conf_neutral_min": CONF_NEUTRAL_MIN,
        "global_tau": args.global_tau,
        "vote_4_of_5": "not simulated (green counted as vote passed; CFPR is an upper bound)",
        "n_gt_toxic": int(mat_a[ti_toxic].sum()),
        "n_gt_edible": int(mat_a[ti_edible].sum()),
        "cfp_asymmetric": int(mat_a[ti_toxic, pj]),
        "cfp_global": int(mat_g[ti_toxic, pj]),
        "cfpr_asymmetric": cfpr(mat_a),
        "cfpr_global": cfpr(mat_g),
        "warn_edible_asymmetric": int(mat_a[ti_edible, pw]),
        "warn_edible_global": int(mat_g[ti_edible, pw]),
        "far_edible_asymmetric": far_edible(mat_a),
        "far_edible_global": far_edible(mat_g),
        "matrix_asymmetric": mat_a.tolist(),
        "matrix_global": mat_g.tolist(),
        "gt_labels": list(GT_LABELS),
        "pred_labels": list(PRED_LABELS),
        "lookalikes": [
            {"gt": s, "pred": d, "n_gt": pair_gt.get(s, 0), "n_confuse": pair_counts.get(f"{s} -> {d}", 0)}
            for a, b in LOOKALIKE_PAIRS
            for s, d in ((a, b), (b, a))
        ],
        "h2_holds": bool(cfpr(mat_a) < cfpr(mat_g)),
    }

    save_heatmap(mat_a, f"TFLite FP16 asymmetric 0.18/0.60 (CFPR={summary['cfpr_asymmetric']:.4f})",
                 args.output / "safety_matrix_tflite_asymmetric.png")
    save_heatmap(mat_g, f"TFLite FP16 global tau={args.global_tau:.2f} (CFPR={summary['cfpr_global']:.4f})",
                 args.output / "safety_matrix_tflite_global.png")
    out = args.output / "tflite_cfpr.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[OK] CFPR asymmetric={summary['cfp_asymmetric']}/{summary['n_gt_toxic']}="
        f"{summary['cfpr_asymmetric']:.4f}  global={summary['cfp_global']}/{summary['n_gt_toxic']}="
        f"{summary['cfpr_global']:.4f}"
    )
    print(
        f"[OK] FAR edible asymmetric={summary['far_edible_asymmetric']:.4f}  "
        f"global={summary['far_edible_global']:.4f}"
    )
    print(f"[OK] H2 (asymm < global): {summary['h2_holds']}")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
