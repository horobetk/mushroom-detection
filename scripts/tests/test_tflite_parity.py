#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

os.environ.setdefault("MPLBACKEND", "Agg")

import yaml
from ultralytics import YOLO

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import DEFAULT_DATA, DEFAULT_MODEL, DEFAULT_OUTPUT  # noqa: E402

KNOWN_FP16 = Path(
    "E:/Kiril_Horobets/mushroom_data/runs/fine_147classes_v2/weights/best_saved_model_fp16.tflite"
)


def find_tflite(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.exists():
        return explicit
    if KNOWN_FP16.is_file():
        return KNOWN_FP16
    repo = Path(__file__).resolve().parents[2]
    roots = [
        repo / "android" / "app" / "src" / "main" / "assets",
        Path("E:/Kiril_Horobets/mushroom_data"),
    ]
    fp16: list[Path] = []
    other: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.tflite"):
            if not p.is_file() or p.stat().st_size < 1_000_000:
                continue
            name = p.name.lower()
            if "int8" in name:
                continue
            if "fp16" in name or "float16" in name:
                fp16.append(p)
            elif "float32" in name or "fp32" in name:
                continue
            else:
                other.append(p)
    for p in fp16 + other:
        return p
    return None


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_root(data_yaml: Path, cfg: dict) -> Path:
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def test_images_dir(root: Path, cfg: dict) -> Path:
    test_key = cfg.get("test", "images/test")
    p = Path(test_key)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_subset_yaml(data_yaml: Path, n: int, seed: int, dest: Path) -> tuple[Path, int]:
    cfg = load_yaml(data_yaml)
    root = dataset_root(data_yaml, cfg)
    images_dir = test_images_dir(root, cfg)
    labels_dir = root / "labels" / "test"
    files = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise SystemExit(f"no test images in {images_dir}")
    n = min(n, len(files))
    rng = random.Random(seed)
    chosen = rng.sample(files, n)

    img_out = dest / "images" / "test"
    lab_out = dest / "labels" / "test"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    for img in chosen:
        link_or_copy(img, img_out / img.name)
        lab = labels_dir / f"{img.stem}.txt"
        if lab.is_file():
            link_or_copy(lab, lab_out / lab.name)

    names = cfg.get("names", [])
    out_yaml = dest / "data.yaml"
    payload = {
        "path": str(dest.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": names,
    }
    if "nc" in cfg:
        payload["nc"] = cfg["nc"]
    out_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    print(f"[INFO] subset yaml {out_yaml} n={n} of {len(files)} seed={seed}")
    return out_yaml, n


def run_val(weights: Path, data: Path, imgsz: int, device: str, max_det: int, task: str | None = None):
    model = YOLO(str(weights), task=task) if task else YOLO(str(weights))
    results = model.val(
        data=str(data),
        split="test",
        imgsz=imgsz,
        device=device,
        plots=False,
        verbose=True,
        max_det=max_det,
    )
    box = results.box
    return {
        "weights": str(weights),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "device": str(device),
        "backend": "ultralytics.val",
    }


def make_interpreter(model_path: Path):
    try:
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=str(model_path), num_threads=8)
    except Exception:
        from tflite_runtime.interpreter import Interpreter as LiteInterp

        interp = LiteInterp(model_path=str(model_path), num_threads=8)
    interp.allocate_tensors()
    return interp


def letterbox_rgb(rgb, size: int):
    import cv2
    import numpy as np

    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = size - nw, size - nh
    dw /= 2.0
    dh /= 2.0
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return out, r, left, top


def nms_xyxy(xyxy, scores, iou_thr: float, max_det: int):
    import numpy as np

    if not xyxy:
        return []
    boxes = np.asarray(xyxy, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    order = sc.argsort()[::-1]
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        ious = inter / (area_i + area_r - inter + 1e-9)
        order = rest[ious <= iou_thr]
    return keep


def decode_tflite(raw, conf_th: float):
    """Android layout: [channels, anchors] with ch0..3 = cx,cy,w,h in px or 0-1."""
    import numpy as np

    t = np.squeeze(raw)
    if t.ndim != 2:
        raise RuntimeError(f"unexpected output ndim {t.shape}")
    if t.shape[0] < t.shape[1]:
        ch, anc = t.shape
    else:
        t = t.T
        ch, anc = t.shape
    if ch < 5:
        raise RuntimeError(f"bad channels {t.shape}")
    nc = ch - 4
    scores = t[4:]
    best = scores.argmax(axis=0)
    best_s = scores[best, np.arange(anc)]
    keep = best_s >= conf_th
    idx = np.where(keep)[0]
    if idx.size == 0:
        return np.zeros((0, 4), dtype=np.float32), best_s[idx], best[idx], nc
    cx, cy, w, h = t[0, idx], t[1, idx], t[2, idx], t[3, idx]
    x1, y1 = cx - w * 0.5, cy - h * 0.5
    x2, y2 = cx + w * 0.5, cy + h * 0.5
    return np.stack([x1, y1, x2, y2], axis=1), best_s[idx], best[idx], nc


def scale_boxes(xyxy, r, pad_x, pad_y, img_w, img_h, in_size):
    import numpy as np

    if xyxy.size == 0:
        return xyxy
    boxes = xyxy.astype(np.float64).copy()
    if float(np.nanmax(np.abs(boxes))) <= 1.5:
        boxes *= in_size
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)
    return boxes


def load_gt(label_path: Path, w: int, h: int):
    boxes = []
    if not label_path.is_file():
        return boxes
    with label_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(float(p[0]))
            cx, cy, bw, bh = map(float, p[1:5])
            boxes.append(
                (
                    cid,
                    (
                        (cx - bw / 2) * w,
                        (cy - bh / 2) * h,
                        (cx + bw / 2) * w,
                        (cy + bh / 2) * h,
                    ),
                )
            )
    return boxes


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def match_predictions(pred_classes, true_classes, iou_mat, iouv):
    import numpy as np

    n_pred = int(pred_classes.shape[0])
    correct = np.zeros((n_pred, len(iouv)), dtype=bool)
    if n_pred == 0 or true_classes.shape[0] == 0:
        return correct
    iou_np = iou_mat.cpu().numpy() if hasattr(iou_mat, "cpu") else np.asarray(iou_mat)
    pred_np = pred_classes.cpu().numpy() if hasattr(pred_classes, "cpu") else np.asarray(pred_classes)
    true_np = true_classes.cpu().numpy() if hasattr(true_classes, "cpu") else np.asarray(true_classes)
    for i, thr in enumerate(iouv):
        matches = np.nonzero((iou_np >= thr) & (true_np[None, :] == pred_np[:, None]))
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 0].astype(int), i] = True
    return correct


def coco_stats(preds, gts):
    """Per-image (tp Nx10, conf, pred_cls, target_cls) for ultralytics ap_per_class."""
    import numpy as np
    import torch
    from ultralytics.utils.metrics import box_iou

    iouv = np.linspace(0.5, 0.95, 10)
    by_img_p: dict = {}
    by_img_g: dict = {}
    for name, cid, conf, box in preds:
        by_img_p.setdefault(name, []).append((cid, conf, box))
    for name, cid, box in gts:
        by_img_g.setdefault(name, []).append((cid, box))
    names = sorted(set(by_img_p) | set(by_img_g))
    stats = []
    for name in names:
        p = by_img_p.get(name, [])
        g = by_img_g.get(name, [])
        tcls = np.array([c for c, _ in g], dtype=np.int64)
        if not p:
            stats.append((np.zeros((0, 10), dtype=bool), np.zeros(0), np.zeros(0), tcls))
            continue
        pred_cls = np.array([c for c, _, _ in p], dtype=np.int64)
        conf = np.array([s for _, s, _ in p], dtype=np.float64)
        pred_box = torch.tensor([b for _, _, b in p], dtype=torch.float32)
        if g:
            gt_box = torch.tensor([b for _, b in g], dtype=torch.float32)
            iou_mat = box_iou(pred_box, gt_box)
        else:
            iou_mat = torch.zeros((len(p), 0))
        tp = match_predictions(pred_cls, tcls, iou_mat, iouv)
        stats.append((tp, conf, pred_cls, tcls))
    return stats


def ultralytics_maps(stats):
    import numpy as np
    from ultralytics.utils.metrics import ap_per_class

    if not stats:
        return 0.0, 0.0, 0.0, 0.0, 0
    tp = np.concatenate([s[0] for s in stats], 0)
    conf = np.concatenate([s[1] for s in stats], 0)
    pred_cls = np.concatenate([s[2] for s in stats], 0)
    target_cls = np.concatenate([s[3] for s in stats], 0)
    if target_cls.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    if tp.size == 0:
        return 0.0, 0.0, 0.0, 0.0, int(len(set(target_cls.tolist())))
    out = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    if len(out) >= 6 and getattr(out[5], "ndim", 0) == 2:
        p, r, ap = out[2], out[3], out[5]
    else:
        p, r, ap = out[0], out[1], out[2]
    mAP50 = float(ap[:, 0].mean()) if getattr(ap, "ndim", 0) == 2 else 0.0
    mAP50_95 = float(ap.mean()) if getattr(ap, "size", 0) else 0.0
    return mAP50, mAP50_95, float(np.mean(p)), float(np.mean(r)), int(ap.shape[0])


def eval_tflite_phone_decode(
    tflite: Path,
    images_dir: Path,
    labels_dir: Path,
    imgsz: int,
    conf_th: float,
    iou_nms: float,
    max_det: int,
):
    import time

    import cv2
    import numpy as np

    interp = make_interpreter(tflite)
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_shape = list(in_d["shape"])
    print(f"[INFO] TFLite input={in_shape} output={list(out_d['shape'])} in={in_d['dtype']} out={out_d['dtype']}")

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    preds = []
    gts = []
    t0 = time.perf_counter()
    n_cand = 0
    max_score = 0.0

    for i, img_path in enumerate(files, start=1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lb, r, pad_x, pad_y = letterbox_rgb(rgb, imgsz)
        x = lb.astype(np.float32) / 255.0
        if len(in_shape) == 4 and in_shape[1] == 3:
            x = np.transpose(x, (2, 0, 1))[None, ...]
        else:
            x = x[None, ...]
        if in_d["dtype"] == np.uint8:
            x = (x * 255).astype(np.uint8)
        elif str(in_d["dtype"]).endswith("float16"):
            x = x.astype(np.float16)
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        raw = interp.get_tensor(out_d["index"])
        scale_q, zp_q = out_d.get("quantization", (0.0, 0))
        if scale_q:
            raw = (raw.astype(np.float32) - zp_q) * scale_q
        xyxy, scores, clss, _nc = decode_tflite(raw, conf_th)
        if i == 1:
            raw_max = float(np.nanmax(np.abs(xyxy))) if xyxy.size else 0.0
            print(
                f"[INFO] first image {img_path.name} candidates={int(scores.size)} "
                f"score_max={float(scores.max()) if scores.size else 0:.4f} box_max={raw_max:.3f}"
            )
        if scores.size:
            max_score = max(max_score, float(scores.max()))
        n_cand += int(scores.size)
        xyxy = scale_boxes(xyxy, r, pad_x, pad_y, w0, h0, imgsz)
        keep = nms_xyxy(xyxy.tolist(), scores.tolist(), iou_nms, max_det) if scores.size else []
        for k in keep:
            preds.append((img_path.name, int(clss[k]), float(scores[k]), tuple(map(float, xyxy[k]))))
        for cid, box in load_gt(labels_dir / f"{img_path.stem}.txt", w0, h0):
            gts.append((img_path.name, cid, box))
        if i == 1 or i % 50 == 0 or i == len(files):
            print(f"[INFO] tflite-decode {i}/{len(files)} max_score={max_score:.4f} raw_cand={n_cand}")

    elapsed = time.perf_counter() - t0
    ms = 1000.0 * elapsed / max(len(files), 1)

    n_gt = len(gts)
    stats = coco_stats(preds, gts)
    mAP50, mAP50_95, precision, recall, n_cls = ultralytics_maps(stats)

    return {
        "weights": str(tflite),
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "precision": precision,
        "recall": recall,
        "device": "tflite-interpreter",
        "backend": "android-style decode [C,A] + letterbox + NMS + ultralytics ap_per_class",
        "n_images": len(files),
        "n_gt": n_gt,
        "n_pred": len(preds),
        "max_score_seen": max_score,
        "ms_per_image": ms,
        "conf_floor": conf_th,
        "nms_iou": iou_nms,
        "n_classes_with_gt": n_cls,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--tflite", type=Path, default=None)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-pt", action="store_true")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou-nms", type=float, default=0.70)
    args = p.parse_args()

    tflite = find_tflite(args.tflite)
    if tflite is None:
        print("[ERR] TFLite FP16 not found. Pass --tflite path-to-file.tflite")
        sys.exit(2)

    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / f"tflite_parity_subset_n{args.n}_seed{args.seed}"
    subset_yaml, n_used = build_subset_yaml(args.data, args.n, args.seed, dest)
    cfg = load_yaml(subset_yaml)
    root = dataset_root(subset_yaml, cfg)
    images_dir = test_images_dir(root, cfg)
    labels_dir = root / "labels" / "test"

    payload = {
        "imgsz": args.imgsz,
        "source_data": str(args.data),
        "subset_yaml": str(subset_yaml),
        "n_images": n_used,
        "seed": args.seed,
        "split": "test",
        "protocol": "same 2500-image subset; PyTorch ultralytics.val; TFLite Android-style decoder",
    }

    prev = args.output / "tflite_parity.json"
    pt_ok = False
    if args.skip_pt and prev.is_file():
        old = json.loads(prev.read_text(encoding="utf-8"))
        pt = old.get("pytorch") or {}
        if float(pt.get("mAP50") or 0) > 0:
            payload["pytorch"] = pt
            payload["pytorch"]["note"] = "reused from previous json (--skip-pt)"
            pt_ok = True
            print("[INFO] reuse PyTorch metrics from", prev)
    if not pt_ok:
        print(f"[INFO] val PyTorch {args.pt}")
        payload["pytorch"] = run_val(args.pt, subset_yaml, args.imgsz, args.device, args.max_det)

    print(f"[INFO] TFLite {tflite} size_mb={tflite.stat().st_size / 1e6:.2f}")
    print(f"[INFO] TFLite phone-decode {tflite}")
    payload["tflite"] = eval_tflite_phone_decode(
        tflite,
        images_dir,
        labels_dir,
        args.imgsz,
        args.conf,
        args.iou_nms,
        args.max_det,
    )
    if payload["tflite"].get("max_score_seen", 0) < 0.05:
        print("[ERR] max class score < 0.05 — layout still wrong, do not write zeros into the thesis")
        sys.exit(3)
    if payload["tflite"]["mAP50"] <= 0:
        print("[ERR] mAP50 is 0 after phone decode — refusing to overwrite json")
        sys.exit(3)

    pt = payload["pytorch"]
    tf = payload["tflite"]
    payload["delta"] = {
        "mAP50": tf["mAP50"] - float(pt["mAP50"]),
        "mAP50_95": tf["mAP50_95"] - float(pt["mAP50_95"]),
        "precision": tf["precision"] - float(pt["precision"]),
        "recall": tf["recall"] - float(pt["recall"]),
    }
    out = args.output / "tflite_parity.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[OK]", json.dumps({k: payload["tflite"][k] for k in ("mAP50", "mAP50_95", "precision", "recall", "ms_per_image", "n_pred", "max_score_seen")}, indent=2))
    print("[OK] wrote", out)
    print(
        "[LATEX] "
        f"mAP50={tf['mAP50']:.4f} mAP50-95={tf['mAP50_95']:.4f} "
        f"P={tf['precision']:.4f} R={tf['recall']:.4f} "
        f"ms={tf['ms_per_image']:.1f}"
    )


if __name__ == "__main__":
    main()
