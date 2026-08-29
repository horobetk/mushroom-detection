#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from tkinter import LEFT, StringVar, Tk, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    print("Need Pillow:  pip install pillow")
    sys.exit(1)

FIELDS = (
    "n_visible_fruitbodies",
    "n_boxes_ok",
    "n_fn_missed",
    "n_fp_extra",
    "notes",
)
INT_FIELDS = FIELDS[:4]
BOX_W = 3
FORM_W = 400


def repo_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "thesis_results" / "fn_audit_sheet.csv"


def label_path_for(image_path: Path) -> Path:
    p = image_path
    parts = list(p.parts)
    if "images" in parts:
        i = parts.index("images")
        parts[i] = "labels"
        out = Path(*parts).with_suffix(".txt")
        if out.is_file():
            return out
    return p.parent.parent.parent / "labels" / "test" / (p.stem + ".txt")


def read_boxes(label_file: Path):
    boxes = []
    if not label_file.is_file():
        return boxes
    with label_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            bits = line.split()
            if len(bits) < 5:
                continue
            cid = int(float(bits[0]))
            cx, cy, w, h = map(float, bits[1:5])
            boxes.append((cid, cx, cy, w, h))
    return boxes


def draw_overlay(img: Image.Image, boxes: list) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    try:
        font = ImageFont.truetype("arial.ttf", max(14, W // 55))
    except OSError:
        font = ImageFont.load_default()
    for i, (_cid, cx, cy, w, h) in enumerate(boxes, start=1):
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        draw.rectangle([x1, y1, x2, y2], outline="#ff3333", width=BOX_W)
        tag = f"{i}"
        bb = draw.textbbox((0, 0), tag, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + th + 4], fill="#ff3333")
        draw.text((x1 + 3, y1 + 1), tag, fill="white", font=font)
    return out


class Auditor:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            self.fieldnames = list(csv.DictReader(fh).fieldnames or [])
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            self.rows = list(csv.DictReader(fh))
        if not self.rows:
            raise SystemExit(f"empty sheet: {csv_path}")
        self.i = 0
        for k, row in enumerate(self.rows):
            if not (row.get("n_visible_fruitbodies") or "").strip():
                self.i = k
                break

        self.root = Tk()
        self.root.title("FN audit — GroundingDINO")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.img_max_w = max(480, sw - FORM_W - 80)
        self.img_max_h = max(480, sh - 100)
        self.root.geometry(f"{min(sw - 30, self.img_max_w + FORM_W + 40)}x{min(sh - 60, self.img_max_h)}")
        self.root.minsize(900, 600)

        self.photo = None
        self.vars = {k: StringVar() for k in FIELDS}

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0, minsize=FORM_W)
        self.root.rowconfigure(0, weight=1)
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        left = ttk.Frame(self.root, padding=6)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self.status = StringVar()
        ttk.Label(left, textvariable=self.status, font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self.img_label = ttk.Label(left, anchor="center")
        self.img_label.grid(row=1, column=0, sticky="nsew")

        right = ttk.Frame(self.root, padding=10, width=FORM_W)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)
        right.pack_propagate(False)

        ttk.Label(right, text="Что делать", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        rules = (
            "Красные рамки уже стоят из файла разметки DINO.\n"
            "Глазами посчитай грибы на фото.\n\n"
            "1  Сколько грибов ВИДНО?\n"
            "    (куча отдельных шляпок = несколько;\n"
            "     один сросшийся куст можно считать как 1)\n"
            "2  Сколько красных рамок реально на грибе?\n"
            "3  FN: грибы БЕЗ рамки (DINO пропустил)\n"
            "4  FP: красные рамки НЕ на грибе (лишние)\n\n"
            "Проверка: поле2 + поле3 ≈ поле1.\n"
            "Enter — записать и следующее фото."
        )
        ttk.Label(right, text=rules, justify="left", wraplength=FORM_W - 24).pack(anchor="w", pady=(0, 10))

        labels = [
            ("n_visible_fruitbodies", "1. Видно грибов"),
            ("n_boxes_ok", "2. Рамок на грибе"),
            ("n_fn_missed", "3. Пропуски (FN)"),
            ("n_fp_extra", "4. Лишние рамки (FP)"),
        ]
        for key, lab in labels:
            row = ttk.Frame(right)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=lab, width=24).pack(side=LEFT)
            ent = ttk.Entry(row, textvariable=self.vars[key], width=8, font=("Segoe UI", 12))
            ent.pack(side=LEFT)
            if key == "n_visible_fruitbodies":
                self.first_entry = ent

        ttk.Label(right, text="Заметка (необязательно)").pack(anchor="w", pady=(8, 0))
        self.notes = ScrolledText(right, height=4, wrap="word", width=36)
        self.notes.pack(fill="x")

        ttk.Button(right, text="Сохранить + дальше  (Enter)", command=self.save_next).pack(fill="x", pady=(12, 4))
        ttk.Button(right, text="Назад", command=self.prev).pack(fill="x", pady=2)
        ttk.Button(right, text="Следующая пустая  (F2)", command=self.jump_empty).pack(fill="x", pady=2)
        ttk.Button(right, text="Только сохранить", command=self.save_only).pack(fill="x", pady=2)

        self.filled_var = StringVar()
        ttk.Label(right, textvariable=self.filled_var, foreground="#333").pack(anchor="w", pady=8)

        self.root.bind("<Return>", lambda e: self.save_next())
        self.root.bind("<F2>", lambda e: self.jump_empty())
        self.root.bind("<Prior>", lambda e: self.prev())
        self.root.bind("<Next>", lambda e: self.goto(self.i + 1))

        self.show()
        self.root.mainloop()

    def filled(self) -> int:
        return sum(1 for row in self.rows if (row.get("n_visible_fruitbodies") or "").strip())

    def load_vars(self) -> None:
        row = self.rows[self.i]
        for k in INT_FIELDS:
            self.vars[k].set(row.get(k, "") or "")
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", row.get("notes", "") or "")

    def harvest_vars(self) -> bool:
        row = self.rows[self.i]
        vals = {k: self.vars[k].get().strip() for k in INT_FIELDS}
        for k, v in vals.items():
            if v == "":
                messagebox.showwarning("Пусто", f"Заполни: {k}")
                return False
            try:
                n = int(v)
            except ValueError:
                messagebox.showwarning("Не число", k)
                return False
            if n < 0:
                messagebox.showwarning("Отрицательное", k)
                return False
            row[k] = str(n)
        row["notes"] = self.notes.get("1.0", "end").strip()
        vis = int(row["n_visible_fruitbodies"])
        ok = int(row["n_boxes_ok"])
        fn = int(row["n_fn_missed"])
        if abs((ok + fn) - vis) > 2:
            if not messagebox.askyesno("Проверь", f"рамки_ок + FN = {ok + fn}, видно = {vis}. Сохранить?"):
                return False
        return True

    def write_csv(self) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fieldnames)
            w.writeheader()
            w.writerows(self.rows)

    def save_only(self) -> None:
        if self.harvest_vars():
            self.write_csv()
            self.filled_var.set(f"Заполнено {self.filled()} / {len(self.rows)}")

    def save_next(self) -> None:
        if not self.harvest_vars():
            return
        self.write_csv()
        if self.i + 1 >= len(self.rows):
            messagebox.showinfo("Готово", f"{self.filled()}/{len(self.rows)}")
            return
        self.goto(self.i + 1)

    def prev(self) -> None:
        self.goto(self.i - 1)

    def jump_empty(self) -> None:
        for k in range(len(self.rows)):
            j = (self.i + 1 + k) % len(self.rows)
            if not (self.rows[j].get("n_visible_fruitbodies") or "").strip():
                self.goto(j)
                return
        messagebox.showinfo("Готово", "Пустых строк нет.")

    def goto(self, idx: int) -> None:
        if 0 <= idx < len(self.rows):
            self.i = idx
            self.show()

    def show(self) -> None:
        row = self.rows[self.i]
        path = Path(row["image_path"])
        boxes = read_boxes(label_path_for(path))
        bucket = row.get("gt_bucket", "")
        self.status.set(f"{self.i + 1} / {len(self.rows)}   [{bucket}]   {path.name}")
        self.filled_var.set(f"Заполнено {self.filled()} / {len(self.rows)}   красных рамок на фото: {len(boxes)}")
        self.load_vars()
        if not path.is_file():
            self.img_label.configure(image="", text=f"Нет файла\n{path}")
            self.photo = None
            return
        vis = draw_overlay(Image.open(path), boxes)
        w, h = vis.size
        scale = min(self.img_max_w / w, self.img_max_h / h, 1.0)
        if scale < 1:
            vis = vis.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(vis)
        self.img_label.configure(image=self.photo, text="")
        self.first_entry.focus_set()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=repo_csv())
    args = p.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"missing {args.csv}")
    Auditor(args.csv)


if __name__ == "__main__":
    main()
