"""Compute end-to-end latency statistics from on-device profiler CSV files."""
import csv
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "dyploma" / "rysunki" / "wyniki"


def pct(values, p):
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_inf():
    rows = []
    with open(HERE / "inference_profile.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("scene_locked",) else v) for k, v in r.items()})
    return rows


def summ(rows, label):
    def col(name):
        return [r[name] for r in rows]
    print(f"\n=== {label}  (n={len(rows)}) ===")
    print(f"  battery_c: {min(col('battery_c')):.1f}..{max(col('battery_c')):.1f}")
    for name in ("detect_ms", "convert_ms", "letterbox_ms", "post_ms", "sensor_to_app_ms", "e2e_ms"):
        v = col(name)
        print(f"  {name:16s} mean={st.mean(v):7.1f}  p95={pct(v,95):7.1f}  max={max(v):7.1f}")


rows = load_inf()
print("total frames:", len(rows))
# split by battery temperature buckets
cold = [r for r in rows if r["battery_c"] <= 30]
hot = [r for r in rows if r["battery_c"] >= 40.5]
mid = [r for r in rows if 30 < r["battery_c"] < 40.5]
summ(cold[:150], "COLD first150")
summ(hot[:150], "HOT first150")
summ(hot, "HOT all")
summ(mid, "MID (post battery drop)")

# response times
resp = []
with open(HERE / "response_profile.csv", newline="") as f:
    for r in csv.DictReader(f):
        resp.append({"status": r["status"], "lat": float(r["latency_ms"]), "species": r["species"]})
lat = [r["lat"] for r in resp]
print(f"\n=== RESPONSE display  (n={len(resp)}) ===")
print(f"  mean={st.mean(lat):.1f}  p50={pct(lat,50):.1f}  p95={pct(lat,95):.1f}  max={max(lat):.1f}")
print(f"  >1000ms: {sum(1 for x in lat if x>1000)}")
green = [r for r in resp if r["status"] == "edible"]
print(f"  green/edible assertions: {len(green)}")
print(f"  statuses: {sorted(set(r['status'] for r in resp))}")
