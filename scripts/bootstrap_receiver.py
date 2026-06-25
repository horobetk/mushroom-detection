#!/usr/bin/env python3
"""
Jednorazowy bootstrap: wgranie NOWEGO remote_receiver_tcp.py na zdalny PC,
gdy dziala tam jeszcze STARA wersja serwera (obslugujaca tylko SYNC_PUT).

Problem: VPN uczelni resetuje polaczenia niosace wiekszy payload (>~3 KB), wiec
nowego (wiekszego) receivera nie da sie wyslac w calosci. To narzedzie dzieli
go na male czesci (<2 KB), wysyla je STARYM protokolem SYNC_PUT (kazda osobno,
male = przechodzi), a nastepnie wgrywa malenki skrypt skladajacy.

Krok po kroku:
  1) Na zdalnym PC dziala stary serwer:  run_receiver_tcp.bat
  2) Na laptopie:  python scripts\\bootstrap_receiver.py --host 10.44.25.85
  3) Na zdalnym PC:  zatrzymaj serwer (Ctrl+C) i uruchom skladanie:
        python scripts\\.boot\\_assemble.py
  4) Na zdalnym PC ponownie:  run_receiver_tcp.bat   (juz NOWA wersja)

Od tego momentu local_watcher_tcp.py i pull_results_tcp.py uzywaja chunkow
(SYNC_PUTC / SYNC_GETC) i przechodza niezaleznie od rozmiaru pliku.

Autor: Kiril Horobets, Politechnika Warszawska, 2026
"""

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECEIVER = PROJECT_ROOT / "scripts" / "remote_receiver_tcp.py"
PART_SIZE = 1800  # < ~3 KB progu resetow VPN
SOCKET_TIMEOUT = 30
RETRIES = 6

# Maly skrypt skladajacy (uruchamiany na zdalnym PC). Sam jest < 2 KB.
ASSEMBLER = '''import glob, os
d = os.path.dirname(os.path.abspath(__file__))
parts = sorted(glob.glob(os.path.join(d, "remote_receiver_tcp.py.part*")))
if not parts:
    raise SystemExit("Brak czesci .part* w " + d)
data = b"".join(open(p, "rb").read() for p in parts)
out = os.path.join(os.path.dirname(d), "remote_receiver_tcp.py")
with open(out, "wb") as f:
    f.write(data)
print("Zlozono:", out)
print("Rozmiar:", len(data), "B z", len(parts), "czesci")
print("Teraz uruchom ponownie: run_receiver_tcp.bat")
'''


def send_put(host: str, port: int, rel: str, data: bytes) -> bool:
    """Wysyla maly plik STARYM protokolem SYNC_PUT (jedna komenda na polaczenie)."""
    pb = rel.encode("utf-8")
    for attempt in range(1, RETRIES + 1):
        try:
            with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT) as conn:
                conn.sendall(b"SYNC_PUT\n")
                conn.sendall(struct.pack(">I", len(pb)))
                conn.sendall(pb)
                conn.sendall(struct.pack(">I", len(data)))
                if data:
                    conn.sendall(data)
                resp = conn.recv(256)
            if resp.startswith(b"SYNC_OK"):
                print(f"[->] {rel} ({len(data)} B)" + (f" (proba {attempt})" if attempt > 1 else ""))
                return True
            print(f"[BLAD] {rel}: {resp.decode('utf-8', 'replace').strip()}")
            return False
        except OSError as exc:
            if attempt < RETRIES:
                time.sleep(1.0 * min(attempt, 3))
                continue
            print(f"[BLAD] {rel}: {exc}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Bootstrap nowego receivera przez stary serwer.")
    parser.add_argument("--host", "-H", required=True)
    parser.add_argument("--port", "-p", type=int, default=5500)
    args = parser.parse_args()

    if not RECEIVER.is_file():
        print(f"[BLAD] Nie znaleziono {RECEIVER}")
        return 1

    blob = RECEIVER.read_bytes()
    parts = [blob[i:i + PART_SIZE] for i in range(0, len(blob), PART_SIZE)] or [b""]

    print("=" * 60)
    print(" BOOTSTRAP RECEIVERA (przez stary serwer SYNC_PUT)")
    print("=" * 60)
    print(f"  Serwer:  tcp://{args.host}:{args.port}")
    print(f"  Plik:    remote_receiver_tcp.py ({len(blob)} B)")
    print(f"  Czesci:  {len(parts)} x do {PART_SIZE} B")
    print("=" * 60)

    ok = True
    for i, part in enumerate(parts):
        rel = f"scripts/.boot/remote_receiver_tcp.py.part{i:03d}"
        if not send_put(args.host, args.port, rel, part):
            ok = False
        time.sleep(0.2)

    if not send_put(args.host, args.port, "scripts/.boot/_assemble.py", ASSEMBLER.encode("utf-8")):
        ok = False

    print("=" * 60)
    if ok:
        print("Wszystkie czesci wyslane.")
        print("Teraz na ZDALNYM PC:")
        print("  1) zatrzymaj stary serwer (Ctrl+C)")
        print("  2) python scripts\\.boot\\_assemble.py")
        print("  3) run_receiver_tcp.bat   (juz nowa wersja z chunkami)")
        return 0
    print("[UWAGA] Nie wszystkie czesci doszly - uruchom skrypt ponownie.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
