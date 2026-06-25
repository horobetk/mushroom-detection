#!/usr/bin/env python3
"""
Pobieranie wynikow ze ZDALNEGO PC na LAPTOP przez TCP (bez HTTP).

Odwrotnosc local_watcher_tcp.py - laczy sie z remote_receiver_tcp.py i sciaga
pliki (wytrenowane wagi, wykresy, model TFLite). Uzywa tego samego prostego
protokolu binarnego, ktory dziala przez niestabilny VPN uczelni.

Przyklad:
  python scripts\\pull_results_tcp.py --host 10.44.25.85 --port 5500
  python scripts\\pull_results_tcp.py --host 10.44.25.85 --paths runs android/app/models

Autor: Kiril Horobets, Politechnika Warszawska, 2026
"""

import argparse
import json
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
SOCKET_TIMEOUT = 60
RETRY_DELAY = 1.0
PER_FILE_RETRIES = 5
NOT_FOUND = 0xFFFFFFFF
MAX_CHUNK = 1 * 1024 * 1024  # gorny limit pojedynczego fragmentu (ochrona)

DEFAULT_PATHS = ["runs/detect", "android/app/models"]


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("polaczenie zerwane podczas odbioru")
        buf += chunk
    return buf


def tcp_ping(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT) as conn:
            conn.sendall(b"SYNC_PING\n")
            resp = conn.recv(64)
            conn.sendall(b"SYNC_BYE\n")
            return resp.startswith(b"SYNC_PONG")
    except OSError:
        return False


def _list_remote(conn: socket.socket, prefix: str):
    pb = prefix.encode("utf-8")
    conn.sendall(b"SYNC_LIST\n")
    conn.sendall(struct.pack(">I", len(pb)))
    conn.sendall(pb)
    blob_len = struct.unpack(">I", recv_exact(conn, 4))[0]
    blob = recv_exact(conn, blob_len) if blob_len else b"[]"
    return json.loads(blob.decode("utf-8"))


def _get_remote(conn: socket.socket, rel: str) -> bytes:
    """Pobiera plik chunkami (SYNC_GETC) - male, potwierdzane fragmenty."""
    pb = rel.encode("utf-8")
    conn.sendall(b"SYNC_GETC\n")
    conn.sendall(struct.pack(">I", len(pb)))
    conn.sendall(pb)
    total = struct.unpack(">I", recv_exact(conn, 4))[0]
    if total == NOT_FOUND:
        raise FileNotFoundError(rel)
    buf = bytearray()
    while len(buf) < total:
        clen = struct.unpack(">I", recv_exact(conn, 4))[0]
        if clen == 0 or clen > MAX_CHUNK or len(buf) + clen > total:
            raise ConnectionError("niepoprawny rozmiar fragmentu")
        buf += recv_exact(conn, clen)
        conn.sendall(b"SYNC_ACK\n")
    return bytes(buf)


def _write_local(rel: str, data: bytes) -> Path:
    target = (PROJECT_ROOT / rel).resolve()
    target.relative_to(PROJECT_ROOT)  # ochrona przed path traversal
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(target) + ".tmp_pull")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def pull(host: str, port: int, prefixes):
    print("=" * 60)
    print(" POBIERANIE WYNIKOW (TCP) - START")
    print("=" * 60)
    print(f"  Serwer:   tcp://{host}:{port}")
    print(f"  Sciezki:  {', '.join(prefixes)}")
    print("=" * 60)

    if not tcp_ping(host, port):
        print(f"\n[BLAD] Brak SYNC_PONG z tcp://{host}:{port}")
        print("       Uruchom remote_receiver_tcp.py na zdalnym PC.\n")
        return 1

    print("\nPolaczenie TCP OK (SYNC_PONG).\n")

    # Zbierz liste plikow ze wszystkich prefiksow (jedno polaczenie).
    entries = []
    seen = set()
    try:
        with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT) as conn:
            for prefix in prefixes:
                try:
                    items = _list_remote(conn, prefix.replace("\\", "/"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[UWAGA] nie udalo sie wylistowac '{prefix}': {exc}")
                    continue
                for it in items:
                    rel = it.get("path")
                    if rel and rel not in seen:
                        seen.add(rel)
                        entries.append((rel, int(it.get("size", 0))))
            conn.sendall(b"SYNC_BYE\n")
    except OSError as exc:
        print(f"[BLAD] nie udalo sie pobrac listy plikow: {exc}")
        return 1

    if not entries:
        print("[INFO] Brak plikow do pobrania (jeszcze nie ma wynikow?).")
        return 0

    total = len(entries)
    total_bytes = sum(sz for _, sz in entries)
    print(f"Do pobrania: {total} plikow ({total_bytes / 1024 / 1024:.1f} MB)\n")

    idx = 0
    ok = 0
    failed = []
    file_fail = 0  # ile razy z rzedu padl AKTUALNY plik

    while idx < total:
        try:
            with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT) as conn:
                while idx < total:
                    rel, _ = entries[idx]
                    try:
                        data = _get_remote(conn, rel)
                    except FileNotFoundError:
                        print(f"[BRAK] {rel} - pominieto")
                        failed.append(rel)
                        idx += 1
                        file_fail = 0
                        continue
                    _write_local(rel, data)
                    ok += 1
                    print(f"[<-] pobrano: {rel} ({len(data)} B)")
                    idx += 1
                    file_fail = 0
                conn.sendall(b"SYNC_BYE\n")
        except OSError as exc:
            cur = entries[idx][0] if idx < total else "?"
            file_fail += 1
            if file_fail >= PER_FILE_RETRIES:
                print(f"[POMINIETO] {cur} - {file_fail} nieudanych prob, ide dalej: {exc}")
                failed.append(cur)
                idx += 1
                file_fail = 0
                continue
            print(f"[VPN] zerwano polaczenie, wznawiam od '{cur}' (proba {file_fail}/{PER_FILE_RETRIES}): {exc}")
            time.sleep(RETRY_DELAY * min(file_fail, 3))

    print(f"\nPobrano {ok}/{total} plikow.")
    if failed:
        print(f"[UWAGA] Nie pobrano {len(failed)} plikow:")
        for name in failed:
            print(f"         - {name}")
        return 1
    print("Gotowe.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Pobieranie wynikow ze zdalnego PC (TCP, bez HTTP).")
    parser.add_argument("--host", "-H", required=True, help="IP zdalnego PC")
    parser.add_argument("--port", "-p", type=int, default=5500, help="Port TCP (domyslnie 5500)")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_PATHS,
        help="Prefiksy/katalogi do pobrania (domyslnie: runs/detect android/app/models)",
    )
    args = parser.parse_args()
    sys.exit(pull(args.host, args.port, args.paths))


if __name__ == "__main__":
    main()
