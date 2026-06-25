#!/usr/bin/env python3
"""
Obserwator plikow + wysylka przez TCP (bez HTTP) - uruchamiany na LAPTOPIE.

Dziala z remote_receiver_tcp.py gdy firewalle VPN blokuja ruch HTTP.

Autor: Kiril Horobets, Politechnika Warszawska, 2026
"""

import argparse
import os
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

# Import wspolnej logiki filtrowania plikow z local_watcher.py
from local_watcher import (  # noqa: E402
    PROJECT_ROOT,
    is_relevant,
    iter_relevant_files,
)

SOCKET_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 1.0
SYNC_FILE_DELAY = 0.1  # mala przerwa miedzy plikami w jednym polaczeniu
PER_FILE_RETRIES = 5  # ile razy ponawiamy pojedynczy plik zanim go pominiemy
CHUNK_SIZE = 2048  # maly fragment - VPN/IDS nie resetuje malych porcji


def _recv_line(conn: socket.socket, limit: int = 64) -> bytes:
    buf = b""
    while b"\n" not in buf and len(buf) < limit:
        chunk = conn.recv(1)
        if not chunk:
            break
        buf += chunk
    return buf


def tcp_ping(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT) as conn:
            conn.sendall(b"SYNC_PING\n")
            resp = conn.recv(64)
            return resp.startswith(b"SYNC_PONG")
    except OSError:
        return False


def _send_put(conn: socket.socket, rel: str, data: bytes) -> bytes:
    """Wysyla plik chunkami (SYNC_PUTC) w istniejacym polaczeniu.

    Kazdy fragment jest maly i potwierdzany przez serwer (SYNC_ACK), wiec w
    sieci nigdy nie ma duzej porcji naraz - to omija resety IDS na wiekszych
    pakietach. Pusta odpowiedz = serwer zamknal polaczenie -> reconnect."""
    path_bytes = rel.encode("utf-8")
    conn.sendall(b"SYNC_PUTC\n")
    conn.sendall(struct.pack(">I", len(path_bytes)))
    conn.sendall(path_bytes)
    conn.sendall(struct.pack(">I", len(data)))

    view = memoryview(data)
    sent = 0
    n = len(data)
    while sent < n:
        chunk = view[sent:sent + CHUNK_SIZE]
        conn.sendall(struct.pack(">I", len(chunk)))
        conn.sendall(chunk)
        ack = _recv_line(conn, 16)
        if not ack:
            raise ConnectionError("serwer zamknal polaczenie (fragment)")
        if not ack.startswith(b"SYNC_ACK"):
            return ack  # blad serwera
        sent += len(chunk)

    resp = _recv_line(conn, 256)
    if not resp:
        raise ConnectionError("serwer zamknal polaczenie (brak odpowiedzi)")
    return resp


def tcp_send_file(host: str, port: int, path: Path, retries: int = MAX_RETRIES) -> bool:
    """Wysyla pojedynczy plik (nowe polaczenie) - uzywane przez watchdog/polling."""
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return False

    try:
        data = path.read_bytes()
    except (FileNotFoundError, PermissionError):
        return False

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT) as conn:
                resp = _send_put(conn, rel, data)
                conn.sendall(b"SYNC_BYE\n")

            if resp.startswith(b"SYNC_OK"):
                suffix = f" (proba {attempt})" if attempt > 1 else ""
                print(f"[->] wyslano: {rel} ({len(data)} B){suffix}")
                return True

            msg = resp.decode("utf-8", errors="replace").strip()
            print(f"[BLAD] nie wyslano {rel}: {msg}")
            return False

        except OSError as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(RETRY_DELAY * attempt)
            continue

    print(f"[BLAD] nie wyslano {rel}: {last_exc}")
    return False


def _prepare_entries():
    entries = []
    for path in iter_relevant_files():
        try:
            rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
            data = path.read_bytes()
        except (ValueError, OSError):
            continue
        entries.append((rel, data))
    return entries


def initial_sync_tcp(host: str, port: int):
    """Pierwsza synchronizacja przez JEDNO polaczenie TCP.

    Wszystkie pliki ida jednym strumieniem - duzo mniej handshake'ow niz
    polaczenie-na-plik, wiec VPN/IDS rzadziej resetuje ruch. Gdy mimo to
    polaczenie padnie, wznawiamy od tego samego pliku (bez gubienia)."""
    print("Pelna synchronizacja poczatkowa (TCP, jedno polaczenie)...")
    entries = _prepare_entries()
    total = len(entries)
    idx = 0
    ok = 0
    failed = []
    file_fail = 0  # ile razy z rzedu padl AKTUALNY plik

    while idx < total:
        try:
            with socket.create_connection((host, port), timeout=SOCKET_TIMEOUT) as conn:
                while idx < total:
                    rel, data = entries[idx]
                    resp = _send_put(conn, rel, data)
                    if resp.startswith(b"SYNC_OK"):
                        ok += 1
                        print(f"[->] wyslano: {rel} ({len(data)} B)")
                    else:
                        msg = resp.decode("utf-8", errors="replace").strip()
                        print(f"[BLAD] nie wyslano {rel}: {msg}")
                        failed.append(rel)
                    idx += 1
                    file_fail = 0
                    if idx < total:
                        time.sleep(SYNC_FILE_DELAY)
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

    print(f"\nZsynchronizowano {ok}/{total} plikow.")
    if failed:
        print(f"[UWAGA] Nie udalo sie wyslac {len(failed)} plikow:")
        for name in failed:
            print(f"         - {name}")
        print("         Zapisz ponownie plik w Cursor lub uruchom watcher jeszcze raz.\n")
    else:
        print()


def run_polling_tcp(host: str, port: int, interval: float):
    mtimes = {}
    for path in iter_relevant_files():
        try:
            mtimes[path] = path.stat().st_mtime
        except OSError:
            pass

    print(f"Tryb: polling TCP (co {interval}s). Ctrl+C aby zatrzymac.\n")
    try:
        while True:
            time.sleep(interval)
            for path in iter_relevant_files():
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtimes.get(path) != mtime:
                    mtimes[path] = mtime
                    tcp_send_file(host, port, path)
    except KeyboardInterrupt:
        pass


def run_watchdog_tcp(host: str, port: int, debounce: float):
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    last_sent = {}

    class Handler(FileSystemEventHandler):
        def _maybe_send(self, path_str: str):
            path = Path(path_str)
            if not is_relevant(path):
                return
            now = time.time()
            if now - last_sent.get(path_str, 0.0) < debounce:
                return
            last_sent[path_str] = now
            tcp_send_file(host, port, path)

        def on_modified(self, event):
            if not event.is_directory:
                self._maybe_send(event.src_path)

        def on_created(self, event):
            if not event.is_directory:
                self._maybe_send(event.src_path)

        def on_moved(self, event):
            if not event.is_directory:
                self._maybe_send(event.dest_path)

    observer = Observer()
    observer.schedule(Handler(), str(PROJECT_ROOT), recursive=True)
    observer.start()
    print("Tryb: watchdog TCP (natychmiastowy). Ctrl+C aby zatrzymac.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main():
    parser = argparse.ArgumentParser(description="Obserwator plikow - sync TCP (bez HTTP).")
    parser.add_argument("--host", "-H", required=True, help="IP zdalnego PC")
    parser.add_argument("--port", "-p", type=int, default=5500, help="Port TCP (domyslnie 5500)")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--debounce", type=float, default=0.5)
    parser.add_argument("--no-initial", action="store_true")
    parser.add_argument("--poll", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(" OBSERWATOR TCP - START")
    print("=" * 60)
    print(f"  Projekt:  {PROJECT_ROOT}")
    print(f"  Serwer:   tcp://{args.host}:{args.port}")
    print("=" * 60)

    if not tcp_ping(args.host, args.port):
        print(f"\n[OSTRZEZENIE] Brak odpowiedzi SYNC_PONG z tcp://{args.host}:{args.port}")
        print("              Uruchom remote_receiver_tcp.py na zdalnym PC.\n")
    else:
        print("\nPolaczenie TCP OK (SYNC_PONG).\n")

    if not args.no_initial:
        initial_sync_tcp(args.host, args.port)

    use_watchdog = not args.poll
    if use_watchdog:
        try:
            import watchdog  # noqa: F401
        except ImportError:
            print("[INFO] Brak watchdog - tryb polling.")
            use_watchdog = False

    if use_watchdog:
        run_watchdog_tcp(args.host, args.port, args.debounce)
    else:
        run_polling_tcp(args.host, args.port, args.interval)

    print("\nObserwator zatrzymany.")


if __name__ == "__main__":
    main()
