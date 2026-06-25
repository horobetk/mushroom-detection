#!/usr/bin/env python3
"""
Lekki serwer odbierajacy pliki przez siec (uruchamiany na ZDALNYM komputerze).

Czesc systemu prostej autosynchronizacji kodu (laptop -> zdalny PC z GPU).
Nasluchuje na wskazanym porcie (domyslnie 9999) i zapisuje przychodzace pliki
do katalogu projektu, zachowujac ich wzgledna sciezke.

Protokol (bardzo prosty):
  POST /upload
    - naglowek 'X-Rel-Path': wzgledna sciezka pliku (URL-encoded), np. scripts%2Ftrain_mvp.py
    - cialo zadania: surowa zawartosc pliku (bajty)
  GET /ping  -> 'pong' (sprawdzenie dostepnosci serwera)

Bezpieczenstwo: zapis jest ograniczony WYLACZNIE do katalogu projektu
(odrzucamy sciezki absolutne oraz wyjscia poza katalog przez '..').

Nie wymaga uprawnien administratora ani zadnych zewnetrznych bibliotek
(uzywa wbudowanego modulu http.server).

Autor: Kiril Horobets
Politechnika Warszawska, 2026
"""

import argparse
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

# Wymuszenie UTF-8 na konsoli (sciezki z polskimi/cyrylickimi znakami)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Katalog glowny projektu = rodzic folderu scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Nazwa naglowka przenoszacego wzgledna sciezke pliku
REL_PATH_HEADER = "X-Rel-Path"


def safe_target(rel_path: str) -> Optional[Path]:
    """
    Zamienia wzgledna sciezke z zadania na bezpieczna sciezke absolutna
    wewnatrz katalogu projektu. Zwraca None, jesli sciezka jest niepoprawna
    lub probuje wyjsc poza projekt (ochrona przed path traversal).
    """
    rel = unquote(rel_path).replace("\\", "/").strip()
    if not rel or rel.startswith("/"):
        return None

    candidate = (PROJECT_ROOT / rel).resolve()
    try:
        # Rzuci ValueError, jesli candidate jest poza PROJECT_ROOT
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return candidate


class SyncHandler(BaseHTTPRequestHandler):
    """Obsluga zadan HTTP synchronizacji."""

    def _reply(self, code: int, message: str):
        body = message.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Health-check uzywany przez local_watcher do sprawdzenia polaczenia
        if self.path == "/ping":
            self._reply(200, "pong")
        else:
            self._reply(404, "Nieznany endpoint")

    def do_POST(self):
        if self.path != "/upload":
            self._reply(404, "Nieznany endpoint")
            return

        rel = self.headers.get(REL_PATH_HEADER, "")
        target = safe_target(rel)
        if target is None:
            self._reply(400, f"Niepoprawna lub niebezpieczna sciezka: {rel}")
            return

        # Odczyt calej zawartosci pliku z ciala zadania
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length) if length > 0 else b""

        # Zapis atomowy: najpierw plik tymczasowy, potem zamiana (os.replace)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(target) + ".tmp_sync")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, target)
        except OSError as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            self._reply(500, f"Blad zapisu: {exc}")
            return

        rel_display = target.relative_to(PROJECT_ROOT).as_posix()
        print(f"[OK] zapisano: {rel_display} ({len(data)} B)")
        self._reply(200, "OK")

    def log_message(self, *args):
        # Wyciszamy domyslny, halasliwy log http.server (mamy wlasne printy)
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Serwer odbierajacy pliki synchronizacji (zdalny PC)."
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Adres nasluchu (domyslnie 0.0.0.0 = wszystkie interfejsy)")
    parser.add_argument("--port", "-p", type=int, default=9999,
                        help="Port nasluchu (domyslnie 9999)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SyncHandler)

    print("=" * 60)
    print(" SERWER SYNCHRONIZACJI - START")
    print("=" * 60)
    print(f"  Katalog projektu: {PROJECT_ROOT}")
    print(f"  Nasluch:          http://{args.host}:{args.port}")
    print(f"  Endpointy:        POST /upload | GET /ping")
    print("  Zatrzymanie:      Ctrl+C")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nZatrzymywanie serwera...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
