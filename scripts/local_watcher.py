#!/usr/bin/env python3
"""
Lekki obserwator zmian plikow (uruchamiany na LOKALNYM laptopie).

Czesc systemu prostej autosynchronizacji kodu (laptop -> zdalny PC z GPU).
Sledzi katalog projektu i po kazdym zapisie pliku .py / .yaml natychmiast
wysyla jego tresc zadaniem POST na serwer remote_receiver.py
(http://<REMOTE_IP>:9999/upload).

Tryby pracy:
  - watchdog (jesli zainstalowany)  -> reakcja natychmiastowa na zdarzenia,
  - polling (fallback, czysty stdlib) -> sprawdzanie mtime co `--interval` sekund.

Pomijane katalogi: .git, .venv, venv, runs, datasets, weights, __pycache__,
oraz data/external i data/frames (zeby nie wysylac danych/wag).

Nie wymaga uprawnien administratora. Do wysylki uzywa wbudowanego urllib
(biblioteka 'requests' nie jest wymagana).

Autor: Kiril Horobets
Politechnika Warszawska, 2026
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

# Wymuszenie UTF-8 na konsoli (sciezki z polskimi/cyrylickimi znakami)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Katalog glowny projektu = rodzic folderu scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sledzone rozszerzenia plikow
WATCH_EXTS = {".py", ".yaml", ".yml"}

# Katalogi calkowicie pomijane
IGNORE_DIRS = {
    ".git", ".venv", "venv", "runs", "datasets", "weights",
    "__pycache__", ".idea", ".vscode", "node_modules",
}

# Nazwa naglowka przenoszacego wzgledna sciezke pliku (zgodna z remote_receiver.py)
REL_PATH_HEADER = "X-Rel-Path"


def iter_relevant_files():
    """
    Generator: zwraca wszystkie sledzone pliki w projekcie, pomijajac
    ciezkie/niepotrzebne katalogi (przyciecie drzewa w os.walk).
    """
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Przytnij ignorowane katalogi w miejscu (modyfikacja 'dirs' wplywa na walk)
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        rel_root = Path(root).resolve().relative_to(PROJECT_ROOT)
        # Dodatkowo pomijamy data/external i data/frames
        if rel_root.as_posix() == "data":
            dirs[:] = [d for d in dirs if d not in ("external", "frames")]

        for fname in files:
            path = Path(root) / fname
            if path.suffix.lower() in WATCH_EXTS:
                yield path


def is_relevant(path: Path) -> bool:
    """Sprawdza, czy dany plik powinien byc synchronizowany."""
    if path.suffix.lower() not in WATCH_EXTS:
        return False
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    parts = rel.parts
    if any(p in IGNORE_DIRS for p in parts):
        return False
    if len(parts) >= 2 and parts[0] == "data" and parts[1] in ("external", "frames"):
        return False
    return True


def send_file(path: Path, base_url: str):
    """Wysyla pojedynczy plik na serwer zdalny przez POST /upload."""
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return

    try:
        data = path.read_bytes()
    except (FileNotFoundError, PermissionError):
        # Plik moze byc chwilowo zablokowany w trakcie zapisu - pomijamy
        return

    url = base_url.rstrip("/") + "/upload"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header(REL_PATH_HEADER, quote(rel))  # URL-encode (bezpieczne dla naglowka)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        print(f"[->] wyslano: {rel} ({len(data)} B)")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"[BLAD] nie wyslano {rel}: {exc}")


def check_server(base_url: str) -> bool:
    """Sprawdza dostepnosc serwera (GET /ping)."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/ping", timeout=5) as resp:
            return resp.read().strip() == b"pong"
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def initial_sync(base_url: str):
    """Jednorazowa pelna synchronizacja wszystkich sledzonych plikow na starcie."""
    print("Pelna synchronizacja poczatkowa...")
    count = 0
    for path in iter_relevant_files():
        send_file(path, base_url)
        count += 1
    print(f"Zsynchronizowano {count} plikow.\n")


def run_watchdog(base_url: str, debounce: float):
    """Tryb watchdog - natychmiastowa reakcja na zdarzenia systemu plikow."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    last_sent = {}

    class Handler(FileSystemEventHandler):
        def _maybe_send(self, path_str: str):
            path = Path(path_str)
            if not is_relevant(path):
                return
            # Debounce: edytory potrafia generowac kilka zdarzen na jeden zapis
            now = time.time()
            if now - last_sent.get(path_str, 0.0) < debounce:
                return
            last_sent[path_str] = now
            send_file(path, base_url)

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
    print("Tryb: watchdog (natychmiastowy). Ctrl+C aby zatrzymac.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def run_polling(base_url: str, interval: float):
    """Tryb fallback - cykliczne sprawdzanie czasow modyfikacji (mtime)."""
    mtimes = {}
    # Inicjalne odczytanie mtime (bez wysylania - to robi initial_sync)
    for path in iter_relevant_files():
        try:
            mtimes[path] = path.stat().st_mtime
        except OSError:
            pass

    print(f"Tryb: polling (co {interval}s). Ctrl+C aby zatrzymac.\n")
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
                    send_file(path, base_url)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Obserwator zmian plikow - autosynchronizacja na zdalny PC."
    )
    parser.add_argument("--host", "-H", required=True,
                        help="Adres IP zdalnego komputera (REMOTE_IP)")
    parser.add_argument("--port", "-p", type=int, default=8080,
                        help="Port serwera remote_receiver.py (domyslnie 8080)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Interwal pollingu w sekundach (domyslnie 1.0)")
    parser.add_argument("--debounce", type=float, default=0.5,
                        help="Debounce zdarzen watchdog w sekundach (domyslnie 0.5)")
    parser.add_argument("--no-initial", action="store_true",
                        help="Pomin pelna synchronizacje poczatkowa")
    parser.add_argument("--poll", action="store_true",
                        help="Wymus tryb polling (nawet jesli watchdog jest dostepny)")

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print("=" * 60)
    print(" OBSERWATOR SYNCHRONIZACJI - START")
    print("=" * 60)
    print(f"  Katalog projektu: {PROJECT_ROOT}")
    print(f"  Serwer zdalny:    {base_url}")
    print("=" * 60)

    # Sprawdzenie polaczenia z serwerem
    if not check_server(base_url):
        print(f"\n[OSTRZEZENIE] Serwer {base_url} nie odpowiada na /ping.")
        print("              Uruchom remote_receiver.py na zdalnym PC i sprawdz VPN/firewall.")
        print("              Kontynuuje mimo to (wysylki beda probowane na biezaco).\n")

    # Pelna synchronizacja na starcie (chyba ze wylaczona)
    if not args.no_initial:
        initial_sync(base_url)

    # Wybor trybu pracy
    use_watchdog = not args.poll
    if use_watchdog:
        try:
            import watchdog  # noqa: F401
        except ImportError:
            print("[INFO] Biblioteka 'watchdog' niedostepna - przelaczam na tryb polling.")
            print("       (Mozesz ja zainstalowac: pip install watchdog)\n")
            use_watchdog = False

    if use_watchdog:
        run_watchdog(base_url, args.debounce)
    else:
        run_polling(base_url, args.interval)

    print("\nObserwator zatrzymany.")


if __name__ == "__main__":
    main()
