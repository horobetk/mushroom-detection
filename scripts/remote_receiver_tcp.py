#!/usr/bin/env python3
"""
Serwer TCP odbierajacy pliki (bez HTTP) - uruchamiany na ZDALNYM PC.

Universyteckie firewalle/IDS czesto przepuszczaja polaczenie TCP, ale resetuja
ruch rozpoznany jako HTTP miedzy roznymi podsieciami VPN. Ten serwer uzywa
prostego binarnego protokolu (bez naglowkow GET/POST/HTTP).

Protokol (jedno polaczenie moze przeniesc wiele plikow w petli):
  PING:  klient -> b"SYNC_PING\\n"     serwer -> b"SYNC_PONG\\n"
  UPLOAD: klient -> b"SYNC_PUT\\n"
          + 4B BE dlugosc_sciezki + sciezka UTF-8
          + 4B BE dlugosc_danych + bajty pliku
          serwer -> b"SYNC_OK\\n" lub b"SYNC_ERR:...\\n"
  LISTA:  klient -> b"SYNC_LIST\\n"
          + 4B BE dlugosc_prefiksu + prefiks UTF-8 (np. "runs")
          serwer -> 4B BE dlugosc_JSON + JSON [{"path":..,"size":..}, ...]
  POBRANIE: klient -> b"SYNC_GET\\n"
          + 4B BE dlugosc_sciezki + sciezka UTF-8
          serwer -> 4B BE dlugosc_danych + bajty pliku
                    (dlugosc 0xFFFFFFFF = plik nie istnieje)
  UPLOAD CHUNKOWANY (zalecany przez niestabilny VPN): klient -> b"SYNC_PUTC\\n"
          + 4B BE dlugosc_sciezki + sciezka UTF-8
          + 4B BE calkowity_rozmiar
          + powtarzane: 4B BE dlugosc_fragmentu + bajty fragmentu
            (po kazdym fragmencie serwer odsyla b"SYNC_ACK\\n")
          serwer na koniec -> b"SYNC_OK\\n"
  POBRANIE CHUNKOWANE: klient -> b"SYNC_GETC\\n"
          + 4B BE dlugosc_sciezki + sciezka UTF-8
          serwer -> 4B BE calkowity_rozmiar (0xFFFFFFFF = brak pliku)
          + powtarzane: 4B BE dlugosc_fragmentu + bajty fragmentu
            (po kazdym fragmencie klient odsyla b"SYNC_ACK\\n")
  KONIEC: klient -> b"SYNC_BYE\\n" lub zamkniecie polaczenia

Tryb chunkowany dzieli plik na male, potwierdzane fragmenty - dzieki temu w
sieci nigdy nie ma duzej porcji naraz, co omija resety IDS na wiekszych pakietach.

Po kazdej komendzie serwer czeka na kolejna w tym samym polaczeniu, dopoki
klient nie zamknie polaczenia. Dzieki temu cala pierwsza synchronizacja moze
isc przez JEDNO polaczenie TCP (mniej handshake'ow = mniej resetow VPN/IDS).

Autor: Kiril Horobets, Politechnika Warszawska, 2026
"""

import argparse
import json
import os
import socket
import socketserver
import struct
import sys
from pathlib import Path
from typing import Optional

NOT_FOUND = 0xFFFFFFFF

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_PATH = 512
MAX_FILE = 200 * 1024 * 1024  # 200 MB (wagi .pt moga byc duze)
MAX_CHUNK = 1 * 1024 * 1024  # 1 MB - gorny limit pojedynczego fragmentu
GET_CHUNK = 16 * 1024  # rozmiar fragmentu przy chunkowanym pobieraniu


def safe_target(rel_path: str) -> Optional[Path]:
    rel = rel_path.replace("\\", "/").strip()
    if not rel or rel.startswith("/"):
        return None
    candidate = (PROJECT_ROOT / rel).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return candidate


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Polaczenie zerwane przed odebraniem danych")
        buf += chunk
    return buf


def read_line(conn: socket.socket, limit: int = 256) -> bytes:
    buf = b""
    while b"\n" not in buf and len(buf) < limit:
        chunk = conn.recv(1)
        if not chunk:
            break
        buf += chunk
    return buf


class SyncTCPHandler(socketserver.BaseRequestHandler):
    def _handle_put(self, conn: socket.socket, peer: str):
        path_len = struct.unpack(">I", recv_exact(conn, 4))[0]
        if path_len == 0 or path_len > MAX_PATH:
            conn.sendall(b"SYNC_ERR:niepoprawna dlugosc sciezki\n")
            return

        rel = recv_exact(conn, path_len).decode("utf-8", errors="strict")
        data_len = struct.unpack(">I", recv_exact(conn, 4))[0]
        if data_len > MAX_FILE:
            conn.sendall(b"SYNC_ERR:plik za duzy\n")
            return

        data = recv_exact(conn, data_len) if data_len else b""
        target = safe_target(rel)
        if target is None:
            conn.sendall(b"SYNC_ERR:niebezpieczna sciezka\n")
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(target) + ".tmp_sync")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, target)

        print(f"[OK] zapisano: {rel} ({len(data)} B) <- {peer}")
        conn.sendall(b"SYNC_OK\n")

    def _handle_putc(self, conn: socket.socket, peer: str):
        path_len = struct.unpack(">I", recv_exact(conn, 4))[0]
        if path_len == 0 or path_len > MAX_PATH:
            conn.sendall(b"SYNC_ERR:niepoprawna dlugosc sciezki\n")
            return
        rel = recv_exact(conn, path_len).decode("utf-8", errors="strict")
        total = struct.unpack(">I", recv_exact(conn, 4))[0]
        if total > MAX_FILE:
            conn.sendall(b"SYNC_ERR:plik za duzy\n")
            return
        target = safe_target(rel)
        if target is None:
            conn.sendall(b"SYNC_ERR:niebezpieczna sciezka\n")
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(target) + ".tmp_sync")
        received = 0
        try:
            with open(tmp, "wb") as f:
                while received < total:
                    clen = struct.unpack(">I", recv_exact(conn, 4))[0]
                    if clen == 0 or clen > MAX_CHUNK or received + clen > total:
                        conn.sendall(b"SYNC_ERR:niepoprawny rozmiar fragmentu\n")
                        return
                    f.write(recv_exact(conn, clen))
                    received += clen
                    conn.sendall(b"SYNC_ACK\n")
            os.replace(tmp, target)
        except (ConnectionError, OSError):
            tmp.unlink(missing_ok=True)
            raise

        print(f"[OK] zapisano: {rel} ({received} B, chunk) <- {peer}")
        conn.sendall(b"SYNC_OK\n")

    def _handle_getc(self, conn: socket.socket, peer: str):
        path_len = struct.unpack(">I", recv_exact(conn, 4))[0]
        if path_len == 0 or path_len > MAX_PATH:
            conn.sendall(struct.pack(">I", NOT_FOUND))
            return
        rel = recv_exact(conn, path_len).decode("utf-8", errors="strict")
        target = safe_target(rel)
        if target is None or not target.is_file():
            conn.sendall(struct.pack(">I", NOT_FOUND))
            return
        size = target.stat().st_size
        conn.sendall(struct.pack(">I", size))
        sent = 0
        with open(target, "rb") as f:
            while sent < size:
                chunk = f.read(GET_CHUNK)
                if not chunk:
                    break
                conn.sendall(struct.pack(">I", len(chunk)))
                conn.sendall(chunk)
                ack = read_line(conn, 16)  # czekamy na SYNC_ACK od klienta
                if not ack.startswith(b"SYNC_ACK"):
                    return
                sent += len(chunk)
        print(f"[->] wyslano: {rel} ({sent} B, chunk) -> {peer}")

    def _handle_list(self, conn: socket.socket):
        prefix_len = struct.unpack(">I", recv_exact(conn, 4))[0]
        prefix = recv_exact(conn, prefix_len).decode("utf-8", errors="strict") if prefix_len else ""
        base = safe_target(prefix) if prefix else PROJECT_ROOT
        payload = []
        if base is not None and base.exists():
            candidates = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
            for p in candidates:
                try:
                    rel = p.resolve().relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    continue
                if p.suffix == ".tmp_sync":
                    continue
                payload.append({"path": rel, "size": p.stat().st_size})
        blob = json.dumps(payload).encode("utf-8")
        conn.sendall(struct.pack(">I", len(blob)))
        conn.sendall(blob)

    def _handle_get(self, conn: socket.socket, peer: str):
        path_len = struct.unpack(">I", recv_exact(conn, 4))[0]
        if path_len == 0 or path_len > MAX_PATH:
            conn.sendall(struct.pack(">I", NOT_FOUND))
            return
        rel = recv_exact(conn, path_len).decode("utf-8", errors="strict")
        target = safe_target(rel)
        if target is None or not target.is_file():
            conn.sendall(struct.pack(">I", NOT_FOUND))
            return
        data = target.read_bytes()
        conn.sendall(struct.pack(">I", len(data)))
        conn.sendall(data)
        print(f"[->] wyslano: {rel} ({len(data)} B) -> {peer}")

    def handle(self):
        conn: socket.socket = self.request
        conn.settimeout(120)
        peer = self.client_address[0]
        try:
            while True:
                line = read_line(conn)
                if not line:
                    break  # klient zamknal polaczenie
                if line == b"SYNC_PING\n":
                    conn.sendall(b"SYNC_PONG\n")
                    continue
                if line == b"SYNC_BYE\n":
                    break
                if line == b"SYNC_PUT\n":
                    self._handle_put(conn, peer)
                    continue
                if line == b"SYNC_PUTC\n":
                    self._handle_putc(conn, peer)
                    continue
                if line == b"SYNC_LIST\n":
                    self._handle_list(conn)
                    continue
                if line == b"SYNC_GET\n":
                    self._handle_get(conn, peer)
                    continue
                if line == b"SYNC_GETC\n":
                    self._handle_getc(conn, peer)
                    continue
                conn.sendall(b"SYNC_ERR:nieznana komenda\n")

        except (ConnectionError, OSError, struct.error, UnicodeDecodeError) as exc:
            try:
                conn.sendall(f"SYNC_ERR:{exc}\n".encode("utf-8", errors="replace"))
            except OSError:
                pass
        except Exception as exc:  # noqa: BLE001
            try:
                conn.sendall(f"SYNC_ERR:{exc}\n".encode("utf-8", errors="replace"))
            except OSError:
                pass


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _local_ips() -> list[str]:
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips


def main():
    parser = argparse.ArgumentParser(description="Serwer TCP synchronizacji (bez HTTP).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=5500)
    args = parser.parse_args()

    with ThreadingTCPServer((args.host, args.port), SyncTCPHandler) as server:
        print("=" * 60)
        print(" SERWER TCP SYNCHRONIZACJI - START")
        print("=" * 60)
        print(f"  Katalog projektu: {PROJECT_ROOT}")
        print(f"  Nasluch:          tcp://0.0.0.0:{args.port}")
        for ip in _local_ips():
            print(f"  Adres VPN/LAN:    tcp://{ip}:{args.port}")
        print("  Protokol:         SYNC_PING / SYNC_PUTC / SYNC_GETC (bez HTTP)")
        print("  Zatrzymanie:      Ctrl+C")
        print("=" * 60)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nZatrzymywanie serwera...")


if __name__ == "__main__":
    main()
