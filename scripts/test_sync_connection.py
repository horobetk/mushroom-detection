#!/usr/bin/env python3
"""
Test polaczenia z serwerem synchronizacji (uruchom na LAPTOPIE).

Autor: Kiril Horobets, Politechnika Warszawska, 2026
"""

import argparse
import socket
import sys
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def test_tcp(host: str, port: int) -> bool:
    """Sprawdza, czy port TCP jest osiagalny (bez HTTP)."""
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError as exc:
        print(f"[TCP]  FAIL - {exc}")
        return False


def test_http(host: str, port: int) -> bool:
    """Sprawdza GET /ping przez urllib (tak samo jak local_watcher)."""
    url = f"http://{host}:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip()
        ok = body == "pong"
        print(f"[HTTP] {'OK' if ok else 'FAIL'} - odpowiedz: {body!r}")
        return ok
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"[HTTP] FAIL - {exc}")
        return False


def test_tcp(host: str, port: int) -> bool:
    """Sprawdza protokol SYNC_PING (bez HTTP)."""
    try:
        with socket.create_connection((host, port), timeout=10) as conn:
            conn.sendall(b"SYNC_PING\n")
            resp = conn.recv(64)
        ok = resp.startswith(b"SYNC_PONG")
        print(f"[TCP+] {'OK' if ok else 'FAIL'} - odpowiedz: {resp!r}")
        return ok
    except OSError as exc:
        print(f"[TCP+] FAIL - {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test polaczenia sync z zdalnym PC.")
    parser.add_argument("--host", "-H", required=True, help="IP zdalnego PC (np. 10.44.25.85)")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port HTTP (domyslnie 8080)")
    parser.add_argument("--tcp-port", type=int, default=5500, help="Port TCP sync (domyslnie 5500)")
    parser.add_argument("--mode", choices=["http", "tcp", "both"], default="both")
    args = parser.parse_args()

    print(f"Test polaczenia -> {args.host}\n")

    tcp_ok = False
    http_ok = False

    if args.mode in ("tcp", "both"):
        print(f"--- Protokol TCP (port {args.tcp_port}) ---")
        try:
            with socket.create_connection((args.host, args.tcp_port), timeout=5):
                print("[TCP]  OK - port otwarty")
        except OSError as exc:
            print(f"[TCP]  FAIL - {exc}")
        else:
            tcp_ok = test_tcp(args.host, args.tcp_port)
        print()

    if args.mode in ("http", "both"):
        print(f"--- Protokol HTTP (port {args.port}) ---")
        tcp_open = False
        try:
            with socket.create_connection((args.host, args.port), timeout=5):
                tcp_open = True
                print("[TCP]  OK - port otwarty")
        except OSError as exc:
            print(f"[TCP]  FAIL - {exc}")
        if tcp_open:
            http_ok = test_http(args.host, args.port)
        print()

    if args.mode == "tcp":
        ok = tcp_ok
    elif args.mode == "http":
        ok = http_ok
    else:
        ok = tcp_ok or http_ok

    if ok:
        if tcp_ok:
            print("GOTOWE (TCP). Uruchom: python scripts\\local_watcher_tcp.py --host ...")
        if http_ok:
            print("GOTOWE (HTTP). Uruchom: python scripts\\local_watcher.py --host ...")
        sys.exit(0)

    if args.mode == "both" and not tcp_ok and not http_ok:
        print("Oba protokoly nie dzialaja.")
        print("1) Uruchom remote_receiver_tcp.py na zdalnym PC")
        print("2) Sprawdz firewall Windows (Zezwol dla Python - sieci prywatne)")
        print("3) Na zdalnym PC przetestuj lokalnie:")
        print("   python scripts\\test_sync_connection.py --host 127.0.0.1 --mode tcp")
    sys.exit(1)


if __name__ == "__main__":
    main()
