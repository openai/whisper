#!/usr/bin/env python3
"""
Jan Sahaay AI — desktop launcher.

Serves the bundled web app on a local port and opens it in the default
browser. Packaged into JanSahaayAI.exe with PyInstaller (see the
build-jan-sahaay-exe GitHub Actions workflow).

The app pings /ping every 5 seconds; once the browser tab is closed the
pings stop and the server shuts itself down, so no stray process is left
behind even though the exe runs windowless (--noconsole).
"""

import http.server
import os
import socket
import socketserver
import sys
import threading
import time
import webbrowser

HEARTBEAT_GRACE = 90          # seconds without a ping before exiting
STARTUP_GRACE = 300           # allow slow first browser start

_last_ping = {"t": None, "started": time.time()}


def app_dir():
    if getattr(sys, "_MEIPASS", None):      # inside a PyInstaller bundle
        return os.path.join(sys._MEIPASS, "app")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=app_dir(), **kwargs)

    def do_GET(self):
        if self.path == "/ping":
            _last_ping["t"] = time.time()
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, *args):           # keep the windowless exe quiet
        pass


def watchdog(server):
    while True:
        time.sleep(5)
        now = time.time()
        if _last_ping["t"] is None:
            if now - _last_ping["started"] > STARTUP_GRACE:
                break
        elif now - _last_ping["t"] > HEARTBEAT_GRACE:
            break
    server.shutdown()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = free_port()
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=watchdog, args=(server,), daemon=True).start()
    url = "http://127.0.0.1:%d/" % port
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    print("Jan Sahaay AI running at %s (close the browser tab to exit)" % url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
