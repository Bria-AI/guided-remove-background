"""Feedback-enabled HTTP server for the live benchmark dashboard.

Replaces `python -m http.server` with:
  - Static file serving from benchmark/
  - POST /api/feedback — accepts {case_id, vote, comment}, appends to feedback.json
  - GET  /api/feedback — returns current feedback entries

Usage:
  python benchmark/feedback_server.py [--port 8899]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

log = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = BENCHMARK_DIR / "results" / "feedback.json"


def _load_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    try:
        return json.loads(FEEDBACK_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_feedback(entries: list[dict]) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(json.dumps(entries, indent=2))


class FeedbackHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BENCHMARK_DIR), **kwargs)

    def do_POST(self):
        if self.path == "/api/feedback":
            self._handle_feedback_post()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/api/feedback":
            self._handle_feedback_get()
        else:
            super().do_GET()

    def _handle_feedback_post(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        case_id = data.get("case_id", "")
        vote = data.get("vote", "")
        comment = data.get("comment", "")

        if not case_id or vote not in ("like", "dislike"):
            self.send_error(400, "Missing case_id or invalid vote")
            return

        entries = _load_feedback()
        entries.append({
            "case_id": case_id,
            "vote": vote,
            "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved": False,
        })
        _save_feedback(entries)

        log.info("[Feedback] %s %s — %s", vote.upper(), case_id, comment or "(no comment)")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode())

    def _handle_feedback_get(self):
        entries = _load_feedback()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(entries).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        if "/api/" in (args[0] if args else ""):
            log.info(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Benchmark feedback server")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    os.chdir(BENCHMARK_DIR)
    server = HTTPServer(("", args.port), FeedbackHandler)
    log.info("Serving benchmark at http://localhost:%d/live.html", args.port)
    log.info("Feedback API at http://localhost:%d/api/feedback", args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
