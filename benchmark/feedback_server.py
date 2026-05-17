"""Feedback-enabled HTTP server for the live benchmark dashboard + playground.

Replaces `python -m http.server` with:
  - Static file serving from benchmark/
  - POST /api/feedback — accepts {case_id, vote, comment}, appends to feedback.json
  - GET  /api/feedback — returns current feedback entries
  - GET  /api/playground/images — returns list of benchmark image filenames
  - POST /api/playground/run — runs the pipeline on an image with a prompt

Usage:
  python benchmark/feedback_server.py [--port 8899]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

log = logging.getLogger(__name__)

_file_lock = threading.Lock()

BENCHMARK_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = BENCHMARK_DIR / "results" / "feedback.json"
PLAYGROUND_DIR = BENCHMARK_DIR / "results" / "playground"
PLAYGROUND_HISTORY = BENCHMARK_DIR / "results" / "playground_history.json"
IMAGES_DIR = BENCHMARK_DIR / "images"


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
        elif self.path == "/api/playground/run":
            self._handle_playground_run()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/api/feedback":
            self._handle_feedback_get()
        elif self.path == "/api/playground/images":
            self._handle_playground_images()
        elif self.path.startswith("/api/playground/history"):
            self._handle_playground_history()
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

        with _file_lock:
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

        self._json_response({"ok": True})

    def _handle_feedback_get(self):
        entries = _load_feedback()
        self._json_response(entries)

    def _handle_playground_images(self):
        images = sorted(
            f.name for f in IMAGES_DIR.glob("*.jpg")
            if not f.name.startswith("_")
        )
        self._json_response(images)

    def _handle_playground_history(self):
        if PLAYGROUND_HISTORY.exists():
            try:
                entries = json.loads(PLAYGROUND_HISTORY.read_text())
            except (json.JSONDecodeError, OSError):
                entries = []
        else:
            entries = []
        self._json_response(entries)

    def _handle_playground_run(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        image_name = data.get("image", "")
        prompt = data.get("prompt", "").strip()
        if not image_name or not prompt:
            self.send_error(400, "Missing image or prompt")
            return

        image_path = IMAGES_DIR / image_name
        if not image_path.exists():
            self.send_error(404, f"Image not found: {image_name}")
            return

        run_id = datetime.now().strftime("%H%M%S_%f")[:11]
        stem = f"{image_path.stem}__{run_id}"
        PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PLAYGROUND_DIR / f"{stem}.png"

        log.info("[Playground] Running: %s — \"%s\"", image_name, prompt)

        try:
            from guided_remove_background.remove_bg import remove_bg
            result = remove_bg(
                image_path, [prompt], output_path,
                mode="guided", save_steps=True,
            )
            step_urls = {}
            for key, path_str in result.step_images.items():
                rel = os.path.relpath(path_str, BENCHMARK_DIR)
                step_urls[key] = rel

            preview_rel = os.path.relpath(str(result.preview_path), BENCHMARK_DIR)
            vlm = result.vlm_decompose

            judge_data = []
            for v in result.judge_verdicts:
                judge_data.append({
                    "passed": v.passed,
                    "error": v.error,
                    "issues": [{"type": i.type, "description": i.description, "fix": i.fix}
                               for i in v.issues],
                })

            resp = {
                "ok": True,
                "preview_url": preview_rel,
                "vlm_mode": vlm.mode if vlm else None,
                "vlm_targets": vlm.targets if vlm else [],
                "sam_scores": result.sam_scores,
                "elapsed_s": round(result.elapsed_s, 1),
                "step_images": step_urls,
                "image": image_name,
                "prompt": prompt,
                "judge_verdicts": judge_data,
            }
            log.info("[Playground] Done in %.1fs — mode=%s targets=%s",
                     result.elapsed_s, vlm.mode if vlm else "?", vlm.targets if vlm else [])

            with _file_lock:
                try:
                    history = json.loads(PLAYGROUND_HISTORY.read_text()) if PLAYGROUND_HISTORY.exists() else []
                except (json.JSONDecodeError, OSError):
                    history = []
                resp["timestamp"] = datetime.now(timezone.utc).isoformat()
                history.append(resp)
                PLAYGROUND_HISTORY.parent.mkdir(parents=True, exist_ok=True)
                PLAYGROUND_HISTORY.write_text(json.dumps(history, indent=2))

            self._json_response(resp)

        except Exception as e:
            log.error("[Playground] Pipeline error: %s", e, exc_info=True)
            self._json_response({"ok": False, "error": str(e)}, status=500)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        try:
            msg = str(args[0]) if args else ""
            if "/api/" in msg:
                log.info(format, *args)
        except Exception:
            pass


def _load_env():
    """Load .env file from project root if it exists."""
    env_path = BENCHMARK_DIR.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if val and key not in os.environ:
            os.environ[key] = val


def main():
    parser = argparse.ArgumentParser(description="Benchmark feedback server")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    _load_env()
    os.chdir(BENCHMARK_DIR)

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("", args.port), FeedbackHandler)
    log.info("Dashboard:   http://localhost:%d/live.html", args.port)
    log.info("Playground:  http://localhost:%d/playground.html", args.port)
    log.info("Feedback API: http://localhost:%d/api/feedback", args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
