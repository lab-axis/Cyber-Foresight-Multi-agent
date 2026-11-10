"""Finish the already-started Gemma download, verify it, then run the audit.

This process is started explicitly after the download has been launched. It
does not modify existing Qwen inference records or the manuscript.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import psutil

from Stage7.download_gemma4 import DESTINATION, FILENAME, SHA256, SIZE


def _alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-pid", type=int, required=True)
    args = parser.parse_args()

    model = DESTINATION / FILENAME
    marker = DESTINATION / "download_verified.sha256"

    while _alive(args.download_pid):
        print("Gemma model download is still running", flush=True)
        time.sleep(30)

    if (
        not model.is_file()
        or model.stat().st_size != SIZE
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != SHA256
    ):
        print("Completing or verifying the downloaded GGUF", flush=True)
        subprocess.run(
            [sys.executable, "-u", "-m", "Stage7.download_gemma4"],
            check=True,
        )

    print("Verified Gemma model is ready. Starting frozen-input blind audit", flush=True)
    subprocess.run(
        [sys.executable, "-u", "-m", "Stage7.mediator_ablation", "--run"],
        check=True,
    )
    print("Gemma audit and case-level analysis completed successfully", flush=True)


if __name__ == "__main__":
    main()
