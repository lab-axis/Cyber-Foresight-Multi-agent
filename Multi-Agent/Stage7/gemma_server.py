"""Temporarily swap an idle local Qwen server for Gemma and restore Qwen.

The aggregate context, KV types, batch sizes and GPU offloading remain those
of the original launcher. Gemma uses one 131072-token slot instead of Qwen's
two 65536-token slots. Qwen-only MTP/reasoning flags are omitted.
"""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import requests

from Stage7.download_gemma4 import DESTINATION, FILENAME, SHA256, SIZE
from Stage7.forecast_link_counterfactual import _ensure_idle, _ready, _server
from Stage7.mediator_ablation import OUTPUT, ROOT


MODEL = DESTINATION / FILENAME
MODEL_ALIAS = "Gemma4-31B-it-Q6_K_L"
SERVER = "http://127.0.0.1:8080"

QWEN_ONLY_VALUE_FLAGS = {
    "--spec-draft-type-k",
    "--spec-draft-type-v",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-n-min",
    "--spec-draft-p-min",
    "--reasoning",
    "--reasoning-effort",
    "--reasoning-budget",
    "--reasoning-format",
}
QWEN_ONLY_SWITCHES = {"--reasoning-preserve"}


def _gemma_command(original: list[str]) -> list[str]:
    if not MODEL.is_file() or MODEL.stat().st_size != SIZE:
        raise FileNotFoundError(
            f"Verified Gemma GGUF download is incomplete: {MODEL}. "
            "See Results/Stage7/mediator_ablation/gemma_download_stdout.log"
        )

    verified = DESTINATION / "download_verified.sha256"
    if not verified.is_file() or verified.read_text(encoding="utf-8").strip() != SHA256:
        raise ValueError("Gemma GGUF has no matching full SHA256 verification marker")

    result = []
    index = 0

    while index < len(original):
        option = original[index]
        if option in QWEN_ONLY_SWITCHES:
            index += 1
            continue
        if option in QWEN_ONLY_VALUE_FLAGS:
            index += 2
            continue
        if option in {"--model", "--alias"}:
            result.extend([
                option,
                str(MODEL) if option == "--model" else MODEL_ALIAS,
            ])
            index += 2
            continue

        result.append(option)
        index += 1

    for flag, value in {
        "--ctx-size": "131072",
        "--batch-size": "2048",
        "--ubatch-size": "512",
        "--cache-type-k": "q8_0",
        "--cache-type-v": "q8_0",
        "--n-gpu-layers": "all",
    }.items():
        if result.count(flag) != 1 or result[result.index(flag) + 1] != value:
            raise ValueError(f"Gemma runtime setting differs from the Qwen baseline: {flag}")

    if result.count("--parallel") != 1 or result[result.index("--parallel") + 1] != "2":
        raise ValueError("Expected the original Qwen two-slot launcher")

    result[result.index("--parallel") + 1] = "1"

    return result


def _stop(process) -> None:
    if process.is_running():
        process.terminate()
        process.wait(timeout=30)


def _wait_idle_after_requests() -> None:
    # An HTTP read timeout may precede llama.cpp releasing the cancelled slot.
    for attempt in range(30):
        try:
            _ensure_idle()
            return
        except RuntimeError:
            if attempt == 29:
                raise
            time.sleep(1)


def _launch(command: list[str], label: str) -> subprocess.Popen:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logfile = OUTPUT / f"server_{label}.log"

    with logfile.open("ab") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT / "Multi-Agent",
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    try:
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(f"{label} server exited while loading; inspect {logfile}")

            try:
                health = requests.get(f"{SERVER}/health", timeout=2)
                slots = requests.get(f"{SERVER}/slots", timeout=2).json()
                models = requests.get(f"{SERVER}/v1/models", timeout=2).json()

                if not health.ok:
                    raise RuntimeError("Server still loading")
                expected_slots = 1 if label == "gemma" else 2
                expected_context = 131072 // expected_slots
                if len(slots) != expected_slots or any(
                    int(slot.get("n_ctx", -1)) != expected_context for slot in slots
                ):
                    raise RuntimeError("Server context or slot settings changed")
                if label == "gemma":
                    ids = {entry.get("id") for entry in models.get("data", [])}
                    if MODEL_ALIAS not in ids:
                        raise RuntimeError("Gemma server reported an unexpected model")
                else:
                    _ready(2)

                return process
            except (requests.RequestException, ValueError, RuntimeError, KeyError, TypeError):
                time.sleep(1)

        raise RuntimeError(f"{label} server was not ready with unchanged context; inspect {logfile}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        raise


@contextmanager
def use_gemma():
    process, original = _server()
    _ready(2)
    target = _gemma_command(original)
    _ensure_idle()

    gemma = None
    print("Switching idle Qwen server to Gemma 4 31B Q6_K_L (131072 total context, 1 slot)", flush=True)
    _stop(process)

    try:
        gemma = _launch(target, "gemma")
        yield
    finally:
        if gemma is not None and gemma.poll() is None:
            _wait_idle_after_requests()
            gemma.terminate()
            gemma.wait(timeout=30)

        print("Restoring the original Qwen server launch configuration", flush=True)
        _launch(original, "qwen_restored")
