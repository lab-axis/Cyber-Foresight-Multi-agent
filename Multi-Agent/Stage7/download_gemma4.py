"""Pinned Gemma GGUF location and verification for the local llama.cpp server."""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = "bartowski/google_gemma-4-31B-it-GGUF"
REVISION = "619aca7589c72be65dd968de8602ad23d68b5c14"
FILENAME = "google_gemma-4-31B-it-Q6_K_L.gguf"
SIZE = 27_068_033_376
SHA256 = "f77819db7b83a145425084048efec024b5efbc175bc0fdc4bc4bfe7d4b5802df"
DESTINATION = Path(__file__).resolve().parents[4] / "LLMs" / "Gemma4-31B-it-Q6-K-L" / "Model"


def main() -> None:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=REPO, filename=FILENAME,
        revision=REVISION, local_dir=DESTINATION,
    ))
    if path.stat().st_size != SIZE:
        raise ValueError(f"Unexpected Gemma GGUF size: {path.stat().st_size}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("Gemma GGUF SHA256 mismatch")

    (DESTINATION / "download_verified.sha256").write_text(SHA256 + "\n", encoding="utf-8")
    print(f"Verified {path}")


if __name__ == "__main__":
    main()
