"""Download the official statute PDFs referenced in legal_knowledge.json into data/statutes so
statutory excerpts can be verified. India Code is sometimes unavailable (HTTP 504); if a
download fails, download the PDF manually from the listed URL and save it under the listed
file name. Responses that are not real PDFs are discarded.

    python -m scripts.fetch_statutes
    python -m scripts.seed_graph          # then rebuild the graph to record verification
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.knowledge.statutes import load_knowledge, verify_statutes  # noqa: E402

if __name__ == "__main__":
    target = get_settings().statutes_dir
    target.mkdir(parents=True, exist_ok=True)
    for statute in load_knowledge()["statutes"]:
        url, name = statute["source_url"], statute.get("local_file")
        if not name or (target / name).exists():
            continue
        try:
            response = httpx.get(url, timeout=120, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code == 200 and response.content.startswith(b"%PDF"):
                (target / name).write_bytes(response.content)
                print(f"downloaded {name}")
            else:
                print(f"failed {name}: HTTP {response.status_code} or not a PDF; download manually from {url}")
        except httpx.HTTPError as exc:
            print(f"failed {name}: {type(exc).__name__}; download manually from {url}")
    print(json.dumps({k: v["status"] for k, v in verify_statutes().items()}, indent=1))
