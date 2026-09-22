"""Adquisición de datos abiertos OECE (OCDS 2026). OFFLINE, idempotente y reanudable.

El estado de paginación se persiste en fetch_state_file: si el proceso se
interrumpe, una nueva ejecución retoma desde la última página confirmada en
lugar de re-descargar todo desde el inicio.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_fixed


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"next_page": 1, "completed": False, "total_releases": 0}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _make_request_fn(timeout: int, retry_attempts: int, retry_wait: int):
    @retry(stop=stop_after_attempt(retry_attempts), wait=wait_fixed(retry_wait))
    def _request(url: str, params: dict, headers: dict) -> dict:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    return _request


def fetch_releases(oece_config: dict, raw_releases_path: Path, state_path: Path) -> int:
    """Descarga paginada de releases OCDS. Retorna cantidad de releases nuevos escritos."""
    state = _load_state(state_path)
    if state.get("completed"):
        return 0

    token = os.environ.get("OECE_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request_fn = _make_request_fn(
        oece_config["request_timeout_sec"], oece_config["retry_attempts"], oece_config["retry_wait_seconds"],
    )
    url = oece_config["api_base_url"].rstrip("/") + oece_config["endpoint"]

    raw_releases_path.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    page = state["next_page"]
    max_pages = oece_config.get("max_pages")

    with raw_releases_path.open("a", encoding="utf-8") as f:
        while max_pages is None or page <= max_pages:
            params = {"page": page, "pageSize": oece_config["page_size"], "year": oece_config["ocds_year"]}
            payload = request_fn(url, params, headers)
            releases = payload.get("releases", [])
            if not releases:
                state["completed"] = True
                break

            for release in releases:
                f.write(json.dumps(release, ensure_ascii=False) + "\n")
                new_count += 1

            state["next_page"] = page + 1
            state["total_releases"] = state.get("total_releases", 0) + len(releases)
            _save_state(state_path, state)
            page += 1

    _save_state(state_path, state)
    return new_count
