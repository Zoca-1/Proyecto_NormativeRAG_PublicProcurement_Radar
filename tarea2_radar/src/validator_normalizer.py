"""Validación y normalización territorial de releases OCDS crudos.

Mapea el esquema OCDS (release.buyer, release.tender, release.awards, ...) a
columnas planas y normaliza el departamento contra el catálogo de 25
departamentos declarado en config.yaml. Filas sin los campos requeridos o con
departamento no reconocible se registran en rejected_log_file en vez de
descartarse silenciosamente.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


def _first_region(release: dict) -> str | None:
    items = release.get("tender", {}).get("items", [])
    for item in items:
        region = item.get("deliveryAddress", {}).get("region")
        if region:
            return region
    party_region = release.get("buyer", {}).get("address", {}).get("region")
    return party_region


def _award_amount(release: dict) -> float | None:
    awards = release.get("awards", [])
    if awards:
        value = awards[0].get("value", {})
        return value.get("amount")
    tender_value = release.get("tender", {}).get("value", {})
    return tender_value.get("amount")


def _number_of_tenderers(release: dict) -> int | None:
    bids_stats = release.get("bids", {}).get("statistics", [])
    for stat in bids_stats:
        if stat.get("measure") == "bids":
            return stat.get("value")
    tender = release.get("tender", {})
    return tender.get("numberOfTenderers")


def _flatten_release(release: dict) -> dict:
    return {
        "ocid": release.get("ocid"),
        "comprador": release.get("buyer", {}).get("name"),
        "departamento_raw": _first_region(release),
        "monto": _award_amount(release),
        "fecha": release.get("date"),
        "categoria": release.get("tender", {}).get("mainProcurementCategory"),
        "objeto_contratacion": release.get("tender", {}).get("title") or release.get("tender", {}).get("description"),
        "numero_postores": _number_of_tenderers(release),
    }


def normalize_departamento(raw_value: str | None, catalogo: list[str], alias_map: dict[str, str]) -> str | None:
    if not raw_value:
        return None
    valor = raw_value.strip().upper()
    valor = alias_map.get(valor, valor)
    return valor if valor in catalogo else None


def load_and_normalize(raw_releases_path: Path, departamentos_cfg: dict, normalizacion_cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    catalogo = departamentos_cfg["lista"]
    alias_map = departamentos_cfg.get("alias_conocidos", {})
    required_cols = normalizacion_cfg["columnas_requeridas"]

    rows, rejected = [], []
    with raw_releases_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            release = json.loads(line)
            flat = _flatten_release(release)
            flat["departamento"] = normalize_departamento(flat.pop("departamento_raw"), catalogo, alias_map)

            missing = [col for col in required_cols if not flat.get(col)]
            if missing:
                rejected.append({**flat, "motivo_rechazo": f"campos faltantes: {', '.join(missing)}"})
                continue
            rows.append(flat)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", utc=True)
        df = df.dropna(subset=["monto", "fecha"])
        df["moneda"] = normalizacion_cfg["moneda_default"]
    return df, rejected


def save_rejected(rejected: list[dict], path: Path) -> None:
    if not rejected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rejected for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejected)
