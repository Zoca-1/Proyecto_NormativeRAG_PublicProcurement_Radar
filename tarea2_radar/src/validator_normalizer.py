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


def _buyer_department(release: dict) -> str | None:
    """Departamento de la entidad contratante.

    Verificado contra releases reales de la API de OECE: release.buyer solo
    trae id/name (sin address); la dirección completa vive en
    release.parties[], cross-referenciada por rol. El campo "department" es
    una extensión propia de OECE (ocds_department_extension) más confiable
    que "region" (en la práctica ambos suelen coincidir, pero se prioriza
    "department" por ser el campo pensado explícitamente para esto).
    """
    for party in release.get("parties", []):
        if "buyer" in party.get("roles", []):
            address = party.get("address", {})
            return address.get("department") or address.get("region")
    return None


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


def _supplier_name(release: dict) -> str | None:
    """Nombre del proveedor ganador (release.awards[0].suppliers[0]), distinto
    del comprador (buyer): el comprador es la entidad contratante, el
    proveedor es quien se adjudica el contrato."""
    awards = release.get("awards", [])
    if not awards:
        return None
    suppliers = awards[0].get("suppliers", [])
    return suppliers[0].get("name") if suppliers else None


def _flatten_release(release: dict) -> dict:
    tender = release.get("tender", {})
    return {
        "ocid": release.get("ocid"),
        "comprador": release.get("buyer", {}).get("name"),
        "proveedor": _supplier_name(release),
        "departamento_raw": _buyer_department(release),
        "monto": _award_amount(release),
        "fecha": release.get("date"),
        "categoria": tender.get("mainProcurementCategory"),
        # tender.title suele ser un código de procedimiento (p. ej.
        # "CP-ABR-3-2026-PRONIS-1"), no una descripción; tender.description
        # trae el texto real del objeto de contratación. Se prioriza
        # description y solo se cae a title si no hay description.
        "objeto_contratacion": tender.get("description") or tender.get("title"),
        "numero_postores": _number_of_tenderers(release),
    }


def _is_missing(value) -> bool:
    """None y strings vacíos cuentan como faltantes; 0 / 0.0 NO (un release en
    etapa 'planning/tender' aún sin adjudicar legítimamente trae monto=0.0;
    tratarlo como dato faltante con `not value` rechazaba ~48% de una muestra
    real solo por estar en etapa temprana, no por tener datos inválidos)."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


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

            missing = [col for col in required_cols if _is_missing(flat.get(col))]
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
    """Siempre reescribe path (incluso vacío): dejar el archivo de una
    corrida anterior con rechazos de una versión previa de la normalización
    (p. ej. antes de arreglar un bug de validación) haría parecer que el
    problema sigue ahí cuando ya no hay ninguna fila rechazada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rejected:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rejected for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejected)
