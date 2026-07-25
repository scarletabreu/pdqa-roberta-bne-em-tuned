#!/usr/bin/env python3
"""
Valida el formato de submission.csv antes de entregarlo.

Uso básico:
  python validate_submission.py submission.csv

Uso recomendado con archivo de IDs oficiales:
  python validate_submission.py submission.csv --expected-ids test_ids.csv

test_ids.csv debe contener una columna llamada id.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Set


REQUIRED_COLUMNS = ["id", "prediction"]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("El archivo no contiene encabezado.")
        rows = list(reader)
        return list(reader.fieldnames), rows


def load_expected_ids(path: Path) -> Set[str]:
    columns, rows = read_csv(path)
    if "id" not in columns:
        raise ValueError("El archivo de IDs oficiales no contiene la columna 'id'.")
    ids = {str(row["id"]).strip() for row in rows if str(row.get("id", "")).strip()}
    if len(ids) != len(rows):
        raise ValueError("El archivo de IDs oficiales contiene IDs vacíos o duplicados.")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida una entrega de QA.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--expected-ids", type=Path)
    parser.add_argument("--allow-empty", action="store_true",
                        help="Permite predicciones vacías; no recomendado para la entrega final.")
    args = parser.parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    try:
        columns, rows = read_csv(args.submission)
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if columns != REQUIRED_COLUMNS:
        errors.append(
            f"Las columnas deben ser exactamente {REQUIRED_COLUMNS} y en ese orden. "
            f"Se encontraron: {columns}"
        )

    ids = []
    for index, row in enumerate(rows, start=2):
        example_id = str(row.get("id", "")).strip()
        prediction = str(row.get("prediction", "")).strip()
        if not example_id:
            errors.append(f"Línea {index}: id vacío.")
        if not prediction and not args.allow_empty:
            errors.append(f"Línea {index}: prediction vacía.")
        if "\n" in prediction or "\r" in prediction:
            warnings.append(f"Línea {index}: la predicción contiene salto de línea.")
        ids.append(example_id)

    duplicates = sorted({x for x in ids if x and ids.count(x) > 1})
    if duplicates:
        errors.append(f"IDs duplicados: {duplicates[:20]}")

    if args.expected_ids:
        try:
            expected = load_expected_ids(args.expected_ids)
        except (OSError, ValueError, UnicodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        submitted = {x for x in ids if x}
        missing = sorted(expected - submitted)
        extra = sorted(submitted - expected)
        if missing:
            errors.append(f"Faltan {len(missing)} IDs. Primeros: {missing[:20]}")
        if extra:
            errors.append(f"Hay {len(extra)} IDs no oficiales. Primeros: {extra[:20]}")
        if len(rows) != len(expected):
            errors.append(
                f"Cantidad de filas incorrecta: {len(rows)}; se esperaban {len(expected)}."
            )

    print(f"Filas leídas: {len(rows)}")
    for warning in warnings:
        print(f"ADVERTENCIA: {warning}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Resultado: ENTREGA NO VÁLIDA", file=sys.stderr)
        return 1

    print("Resultado: ENTREGA VÁLIDA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
