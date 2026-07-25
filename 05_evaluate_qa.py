#!/usr/bin/env python3
"""
Evaluación oficial básica para Question Answering extractivo.

Acepta:
1) referencias JSON/JSONL/CSV con columnas id y answers; o id y answer.
2) predicciones CSV/JSON/JSONL con columnas id y prediction.

Ejemplos:
  python evaluate_qa.py --references validation.json --predictions submission.csv
  python evaluate_qa.py --references validation.csv --predictions submission.csv --output metrics.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import string
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SPANISH_ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}


def normalize_answer(text: Any) -> str:
    """Normaliza mayúsculas, tildes, puntuación, artículos y espacios."""
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    punctuation = string.punctuation + "¡¿“”‘’«»…"
    text = "".join(" " if ch in punctuation else ch for ch in text)
    tokens = [tok for tok in text.split() if tok not in SPANISH_ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_answers(value: Any) -> List[str]:
    """Convierte formatos SQuAD y formatos simples a lista de respuestas."""
    if value is None:
        return []
    if isinstance(value, dict):
        if "text" in value:
            texts = value["text"]
            return [str(x) for x in (texts if isinstance(texts, list) else [texts])]
        for key in ("answer", "answers", "label"):
            if key in value:
                return _extract_answers(value[key])
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            result.extend(_extract_answers(item))
        return result
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _extract_answers(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [value]
    return [str(value)]


def _read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSONL inválido en línea {line_no}: {exc}") from exc
        return records
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "records", "examples"):
                if isinstance(data.get(key), list):
                    return data[key]
            # Permite diccionario id -> respuesta/predicción
            return [{"id": k, "value": v} for k, v in data.items()]
    raise ValueError(f"Formato no soportado: {path}. Use CSV, JSON o JSONL.")


def load_references(path: Path) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {}
    for row in _read_records(path):
        example_id = str(row.get("id", "")).strip()
        if not example_id:
            raise ValueError("Existe una referencia sin id.")
        if example_id in refs:
            raise ValueError(f"id duplicado en referencias: {example_id}")
        raw = row.get("answers", row.get("answer", row.get("value")))
        answers = [x for x in _extract_answers(raw) if str(x).strip()]
        if not answers:
            raise ValueError(f"No se encontraron respuestas para id={example_id}")
        refs[example_id] = answers
    return refs


def load_predictions(path: Path) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    for row in _read_records(path):
        example_id = str(row.get("id", "")).strip()
        if not example_id:
            raise ValueError("Existe una predicción sin id.")
        if example_id in preds:
            raise ValueError(f"id duplicado en predicciones: {example_id}")
        value = row.get("prediction", row.get("answer", row.get("value", "")))
        preds[example_id] = "" if value is None else str(value)
    return preds


def evaluate(refs: Mapping[str, Sequence[str]], preds: Mapping[str, str]) -> Dict[str, Any]:
    ref_ids = set(refs)
    pred_ids = set(preds)
    missing = sorted(ref_ids - pred_ids)
    extra = sorted(pred_ids - ref_ids)

    em_total = 0.0
    f1_total = 0.0
    for example_id, answers in refs.items():
        prediction = preds.get(example_id, "")
        em_total += max(exact_match(prediction, answer) for answer in answers)
        f1_total += max(f1_score(prediction, answer) for answer in answers)

    n = len(refs)
    return {
        "exact_match": round(100.0 * em_total / n, 4) if n else 0.0,
        "f1": round(100.0 * f1_total / n, 4) if n else 0.0,
        "num_references": n,
        "num_predictions": len(preds),
        "missing_ids_count": len(missing),
        "extra_ids_count": len(extra),
        "missing_ids": missing[:20],
        "extra_ids": extra[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa predicciones de QA extractivo.")
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Archivo JSON opcional para guardar métricas.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Falla si hay identificadores faltantes o adicionales."
    )
    args = parser.parse_args()

    try:
        refs = load_references(args.references)
        preds = load_predictions(args.predictions)
        metrics = evaluate(refs, preds)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )

    if args.strict and (metrics["missing_ids_count"] or metrics["extra_ids_count"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
