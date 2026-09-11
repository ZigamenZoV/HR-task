"""Точка входа.

    python run.py --adresses adresses_labeled.jsonl --etalon etalon.jsonl [--predictions preds.jsonl]

Печатает метрики JSON в stdout. Задержка меряется на каждый вызов Matcher.match:
разбор строки, поиск улиц, признаки, скор, решение об отказе. Загрузка эталона,
построение индекса, чтение/запись файлов и прогрев в замер не входят.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from addrmatch.matcher import Matcher          # noqa: E402
from addrmatch import metrics                  # noqa: E402

WARMUP = 20


def resolve(path):
    """Принимаем и 'etalon.jsonl', и 'data/etalon.jsonl'."""
    for p in (path, os.path.join(HERE, path), os.path.join(HERE, "data", os.path.basename(path))):
        if os.path.exists(p):
            return p
    sys.exit(f"file not found: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adresses", required=True)
    ap.add_argument("--etalon", required=True)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "model.json"))
    args = ap.parse_args()

    with open(resolve(args.adresses), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    matcher = Matcher.load(resolve(args.etalon), args.model if os.path.exists(args.model) else None)

    for r in rows[:WARMUP]:                                 # прогрев: кэши rapidfuzz, аллокации
        matcher.match(r.get("raw_adress") or "", r.get("city") or "", r.get("channel"))

    preds, scores, lat = {}, {}, []
    for r in rows:
        t0 = time.perf_counter()
        cands, sc = matcher.match(r.get("raw_adress") or "", r.get("city") or "", r.get("channel"))
        lat.append((time.perf_counter() - t0) * 1000.0)
        preds[r["id"]], scores[r["id"]] = cands, sc

    if args.predictions:
        with open(args.predictions, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "candidates": preds[r["id"]],
                                    "scores": scores[r["id"]]}, ensure_ascii=False) + "\n")

    labeled = all("etalon_id" in r for r in rows)
    out = metrics.compute(rows, preds, lat)
    if not labeled:                                         # sample_input: меток нет
        for k in ("n_positive", "n_negative", "top1", "top3", "reject_precision", "reject_recall"):
            out[k] = None
        out["n_rejected"] = sum(1 for v in preds.values() if not v)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
