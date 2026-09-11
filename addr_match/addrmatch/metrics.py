"""Метрики ровно в формулировке задания."""
import numpy as np


def compute(rows, preds, latencies_ms=None):
    """rows: входные строки с etalon_id; preds: {id: [etalon_id, ...]} (пусто = отказ)."""
    n_pos = n_neg = t1 = t3 = rej = rej_neg = 0
    for r in rows:
        cands = preds.get(r["id"], [])
        gold = r.get("etalon_id")
        if not cands:
            rej += 1
        if gold is None:
            n_neg += 1
            rej_neg += int(not cands)
        else:
            n_pos += 1
            t1 += int(bool(cands) and cands[0] == gold)
            t3 += int(gold in cands[:3])
    out = {
        "n": len(rows), "n_positive": n_pos, "n_negative": n_neg,
        "top1": round(t1 / n_pos, 4) if n_pos else 0.0,
        "top3": round(t3 / n_pos, 4) if n_pos else 0.0,
        "reject_precision": round(rej_neg / rej, 4) if rej else 0.0,
        "reject_recall": round(rej_neg / n_neg, 4) if n_neg else 0.0,
    }
    if latencies_ms is not None and len(latencies_ms):
        out["ms_per_request_p50"] = round(float(np.percentile(latencies_ms, 50)), 3)
        out["ms_per_request_p95"] = round(float(np.percentile(latencies_ms, 95)), 3)
    return out
