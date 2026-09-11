"""Обучение весов ранжирования и модели отказа. Все цифры по кросс-валидации.

    python train.py --adresses data/adresses_labeled.jsonl --etalon data/etalon.jsonl

Пишет models/model.json. Рантайму (run.py) sklearn не нужен: модель это два вектора весов.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from addrmatch.matcher import (Matcher, CAND_FEATURES, REJECT_FEATURES, REJECT_FEATURES_ALL,
                               default_model)
from addrmatch import metrics

# Цена ошибок: неверный адрес (чужой негатив или промах по позитиву) против переспроса.
COST_WRONG, COST_REASK = 3.0, 1.0
THRESHOLD = 1.0 - COST_REASK / COST_WRONG      # принимаем, если P(top-1 верен) > 0.667


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Trainer:
    def __init__(self, matcher, rows):
        self.M, self.rows = matcher, rows
        self.cache = [matcher.candidates(r["raw_adress"], r["city"], r.get("channel"))
                      for r in rows]

    def same(self, a, b):
        d = self.M.idx.dup_of
        return a == b or (a in d and d.get(a) == d.get(b))

    # ------------------------------------------------------------ ранжирование
    def fit_cand(self, ids, C=1.0):
        Xs, ys = [], []
        for i in ids:
            gold = self.rows[i]["etalon_id"]
            if gold is None:
                continue
            meta, X, _ = self.cache[i]
            y = np.array([self.same(m, gold) for m in meta], float)
            if y.sum():
                Xs.append(X); ys.append(y)
        lr = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
        lr.fit(np.vstack(Xs), np.concatenate(ys))
        return lr.coef_[0], float(lr.intercept_[0])

    # ------------------------------------------------------------ отказ
    def reject_table(self, ids, w, b, feats):
        F, y, tops = [], [], []
        for i in ids:
            meta, X, ctx = self.cache[i]
            s = X @ w + b if len(X) else np.zeros(0)
            order = self.M._rank(meta, s)
            F.append(self.M.reject_features(meta, X, s, order, ctx, names=feats))
            gold = self.rows[i]["etalon_id"]
            y.append(gold is not None and bool(order) and self.same(meta[order[0]], gold))
            tops.append([meta[j] for j in order[:3]])
        return np.array(F), np.array(y, float), tops

    @staticmethod
    def fit_reject(F, y, C=0.3):
        mu, sd = F.mean(0), F.std(0) + 1e-6
        lr = LogisticRegression(C=C, max_iter=5000)
        lr.fit((F - mu) / sd, y)
        w = lr.coef_[0] / sd                              # складываем стандартизацию в веса
        b = float(lr.intercept_[0] - (lr.coef_[0] * mu / sd).sum())
        return w, b

    # ------------------------------------------------------------ CV
    def cv(self, cand="learned", feats=None, threshold=THRESHOLD, repeats=5, folds=5):
        feats = feats or REJECT_FEATURES
        strata = [("neg" if r["etalon_id"] is None else "pos") + r["channel"] for r in self.rows]
        results, probs = [], defaultdict(list)
        for rep in range(repeats):
            skf = StratifiedKFold(folds, shuffle=True, random_state=rep)
            preds = {}
            for tr, te in skf.split(np.zeros(len(strata)), strata):
                if cand == "learned":
                    w, b = self.fit_cand(tr)
                else:
                    dm = default_model(); w, b = np.array(dm["cand_w"]), dm["cand_b"]
                Ftr, ytr, _ = self.reject_table(tr, w, b, feats)
                rw, rb = self.fit_reject(Ftr, ytr)
                Fte, _, tops = self.reject_table(te, w, b, feats)
                p = 1 / (1 + np.exp(-(Fte @ rw + rb)))
                for i, pi, t in zip(te, p, tops):
                    preds[self.rows[i]["id"]] = t if pi >= threshold else []
                    probs[self.rows[i]["id"]].append(pi)
            results.append(metrics.compute(self.rows, preds))
        keys = ["top1", "top3", "reject_precision", "reject_recall"]
        mean = {k: round(float(np.mean([r[k] for r in results])), 4) for k in keys}
        std = {k: round(float(np.std([r[k] for r in results])), 4) for k in keys}
        return mean, std, {k: float(np.mean(v)) for k, v in probs.items()}


def wrong_rate(rows, probs, tops_by_id, same, thr):
    """Доля запросов, где бот отправит человека не туда (принятый негатив или промах)."""
    bad = 0
    for r in rows:
        if probs[r["id"]] < thr:
            continue
        t = tops_by_id[r["id"]]
        bad += int(r["etalon_id"] is None or not t or not same(t[0], r["etalon_id"]))
    return bad / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adresses", default="data/adresses_labeled.jsonl")
    ap.add_argument("--etalon", default="data/etalon.jsonl")
    ap.add_argument("--out", default="models/model.json")
    args = ap.parse_args()

    rows = load(args.adresses)
    T = Trainer(Matcher.load(args.etalon), rows)

    print("== 5x5 CV, среднее ± std по повторам")
    variants = {
        "ручные веса, отказ (итог)":        dict(cand="hand"),
        "ручные веса, отказ + city_other":  dict(cand="hand", feats=REJECT_FEATURES_ALL),
        "ручные веса, отказ по top_score":  dict(cand="hand", feats=["top_score"]),
        "обученные веса, отказ":            dict(cand="learned"),
    }
    report = {}
    for name, kw in variants.items():
        mean, std, _ = T.cv(**kw)
        report[name] = {"mean": mean, "std": std}
        print(f"{name:36s}", " ".join(f"{k}={mean[k]:.4f}±{std[k]:.4f}" for k in mean))

    print("== чувствительность к порогу (ручные веса, итоговый отказ)")
    for thr in (0.3, 0.5, THRESHOLD, 0.8, 0.9, 0.97):
        mean, _, _ = T.cv(cand="hand", threshold=thr, repeats=3)
        print(f"threshold={thr:.3f}", " ".join(f"{k}={v:.4f}" for k, v in mean.items()))

    # Финальная модель: ручные веса ранжирования (обученные не лучше и менее
    # интерпретируемы, см. CV выше) + обученная на всех данных модель отказа.
    dm = default_model()
    w, b = np.array(dm["cand_w"]), dm["cand_b"]
    F, y, _ = T.reject_table(range(len(rows)), w, b, REJECT_FEATURES)
    rw, rb = T.fit_reject(F, y)
    model = {
        "cand_features": CAND_FEATURES, "cand_w": [round(float(x), 5) for x in w], "cand_b": round(b, 5),
        "reject_features": REJECT_FEATURES, "rej_w": [round(float(x), 5) for x in rw], "rej_b": round(rb, 5),
        "threshold": round(THRESHOLD, 4),
        "cv": report,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=1)
    print("веса ранжирования:", dict(zip(CAND_FEATURES, np.round(w, 2))))
    print("веса отказа:", dict(zip(REJECT_FEATURES, np.round(rw, 2))), "b=", round(rb, 2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
