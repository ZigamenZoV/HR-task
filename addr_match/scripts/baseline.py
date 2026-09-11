"""Наивные бейзлайны: fuzzy по всей строке эталона без разбора.

python scripts/baseline.py
"""
import json
import os
import re

from rapidfuzz import fuzz, process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = [json.loads(l) for l in open(os.path.join(ROOT, "data/etalon.jsonl"), encoding="utf-8")]
A = [json.loads(l) for l in open(os.path.join(ROOT, "data/adresses_labeled.jsonl"), encoding="utf-8")]
pos = [a for a in A if a["etalon_id"]]


def norm(s):
    return re.sub(r"[^\w ]", " ", s.lower().replace("ё", "е"))


docs = [norm(f"{e['street_type']} {e['street']} {e['house']}") for e in E]
for name, scorer in (("WRatio", fuzz.WRatio), ("token_set_ratio", fuzz.token_set_ratio),
                     ("partial_ratio", fuzz.partial_ratio)):
    hit = 0
    for a in pos:
        best = process.extractOne(norm(a["raw_adress"]), docs, scorer=scorer)
        hit += E[best[2]]["etalon_id"] == a["etalon_id"]
    print(f"{name:16s} top1={hit / len(pos):.4f}")
