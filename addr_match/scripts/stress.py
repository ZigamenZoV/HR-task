"""Стресс-тесты: насколько решение опирается на поле city и на разбор дома.

python scripts/stress.py

Часть 1. Портим поле city у позитивов (негативы не трогаем).
Часть 2. Имитируем сбой разбора дома у позитивов: дом потерян совсем или
распознан как номер, которого нет в эталоне. Смотрим, куда уходит строка:
в верный ответ, в неверный или в отказ. Неверный ответ здесь хуже отказа.
"""
import json
import os
import random
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from addrmatch.matcher import Matcher   # noqa: E402
from addrmatch import metrics           # noqa: E402

M = Matcher.load(os.path.join(ROOT, "data/etalon.jsonl"), os.path.join(ROOT, "models/model.json"))
rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/adresses_labeled.jsonl"), encoding="utf-8")]
cities = M.idx.cities
rnd = random.Random(0)


def run(transform, name):
    preds = {r["id"]: M.match(r["raw_adress"], transform(r), r["channel"])[0] for r in rows}
    m = metrics.compute(rows, preds)
    pos_rej = sum(1 for r in rows if r["etalon_id"] and not preds[r["id"]])
    print(f"{name:44s} top1={m['top1']:.4f} top3={m['top3']:.4f} "
          f"rej_prec={m['reject_precision']:.4f} rej_rec={m['reject_recall']:.4f} pos_rejected={pos_rej}")


def wrong_city(r):
    if not r["etalon_id"]:
        return r["city"]
    gold = M.idx.by_id[r["etalon_id"]]["city"]
    return rnd.choice([c for c in cities if c != gold])


print("== поле city")
run(lambda r: r["city"], "как есть")
run(lambda r: "", "город стёрт у всех")
run(wrong_city, "у позитивов чужой город эталона")
run(lambda r: "Саранск" if r["etalon_id"] else r["city"], "у позитивов город вне эталона (Саранск)")


# ---------------------------------------------------------------- сбой разбора дома
def house_failure(spoil, name):
    parse = M.parser.parse
    M.parser.parse = lambda raw: spoil(parse(raw))
    ok = wrong = rej = 0
    by_n = {"1 дом на улице": [0, 0, 0], ">1 дома на улице": [0, 0, 0]}
    try:
        for r in rows:
            gold = r["etalon_id"]
            if gold is None:
                continue
            cands, _ = M.match(r["raw_adress"], r["city"], r["channel"])
            g = M.idx.dup_of.get(gold, gold)
            e = M.idx.by_id[gold]
            n_houses = sum(1 for x in M.idx.rows if (x["city"], x["street_type"], x["street"])
                           == (e["city"], e["street_type"], e["street"]))
            bucket = by_n["1 дом на улице" if n_houses == 1 else ">1 дома на улице"]
            if not cands:
                rej += 1; bucket[2] += 1
            elif M.idx.dup_of.get(cands[0], cands[0]) == g:
                ok += 1; bucket[0] += 1
            else:
                wrong += 1; bucket[1] += 1
    finally:
        M.parser.parse = parse
    print(f"{name:44s} верно={ok} неверно={wrong} отказ={rej}  "
          + "  ".join(f"[{k}: {v[0]}/{v[1]}/{v[2]}]" for k, v in by_n.items()))


print("== сбой разбора дома у позитивов (верно / неверно / отказ)")
house_failure(lambda q: q, "как есть")
house_failure(lambda q: replace(q, house="", house_alts=()), "дом потерян")
house_failure(lambda q: replace(q, house="9999", house_alts=()), "дом распознан номером, которого нет")
