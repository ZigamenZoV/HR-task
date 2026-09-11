"""Самопроверка по всему эталону: каждый из 1000 адресов подаётся как запрос.

Разметка покрывает 400 адресов из 1000. Этот прогон ловит то, что на ней не видно:
коллизии названий улиц с городами, улицы из одних цифр, адреса-близнецы в разных
городах. Строки чистые, поэтому это проверка логики, а не устойчивости к шуму.

python scripts/selfcheck.py
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from addrmatch.matcher import Matcher  # noqa: E402

MODES = {
    "тип улица дом, с городом": lambda e: (f"{e['street_type']} {e['street']} {e['house']}", e["city"]),
    "тип улица дом, без города": lambda e: (f"{e['street_type']} {e['street']} {e['house']}", ""),
    "как в чате: без типа, нижний регистр": lambda e: (f"{e['street']} {e['house']}".lower(), e["city"]),
}


def main():
    m = Matcher.load(os.path.join(ROOT, "data", "etalon.jsonl"), os.path.join(ROOT, "models", "model.json"))
    etalon = [json.loads(l) for l in open(os.path.join(ROOT, "data", "etalon.jsonl"), encoding="utf-8")]
    grp = lambda i: m.idx.dup_of.get(i, i)
    for name, make in MODES.items():
        c, fails = Counter(), []
        for e in etalon:
            raw, city = make(e)
            ids, _ = m.match(raw, city)
            if not ids:
                c["отказ"] += 1
                fails.append(("отказ", raw, city, ""))
            elif grp(ids[0]) == grp(e["etalon_id"]):
                c["верно"] += 1
            else:
                c["неверно"] += 1
                fails.append(("неверно", raw, city, m.idx.by_id[ids[0]]["address"]))
        print(f"{name:40s} верно={c['верно']} неверно={c['неверно']} отказ={c['отказ']}")
        for kind, raw, city, got in fails:
            print(f"    {kind:8s} {raw!r} city={city!r} {('-> ' + got) if got else ''}")


if __name__ == "__main__":
    main()
